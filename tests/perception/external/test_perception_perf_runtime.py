"""Runtime & performance regression tests for VGGT-height and SAM 3.1 perception services."""
from __future__ import annotations

NI_RUA_TEST_TIER = "component"

import base64
import errno
import io
import json
import math
import socket
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

import numpy as np
from PIL import Image

from services.vggt_height import vggt_height_service as vggt_svc
from services.vggt_height.sam31 import sam31_service as sam31_svc


def _reference_fit_table(X1, conf, cam_z, support_z, rng, thr_rel=0.004, iters=400):
    """Exact pre-vectorization reference RANSAC+SVD implementation for bit-level parity testing."""
    dz_support = float(support_z - cam_z)
    if not math.isfinite(dz_support) or abs(dz_support) <= 1e-6:
        raise ValueError("invalid_support_scale")
    valid = (conf > np.percentile(conf, 10)) & ((X1[..., 2] * dz_support) > 0)
    idx = np.flatnonzero(valid.reshape(-1))
    if len(idx) < 3:
        raise ValueError("no_horizontal_support_plane")
    pts = X1.reshape(-1, 3)
    thr = thr_rel * np.median(np.linalg.norm(pts[idx], axis=1))
    sub = rng.choice(idx, size=min(len(idx), 20000), replace=False)
    best, best_n = None, -1
    for _ in range(iters):
        a, b, c = pts[rng.choice(sub, 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n = n / nn
        if n[2] < 0:
            n = -n
        if n[2] < math.cos(math.radians(25)):
            continue
        d = -n @ a
        cnt = int((np.abs(pts[sub] @ n + d) < thr).sum())
        if cnt > best_n:
            best, best_n = (n, d), cnt
    if best is None:
        raise ValueError("no_horizontal_support_plane")
    n, d = best
    inl = (np.abs(pts @ n + d) < thr) & valid.reshape(-1)
    Q = pts[inl]
    cen = Q.mean(0)
    _, _, vt = np.linalg.svd(Q - cen, full_matrices=False)
    n = vt[2] if vt[2][2] > 0 else -vt[2]
    d = -n @ cen
    inl = (np.abs(pts @ n + d) < thr) & valid.reshape(-1)
    zs = pts[inl, 2]
    s = float(np.median((support_z - cam_z) / zs))
    if not (math.isfinite(s) and s > 0):
        raise ValueError("invalid_support_scale")
    tilt = math.degrees(math.acos(min(1.0, abs(n[2]))))
    resid = np.abs(cam_z + s * zs - support_z)
    return s, dict(
        tilt_deg=tilt,
        residual_p50_m=float(np.median(resid)),
        residual_p90_m=float(np.percentile(resid, 90)),
        inlier_frac=float(inl.mean()),
    )


class TestPerceptionServicesPerfRuntime(unittest.TestCase):
    def tearDown(self) -> None:
        vggt_svc._sam31_close_all_conns()

    def test_fit_table_batched_gemm_eigh_exact_parity(self) -> None:
        rng_data = np.random.default_rng(123)
        H, W = 128, 128
        xs, ys = np.meshgrid(np.linspace(-0.5, 0.5, W), np.linspace(-0.5, 0.5, H))
        X1 = np.stack([xs, ys, -0.7 + rng_data.normal(0.0, 0.001, (H, W))], axis=-1).astype(np.float64)
        # Add 20% off-table object outliers.
        X1[:25, :60, 2] += 0.15
        conf = rng_data.uniform(0.5, 2.0, (H, W)).astype(np.float32)

        s_ref, info_ref = _reference_fit_table(X1, conf, cam_z=1.5, support_z=0.8, rng=np.random.default_rng(0))
        s_opt, info_opt = vggt_svc.fit_table(X1, conf, cam_z=1.5, support_z=0.8, rng=np.random.default_rng(0))

        self.assertAlmostEqual(s_opt, s_ref, delta=1e-12)
        self.assertAlmostEqual(info_opt["tilt_deg"], info_ref["tilt_deg"], delta=1e-9)
        self.assertAlmostEqual(info_opt["residual_p50_m"], info_ref["residual_p50_m"], delta=1e-12)
        self.assertAlmostEqual(info_opt["residual_p90_m"], info_ref["residual_p90_m"], delta=1e-12)
        self.assertAlmostEqual(info_opt["inlier_frac"], info_ref["inlier_frac"], delta=1e-12)

    def test_fail_fast_port_bind_before_model_load(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
            occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            occupied.bind(("127.0.0.1", 0))
            occupied.listen(1)
            busy_port = occupied.getsockname()[1]

            with mock.patch.object(sam31_svc, "load_runtime_model", side_effect=AssertionError("must not load")):
                with self.assertRaises(OSError) as sam_ctx:
                    sam31_svc.main(["--host", "127.0.0.1", "--port", str(busy_port)])
                self.assertEqual(sam_ctx.exception.errno, errno.EADDRINUSE)

            with mock.patch.object(vggt_svc, "load_runtime_model", side_effect=AssertionError("must not load")):
                with self.assertRaises(OSError) as vggt_ctx:
                    vggt_svc.main(["--host", "127.0.0.1", "--port", str(busy_port)])
                self.assertEqual(vggt_ctx.exception.errno, errno.EADDRINUSE)

    def test_sam31_cache_hit_skips_cuda_sync_and_runs_inference_mode(self) -> None:
        buf = io.BytesIO()
        Image.new("RGB", (32, 32), (10, 20, 30)).save(buf, format="PNG")
        png_bytes = buf.getvalue()
        png_b64 = base64.b64encode(png_bytes).decode("ascii")

        in_inference_mode: list[bool] = []
        sync_calls = {"count": 0}

        class _FakeProc:
            def set_image(self, _img):
                if sam31_svc.torch is not None:
                    in_inference_mode.append(bool(sam31_svc.torch.is_inference_mode_enabled()))
                return {"backbone_out": {"feat": [1, 2, 3]}}

            def set_text_prompt(self, prompt, state):
                if sam31_svc.torch is not None:
                    in_inference_mode.append(bool(sam31_svc.torch.is_inference_mode_enabled()))
                return {
                    **state,
                    "scores": np.array([0.9], dtype=np.float32),
                    "masks": np.ones((1, 1, 32, 32), dtype=bool),
                    "boxes": np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
                }

        orig_cache = dict(sam31_svc.CACHE)
        sam31_svc.CACHE.clear()
        try:
            with (
                mock.patch.object(sam31_svc, "PROC", _FakeProc()),
                mock.patch.object(
                    sam31_svc,
                    "_sync_cuda",
                    side_effect=lambda: sync_calls.__setitem__("count", sync_calls["count"] + 1),
                ),
            ):
                r1 = sam31_svc.segment({"image_b64": png_b64, "phrases": ["cup"]})
                self.assertFalse(r1["cache_hit"])
                self.assertEqual(sync_calls["count"], 1)

                r2 = sam31_svc.segment({"image_b64": png_b64, "phrases": ["cup"]})
                self.assertTrue(r2["cache_hit"])
                self.assertEqual(sync_calls["count"], 1)

            if sam31_svc.torch is not None:
                self.assertTrue(in_inference_mode)
                self.assertTrue(all(in_inference_mode))
        finally:
            sam31_svc.CACHE.clear()
            sam31_svc.CACHE.update(orig_cache)

    def test_sam31_post_json_keepalive_reuse_and_stale_socket_retry(self) -> None:
        client_ports: list[int] = []
        drop_once = {"armed": False}

        class _KeepAliveHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *a):
                pass

            def do_POST(self):
                cl = int(self.headers.get("Content-Length", 0))
                if drop_once["armed"]:
                    drop_once["armed"] = False
                    self.rfile.read(cl)
                    self.close_connection = True
                    return
                raw = self.rfile.read(cl)
                req = json.loads(raw) if raw else {}
                client_ports.append(int(self.client_address[1]))
                body = json.dumps({"status": "ok", "echo": req.get("v"), "port": self.client_address[1]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        srv = ThreadingHTTPServer(("127.0.0.1", 0), _KeepAliveHandler)
        srv.daemon_threads = True
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        orig_sam31 = vggt_svc.args.sam31
        try:
            vggt_svc.args.sam31 = f"http://127.0.0.1:{port}"
            a1 = vggt_svc._sam31_post_json("/segment", {"v": 1}, 5.0)
            a2 = vggt_svc._sam31_post_json("/segment", {"v": 2}, 5.0)
            self.assertEqual(a1["echo"], 1)
            self.assertEqual(a2["echo"], 2)
            # Reused HTTP/1.1 Keep-Alive TCP connection keeps the exact same ephemeral client port.
            self.assertEqual(client_ports[0], client_ports[1])

            # Server drops the pooled keep-alive socket mid-reuse: client retries once transparently.
            drop_once["armed"] = True
            a3 = vggt_svc._sam31_post_json("/segment", {"v": 3}, 5.0)
            self.assertEqual(a3["echo"], 3)
            self.assertEqual(len(client_ports), 3)
            self.assertNotEqual(client_ports[1], client_ports[2])
        finally:
            vggt_svc._sam31_close_all_conns()
            vggt_svc.args.sam31 = orig_sam31
            srv.shutdown()
            srv.server_close()
            t.join(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
