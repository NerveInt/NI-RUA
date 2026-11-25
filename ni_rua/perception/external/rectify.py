"""Lens rectification for the external VGGT service, which assumes an ideal pinhole camera (D = 0).

A camera whose calibration has non-zero OpenCV distortion D (for example the SO-101 USB cameras) is sent as the image
undistorted onto the same K (cv2.undistort; JPEG quality 95 when the source is JPEG, else lossless PNG), so the pinhole
K and T_world_camera stay valid. Pixels
cross the boundary in both directions: request pixels (original image) -> rectified image, and service pixels
(label_px, contour_px, bbox_px, point-cloud uv) -> original image, so every pixel the model reads or writes stays in
the original observation image. A camera with D = 0 passes through untouched (same bytes, same pixels).
"""
from __future__ import annotations

import collections
import hashlib
import threading

import cv2
import numpy as np

MIN_COEFFICIENT = 1e-12
JPEG_MAGIC = b"\xff\xd8\xff"
RECTIFIED_JPEG_QUALITY = 95
_RECTIFIED_CACHE_MAX = 8


class Rectified:
    """Image bytes and pixel mappings of one camera for the pinhole service."""

    def __init__(self, camera, image_bytes):
        self.K = np.asarray(camera["K"], dtype=np.float64)
        self.D = np.asarray(camera["D"], dtype=np.float64).reshape(-1)
        self.active = bool(self.D.size and np.any(np.abs(self.D) > MIN_COEFFICIENT))
        self.image_bytes = bytes(image_bytes) if isinstance(image_bytes, (bytearray, memoryview)) else image_bytes
        self._sha256: str | None = None
        if self.active:
            raw_bytes = self.image_bytes
            image = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("invalid_rgb")
            undistorted = cv2.undistort(image, self.K, self.D)
            is_jpeg = isinstance(raw_bytes, bytes) and raw_bytes[:3] == JPEG_MAGIC
            if is_jpeg:
                ok, encoded = cv2.imencode(".jpg", undistorted, [cv2.IMWRITE_JPEG_QUALITY, RECTIFIED_JPEG_QUALITY])
            else:
                ok, encoded = cv2.imencode(".png", undistorted)
            if not ok:
                raise ValueError("rectified_image_encoding_failed")
            self.image_bytes = encoded.tobytes()

    @property
    def sha256(self) -> str:
        digest = self._sha256
        if digest is None:
            digest = hashlib.sha256(self.image_bytes).hexdigest()
            self._sha256 = digest
        return digest

    def to_rectified(self, pixels):
        """[[u,v],...] in the original image -> the same rays as pixels of the rectified image."""
        points = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        if not self.active or not len(points):
            return points
        return cv2.undistortPoints(points.reshape(-1, 1, 2), self.K, self.D, P=self.K).reshape(-1, 2)

    def to_original(self, pixels):
        """[[u,v],...] in the rectified image -> pixels of the original (distorted) image."""
        points = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        if not self.active or not len(points):
            return points
        rays = np.column_stack([(points[:, 0] - self.K[0, 2]) / self.K[0, 0],
                                (points[:, 1] - self.K[1, 2]) / self.K[1, 1], np.ones(len(points))])
        zero = np.zeros(3)
        return cv2.projectPoints(rays, zero, zero, self.K, self.D)[0].reshape(-1, 2)

    def map_objects(self, objects):
        """Service object rows with label_px / centroid_px / contour_px / bbox_px moved to the original image."""
        if not self.active:
            return objects
        for item in objects:
            for key in ("label_px", "centroid_px"):
                if _pair(item.get(key)):
                    item[key] = [int(round(v)) for v in self.to_original([item[key]])[0]]
            contour = item.get("contour_px")
            if contour:
                item["contour_px"] = np.rint(self.to_original(contour)).astype(int).tolist()
            box = item.get("bbox_px")
            if isinstance(box, (list, tuple)) and len(box) == 4:
                corners = self.to_original([[box[0], box[1]], [box[2], box[1]], [box[0], box[3]], [box[2], box[3]]])
                low, high = corners.min(axis=0), corners.max(axis=0)
                item["bbox_px"] = [int(np.floor(low[0])), int(np.floor(low[1])), int(np.ceil(high[0])), int(np.ceil(high[1]))]
        return objects


def _pair(value):
    return isinstance(value, (list, tuple)) and len(value) == 2


def map_npz_uv(data, rectified):
    """scene3d npz bytes with the uv column moved to the original image (unchanged bytes when not active)."""
    if not rectified.active:
        return data
    import io
    with np.load(io.BytesIO(data), allow_pickle=False) as npz:
        arrays = {key: npz[key] for key in npz.files}
    if "uv" not in arrays or not len(arrays["uv"]):
        return data
    uv = rectified.to_original(arrays["uv"].astype(np.float64))
    arrays["uv"] = np.clip(np.rint(uv), 0, np.iinfo(np.uint16).max).astype(arrays["uv"].dtype)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


_RECTIFIED_CACHE: collections.OrderedDict[tuple[str, bytes, bytes], Rectified] = collections.OrderedDict()
_RECTIFIED_LOCK = threading.Lock()


def rectified_cached(camera, image_bytes) -> Rectified:
    """Thread-safe bounded LRU wrapper around ``Rectified`` so concurrent ``/object_table`` and ``/points``
    requests on the same observation frame share one decoded/undistorted/encoded buffer and SHA-256 digest."""
    if not isinstance(image_bytes, (bytes, bytearray, memoryview)):
        return Rectified(camera, image_bytes)
    raw_bytes = bytes(image_bytes) if not isinstance(image_bytes, bytes) else image_bytes
    K = np.asarray(camera["K"], dtype=np.float64)
    D = np.asarray(camera["D"], dtype=np.float64).reshape(-1)
    key = (hashlib.sha256(raw_bytes).hexdigest(), K.tobytes(), D.tobytes())
    with _RECTIFIED_LOCK:
        cached = _RECTIFIED_CACHE.get(key)
        if cached is not None:
            _RECTIFIED_CACHE.move_to_end(key)
            return cached
    built = Rectified(camera, raw_bytes)
    _ = built.sha256
    with _RECTIFIED_LOCK:
        existing = _RECTIFIED_CACHE.get(key)
        if existing is not None:
            _RECTIFIED_CACHE.move_to_end(key)
            return existing
        _RECTIFIED_CACHE[key] = built
        while len(_RECTIFIED_CACHE) > _RECTIFIED_CACHE_MAX:
            _RECTIFIED_CACHE.popitem(last=False)
    return built


def clear_rectified_cache() -> None:
    """Clear the process-local rectified-image LRU cache."""
    with _RECTIFIED_LOCK:
        _RECTIFIED_CACHE.clear()
