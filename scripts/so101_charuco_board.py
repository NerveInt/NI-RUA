"""Render the SO-101 calibration ChArUco board to PNG + PDF at exact A4 print scale.

Default board (docs/development/so101-integration.md section 3.2): 7 x 5 squares, 30 mm squares, 22 mm
markers, DICT_5X5_100, placed on a landscape A4 page (297 x 210 mm). The PDF page size is exactly A4 and the
board is drawn at 1:1 physical scale; print it with "actual size" / 100 % scaling (never "fit to page").

After printing, measure one square edge (or the 100 mm scale bar) with a ruler or calipers. If it is not
30.0 mm, the printer rescaled the page: pass the measured square edge (m) to
``scripts/so101_calibrate.py solve --square-m <value>`` (marker edge is scaled by the same ratio).

Usage:
    python scripts/so101_charuco_board.py --output-dir /tmp/so101_board
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ni_rua.robot.calibration.workflow.charuco import DEFAULT_BOARD_SPEC, make_board, validate_board_spec  # noqa: E402

A4_LANDSCAPE_MM = (297.0, 210.0)
# 24 px/mm (609.6 dpi) keeps every 30 mm square an exact integer number of pixels (720 px).
DEFAULT_PX_PER_MM = 24


def _format_mm(value_m: float) -> str:
    """将米制长度格式化为毫米字符串，整数或一位小数保持一位小数，否则保留两位小数。"""
    mm = value_m * 1000.0
    return f"{mm:.1f}" if abs(mm * 10.0 - round(mm * 10.0)) < 1e-6 else f"{mm:.2f}"


def load_board_spec(args: argparse.Namespace) -> dict:
    """从标定板 JSON 配置与 CLI 覆盖参数组装并校验规格。"""
    spec = dict(DEFAULT_BOARD_SPEC)
    if args.board_config:
        raw = json.loads(Path(args.board_config).expanduser().read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("board config JSON must be an object")
        # 过滤注释键与厚度字段，仅传入标定板几何与字典规格
        spec.update({k: v for k, v in raw.items() if not k.startswith("_") and k != "thickness_m"})
    if args.squares_x is not None:
        spec["squares_x"] = args.squares_x
    if args.squares_y is not None:
        spec["squares_y"] = args.squares_y
    if args.square_mm is not None:
        spec["square_m"] = args.square_mm / 1000.0
    if args.marker_mm is not None:
        spec["marker_m"] = args.marker_mm / 1000.0
    if args.dictionary is not None:
        spec["dictionary"] = args.dictionary
    return validate_board_spec(spec)


def render_board_page(spec: dict | None = None, *, px_per_mm: int = DEFAULT_PX_PER_MM) -> tuple[np.ndarray, dict]:
    """Render the board centred on a landscape A4 page.

    Args:
        spec: board spec (see ``so101_calibration.validate_board_spec``); default the shared 7x5 board.
        px_per_mm: raster resolution, pixels per millimetre. ``square_m * 1000 * px_per_mm`` must be an
            integer so squares are not resampled.

    Returns:
        ``(page, info)``: grayscale uint8 page image and a dict with ``dpi``, ``page_mm``, ``board_mm``.
    """
    spec = validate_board_spec(spec or DEFAULT_BOARD_SPEC)
    board = make_board(spec)
    square_px = spec["square_m"] * 1000.0 * px_per_mm
    if abs(square_px - round(square_px)) > 1e-6:
        raise ValueError(f"square_m * 1000 * px_per_mm = {square_px} must be an integer")
    square_px = int(round(square_px))
    board_w, board_h = spec["squares_x"] * square_px, spec["squares_y"] * square_px
    page_w, page_h = int(A4_LANDSCAPE_MM[0] * px_per_mm), int(A4_LANDSCAPE_MM[1] * px_per_mm)
    if board_w > page_w - 20 * px_per_mm or board_h > page_h - 40 * px_per_mm:
        raise ValueError("board does not fit on landscape A4 with margins for the quiet zone and labels")
    board_img = board.generateImage((board_w, board_h), marginSize=0, borderBits=1)
    page = np.full((page_h, page_w), 255, dtype=np.uint8)
    x0, y0 = (page_w - board_w) // 2, (page_h - board_h) // 2
    page[y0:y0 + board_h, x0:x0 + board_w] = board_img

    # 100 mm scale bar under the board for checking the print scale.
    bar_y = y0 + board_h + 8 * px_per_mm
    bar_len = 100 * px_per_mm
    cv2.rectangle(page, (x0, bar_y), (x0 + bar_len, bar_y + px_per_mm), 0, thickness=-1)
    for k in range(11):
        x = x0 + k * 10 * px_per_mm
        cv2.line(page, (x, bar_y - 2 * px_per_mm), (x, bar_y + 3 * px_per_mm), 0, thickness=max(1, px_per_mm // 6))
    font_scale = px_per_mm / 12.0
    thickness = max(1, px_per_mm // 8)
    cv2.putText(page, "100 mm", (x0 + bar_len + 4 * px_per_mm, bar_y + 2 * px_per_mm), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, 0, thickness, cv2.LINE_AA)
    label = (f"NI-RUA SO-101 ChArUco {spec['squares_x']}x{spec['squares_y']}  square {_format_mm(spec['square_m'])} mm"
             f"  marker {_format_mm(spec['marker_m'])} mm  {spec['dictionary']}  - print at 100% (actual size)")
    cv2.putText(page, label, (x0, y0 - 8 * px_per_mm), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 0, thickness,
                cv2.LINE_AA)
    info = {"dpi": px_per_mm * 25.4, "page_mm": list(A4_LANDSCAPE_MM),
            "board_mm": [board_w / px_per_mm, board_h / px_per_mm], "board": spec}
    return page, info


def write_board_files(output_dir: str | Path, spec: dict | None = None, *, px_per_mm: int = DEFAULT_PX_PER_MM
                      ) -> dict[str, str]:
    """Write ``so101_charuco_board.png`` (with DPI metadata) and ``.pdf`` (exact A4 page) into ``output_dir``.

    The PDF requires Pillow; if Pillow is missing only the PNG is written and ``pdf`` is ``None``.
    Returns ``{"png": path, "pdf": path or None}``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    page, info = render_board_page(spec, px_per_mm=px_per_mm)
    png = output_dir / "so101_charuco_board.png"
    pdf = output_dir / "so101_charuco_board.pdf"
    try:
        from PIL import Image
    except ImportError:
        if not cv2.imwrite(str(png), page):
            raise OSError(f"failed to write {png}")
        return {"png": str(png), "pdf": None}
    image = Image.fromarray(page, mode="L")
    dpi = (info["dpi"], info["dpi"])
    image.save(png, dpi=dpi)
    image.save(pdf, "PDF", resolution=info["dpi"])
    return {"png": str(png), "pdf": str(pdf)}


def print_instructions(paths: dict, spec: dict | None = None) -> None:
    spec = validate_board_spec(spec or DEFAULT_BOARD_SPEC)
    print(f"PNG: {paths['png']}")
    print(f"PDF: {paths['pdf']}" if paths.get("pdf") else "PDF: not written (Pillow not installed)")
    print(f"Print on A4 landscape at 100% / actual size (no 'fit to page'). Expected square edge: "
          f"{_format_mm(spec['square_m'])} mm, scale bar: 100.0 mm.")
    print("Measure the printed square edge. If it differs, pass the measured value in metres, e.g. "
          "`so101_calibrate.py solve --square-m 0.0297`. Glue the sheet flat on a rigid plate and fix it to "
          "the table inside both cameras' view; it must not move during capture.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=".", help="directory for so101_charuco_board.png/.pdf")
    parser.add_argument("--px-per-mm", type=int, default=DEFAULT_PX_PER_MM,
                        help="raster resolution in pixels per millimetre (default 24 = 609.6 dpi)")
    # 可选规格配置与单字段覆盖参数，未传入时完全沿用默认 7x5 标定板
    parser.add_argument("--board-config", help="JSON board spec file (e.g. configs/so101_board.json)")
    parser.add_argument("--squares-x", type=int, help="override chessboard square count along X")
    parser.add_argument("--squares-y", type=int, help="override chessboard square count along Y")
    parser.add_argument("--square-mm", type=float, help="override chessboard square edge length (mm)")
    parser.add_argument("--marker-mm", type=float, help="override ArUco marker edge length (mm)")
    parser.add_argument("--dictionary", help="override OpenCV ArUco dictionary name (e.g. DICT_5X5_100)")
    args = parser.parse_args(argv)
    spec = load_board_spec(args)
    paths = write_board_files(args.output_dir, spec=spec, px_per_mm=args.px_per_mm)
    print_instructions(paths, spec=spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
