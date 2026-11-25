"""Native-resolution Pillow HUD renderer for SO-101 motion composite videos.

Layout (``W`` = even-padded sum of camera tile widths, e.g. ``2 x 640 = 1280``)::

    y = 0              optional header band, HEADER_HEIGHT (40 px), solid black
                       left:  Backend-Model-Effort     (e.g. OpenCode-DeepSeek 4.1 Flash-high)
                       right: Tokens: 7.95M   Cost: $0.28
    y = header_h       camera tiles side by side, source pixels copied 1:1,
                       pre-rendered crisp black camera-id pill at each tile's top-left
    y = header_h + th  timeline footer, FOOTER_HEIGHT (132 px), native FreeType text

Every glyph is rasterised by FreeType at its final pixel size (zero resampling
blur before the single libx264 encode). Only the diagonal ``THINK`` hatching is
drawn at 2x and box-downsampled. The footer caches only the active segment's
static layer (O(1) memory regardless of segment count).
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HEADER_HEIGHT = 40  # Even height (px) placed above camera tiles when run metadata is present.
FOOTER_HEIGHT = 132  # Even height (px) placed below camera tiles; never crops or covers camera pixels.
HUD_HEIGHT = FOOTER_HEIGHT  # Backward-compatible alias for the bottom timeline HUD height.
HUD_RENDER_VERSION = 2

BOLD_FONTS = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
)
REGULAR_FONTS = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
)

# RGB palette tuned for high contrast after yuv420p 4:2:0 encoding.
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BG_HUD = (14, 18, 25)
BORDER_TOP = (58, 72, 94)
DIVIDER = (34, 44, 58)
TRACK_BG = (22, 28, 38)
TRACK_BORDER = (68, 82, 104)

THINK_FILL_PAST = (74, 52, 22)
THINK_FILL_FUTURE = (52, 37, 18)
THINK_HATCH = (158, 108, 36)
THINK_BORDER = (186, 128, 42)
THINK_PILL_BG = (36, 26, 13)
THINK_PILL_TEXT = (232, 184, 92)
THINK_TEXT = (252, 204, 76)

MOVE_PLAYED = (22, 168, 112)
MOVE_PLAYED_TOP = (76, 228, 170)
MOVE_PLAYED_BORDER = (44, 204, 144)
MOVE_ACTIVE_REM = (20, 116, 94)
MOVE_ACTIVE_FILL = (24, 232, 166)
MOVE_ACTIVE_GLOW = (152, 252, 220)
MOVE_UPCOMING = (18, 88, 76)
MOVE_UPCOMING_BORDER = (48, 152, 126)
MOVE_TEXT = (96, 238, 184)

TEXT_PRIMARY = (241, 245, 249)
TEXT_SECONDARY = (182, 196, 214)
TEXT_MUTED = (122, 138, 160)

NEEDLE_HALO = (8, 11, 16)
NEEDLE_LINE = (255, 250, 212)
NEEDLE_CAP = (255, 216, 74)


def fmt_tokens(value: int | None) -> str:
    if value is None:
        return "n/a"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def fmt_cost(value: float | None, currency: str | None = "USD") -> str:
    if value is None:
        return "n/a"
    code = (currency or "USD").strip().upper()
    if code == "USD":
        return f"${value:.2f}"
    return f"{code} {value:.2f}"


def fmt_clock(seconds: float, *, tenths: bool = False) -> str:
    """Format nonnegative seconds as ``MM:SS`` / ``MM:SS.t`` (or ``H:MM:SS`` when >= 1 hour)."""
    clamped = max(0.0, float(seconds))
    if tenths:
        whole, tenth = divmod(int(round(clamped * 10.0)), 10)
    else:
        whole, tenth = int(round(clamped)), None
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    head = f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"
    return head if tenth is None else f"{head}.{tenth}"


def short_action_id(action_id: str | None) -> str:
    if not action_id:
        return "move"
    text = str(action_id)
    if len(text) <= 16:
        return text
    if text.startswith("action-call-") and len(text) > 18:
        return "call-" + text[12:18]
    return text[:14] + ".."


def _find_font_path(candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


class HudFonts:
    """Cross-platform FreeType font cache with graceful Pillow default fallback."""

    def __init__(self) -> None:
        self._bold_path = _find_font_path(BOLD_FONTS)
        self._regular_path = _find_font_path(REGULAR_FONTS) or self._bold_path
        self._cache: dict[tuple[bool, int], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
        self.header = self.get(24, bold=True)
        self.header_kv = self.get(20, bold=False)
        self.header_kv_bold = self.get(20, bold=True)
        self.camera = self.get(15, bold=True)
        self.row = self.get(16, bold=True)
        self.row2 = self.get(15, bold=True)
        self.small = self.get(13, bold=True)
        self.tag = self.get(12, bold=True)
        self.micro = self.get(11, bold=True)

    def get(self, size: int, *, bold: bool = False):
        key = (bold, int(size))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        path = self._bold_path if bold else self._regular_path
        loaded = None
        if path is not None:
            try:
                loaded = ImageFont.truetype(path, int(size))
            except OSError:
                loaded = None
        if loaded is None:
            try:
                loaded = ImageFont.load_default(size=int(size))
            except TypeError:
                loaded = ImageFont.load_default()
        self._cache[key] = loaded
        return loaded


def _text_width(font, text: str) -> float:
    return float(font.getlength(text))


def _spans_width(spans: list[tuple[str, tuple[int, int, int]]], font) -> float:
    return sum(_text_width(font, text) for text, _ in spans)


def _draw_spans(
    draw: ImageDraw.ImageDraw,
    x: float,
    baseline: int,
    spans: list[tuple[str, tuple[int, int, int]]],
    font,
    *,
    max_x: float | None = None,
) -> float:
    cur_x = float(x)
    for text, color in spans:
        tw = _text_width(font, text)
        if max_x is not None and cur_x + tw > max_x:
            break
        draw.text((cur_x, baseline), text, font=font, fill=color, anchor="ls")
        cur_x += tw
    return cur_x


def _truncate_text(text: str, font, max_width: float) -> str:
    if max_width <= 0:
        return ""
    if _text_width(font, text) <= max_width:
        return text
    ellipsis = "..."
    ew = _text_width(font, ellipsis)
    if ew >= max_width:
        return ""
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _text_width(font, text[:mid]) + ew <= max_width:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + ellipsis if lo > 0 else ""


def _hatched(width: int, height: int, fill, step: int, line_w: int, phase: int = 0) -> Image.Image:
    """Rectangle of ``fill`` with 45-degree THINK_HATCH lines every ``step`` px (lines clipped to the rect)."""
    tile = Image.new("RGB", (max(1, width), max(1, height)), fill)
    draw = ImageDraw.Draw(tile)
    for d in range(-height - (phase % step), width + step, step):
        draw.line((d, height, d + height, 0), fill=THINK_HATCH, width=line_w)
    return tile


def _collides(occupied: list[tuple[float, float]], left: float, right: float, margin: float) -> bool:
    return any(not (right + margin < a or left - margin > b) for a, b in occupied)


def render_header(width: int, header_summary: dict, fonts: HudFonts) -> np.ndarray:
    """Render the width-adaptive ``HEADER_HEIGHT`` top band once per video."""
    image = Image.new("RGB", (width, HEADER_HEIGHT), BLACK)
    draw = ImageDraw.Draw(image)
    label = str(header_summary.get("label") or "NI-RUA")
    tokens_txt = fmt_tokens(header_summary.get("total_tokens"))
    cost_txt = fmt_cost(header_summary.get("cost_usd"), header_summary.get("currency"))

    tiers = (
        (
            16,
            24,
            20,
            24,
            HEADER_HEIGHT // 2 + 9,
            [
                ("Tokens: ", TEXT_SECONDARY, fonts.get(20, bold=False)),
                (tokens_txt, TEXT_PRIMARY, fonts.get(20, bold=True)),
                ("     Cost: ", TEXT_SECONDARY, fonts.get(20, bold=False)),
                (cost_txt, TEXT_PRIMARY, fonts.get(20, bold=True)),
            ],
        ),
        (
            14,
            19,
            16,
            18,
            HEADER_HEIGHT // 2 + 7,
            [
                ("Tokens: ", TEXT_SECONDARY, fonts.get(16, bold=False)),
                (tokens_txt, TEXT_PRIMARY, fonts.get(16, bold=True)),
                ("   Cost: ", TEXT_SECONDARY, fonts.get(16, bold=False)),
                (cost_txt, TEXT_PRIMARY, fonts.get(16, bold=True)),
            ],
        ),
        (
            10,
            15,
            13,
            12,
            HEADER_HEIGHT // 2 + 5,
            [
                ("Tok: ", TEXT_SECONDARY, fonts.get(13, bold=False)),
                (tokens_txt, TEXT_PRIMARY, fonts.get(13, bold=True)),
                ("  ", TEXT_MUTED, fonts.get(13, bold=False)),
                (cost_txt, TEXT_PRIMARY, fonts.get(13, bold=True)),
            ],
        ),
    )

    for margin, left_size, _right_size, gap, baseline, right_spans in tiers:
        left_font = fonts.get(left_size, bold=True)
        right_w = sum(_text_width(f, t) for t, _, f in right_spans)
        label_w = _text_width(left_font, label)
        if margin + label_w + gap + right_w <= width - margin:
            draw.text((margin, baseline), label, font=left_font, fill=WHITE, anchor="ls")
            x = width - margin - right_w
            for text, color, font in right_spans:
                draw.text((x, baseline), text, font=font, fill=color, anchor="ls")
                x += _text_width(font, text)
            return np.asarray(image)

    # Ultra-narrow fallback (e.g. narrow single-camera or synthetic test widths): truncate cleanly, never raise.
    margin = 8 if width >= 200 else 4
    baseline = HEADER_HEIGHT // 2 + 5
    left_font = fonts.get(13, bold=True)
    compact_right = [
        (tokens_txt, TEXT_PRIMARY, fonts.get(12, bold=True)),
        (" ", TEXT_MUTED, fonts.get(12, bold=False)),
        (cost_txt, TEXT_SECONDARY, fonts.get(12, bold=False)),
    ]
    right_w = sum(_text_width(f, t) for t, _, f in compact_right)
    usable = width - 2 * margin
    if usable >= right_w + 56:
        fitted = _truncate_text(label, left_font, usable - right_w - 10)
        if fitted:
            draw.text((margin, baseline), fitted, font=left_font, fill=WHITE, anchor="ls")
        x = width - margin - right_w
        for text, color, font in compact_right:
            draw.text((x, baseline), text, font=font, fill=color, anchor="ls")
            x += _text_width(font, text)
    else:
        fitted = _truncate_text(label, left_font, usable)
        if fitted:
            draw.text((margin, baseline), fitted, font=left_font, fill=WHITE, anchor="ls")
    return np.asarray(image)


def make_camera_pill(camera_id: str, fonts: HudFonts, *, max_width: int | None = None) -> np.ndarray:
    """Pre-render a crisp black camera-id pill once per camera."""
    font = fonts.camera
    text = str(camera_id)
    if max_width is not None and max_width > 20:
        text = _truncate_text(text, font, max_width - 14) or text[:1]
    w = max(16, int(round(_text_width(font, text))) + 14)
    if max_width is not None:
        w = min(w, max(1, int(max_width)))
    h = 22
    pill = Image.new("RGB", (w, h), BLACK)
    ImageDraw.Draw(pill).text((7, 16), text, font=font, fill=WHITE, anchor="ls")
    return np.asarray(pill)


def make_no_frame_badge(fonts: HudFonts) -> np.ndarray:
    font = fonts.row2
    text = "no frame"
    w = int(round(_text_width(font, text))) + 16
    h = 24
    badge = Image.new("RGB", (w, h), BLACK)
    ImageDraw.Draw(badge).text((8, 17), text, font=font, fill=TEXT_PRIMARY, anchor="ls")
    return np.asarray(badge)


def stamp_patch(canvas: np.ndarray, patch: np.ndarray, x: int, y: int) -> None:
    """Copy ``patch`` onto ``canvas`` at ``(x, y)``, safely clipped to bounds."""
    ch, cw = canvas.shape[:2]
    ph, pw = patch.shape[:2]
    if x >= cw or y >= ch or x + pw <= 0 or y + ph <= 0:
        return
    sx0 = max(0, -x)
    sy0 = max(0, -y)
    dx0 = max(0, x)
    dy0 = max(0, y)
    w = min(pw - sx0, cw - dx0)
    h = min(ph - sy0, ch - dy0)
    if w > 0 and h > 0:
        canvas[dy0:dy0 + h, dx0:dx0 + w] = patch[sy0:sy0 + h, sx0:sx0 + w]


class TimelineHud:
    """Native-resolution bottom timeline HUD renderer with O(1) segment-layer memory.

    Accepts either a full ``_build_timeline()`` dict or a ``timeline_summary`` dict.
    """

    def __init__(
        self,
        width: int,
        timeline: dict,
        total_clip_frames_or_s: int | float,
        fps: float | None = None,
        *,
        fonts: HudFonts | None = None,
    ) -> None:
        self.width = int(width)
        self.height = FOOTER_HEIGHT
        self.fonts = fonts or HudFonts()
        self.timeline = timeline

        if "raw_intervals" in timeline:
            self.task_start_s = float(timeline["task_start_s"])
            self.total_task_s = float(timeline["total_task_s"])
            self.total_motion_s = float(timeline["total_motion_s"])
            self.total_thinking_s = float(timeline["total_thinking_s"])
            self.intervals = list(timeline["raw_intervals"])
        else:
            self.task_start_s = 0.0
            self.total_task_s = float(timeline["total_task_s"])
            self.total_motion_s = float(timeline["total_motion_s"])
            self.total_thinking_s = float(
                timeline.get("total_thinking_skipped_s", max(0.0, self.total_task_s - self.total_motion_s))
            )
            self.intervals = list(timeline.get("intervals") or [])

        self.span_s = max(self.total_task_s, 1e-6)
        if fps is not None:
            safe_fps = max(float(fps), 1e-6)
            self.total_clip_s = max(0.0, float(total_clip_frames_or_s) / safe_fps)
        else:
            self.total_clip_s = max(0.0, float(total_clip_frames_or_s))

        if self.total_task_s > 0.0:
            self.move_pct = round(100.0 * self.total_motion_s / self.total_task_s, 1)
            self.think_pct = round(max(0.0, 100.0 - self.move_pct), 1)
        else:
            self.move_pct, self.think_pct = 100.0, 0.0

        self.moves: list[dict] = []
        previous_thinking = None
        for iv in self.intervals:
            if iv["kind"] == "thinking":
                previous_thinking = iv
            elif iv["kind"] == "motion":
                self.moves.append({
                    **iv,
                    "move_number": len(self.moves) + 1,
                    "preceding": previous_thinking,
                })
                previous_thinking = None
        self.by_segment = {int(m["segment_index"]): m for m in self.moves if m.get("segment_index") is not None}

        pad = 28 if self.width >= 480 else max(6, self.width // 14)
        self.bx0 = pad
        self.bx1 = max(self.bx0 + 4, self.width - pad)
        self.bw = self.bx1 - self.bx0
        self.bar_y0, self.bar_y1 = 80, 104

        self.skip_labels = self._plan_skip_labels()
        self.tick_labels, self.minor_ticks = self._plan_ticks()

        # Single cached static segment image (O(1) memory regardless of segment count).
        self._cached_seg_idx: int | None = None
        self._cached_base_img: Image.Image | None = None
        self._cached_active_box: tuple[int, int] | None = None

    def x_at(self, rel_s: float) -> int:
        frac = min(1.0, max(0.0, rel_s / self.span_s))
        return self.bx0 + int(round(frac * self.bw))

    def move_box(self, move: dict) -> tuple[int, int]:
        x0, x1 = self.x_at(move["start_s"]), self.x_at(move["end_s"])
        min_w = min(3, self.bw)
        if x1 - x0 < min_w:
            center = (x0 + x1) // 2
            x0 = max(self.bx0, min(self.bx1 - min_w, center - min_w // 2))
            x1 = min(self.bx1, x0 + min_w)
        return x0, x1

    def _plan_skip_labels(self) -> list[tuple[str, float, int]]:
        if self.bw < 120:
            return []
        font, placed, occupied = self.fonts.small, [], []
        thinking = sorted(
            (iv for iv in self.intervals if iv["kind"] == "thinking" and iv["duration_s"] >= 1.0),
            key=lambda iv: (-iv["duration_s"], iv["start_s"]),
        )
        for iv in thinking:
            x0, x1 = self.x_at(iv["start_s"]), self.x_at(iv["end_s"])
            if x1 - x0 < 14:
                continue
            xc = (x0 + x1) / 2.0
            dur = iv["duration_s"]
            dur_txt = f"{dur:.0f}s" if dur >= 10.0 else f"{dur:.1f}s"
            candidates = (
                f"Skip {dur_txt} ({fmt_clock(iv['start_s'])}-{fmt_clock(iv['end_s'])})",
                f"Skip {dur_txt}",
            )
            for text in candidates:
                tw = _text_width(font, text)
                if "(" in text and x1 - x0 < tw + 6:
                    continue
                if tw > self.bw:
                    continue
                lx = max(float(self.bx0), min(float(self.bx1) - tw, xc - tw / 2.0))
                if lx <= xc <= lx + tw and not _collides(occupied, lx, lx + tw, 11.0):
                    occupied.append((lx, lx + tw))
                    placed.append((text, lx, int(round(xc))))
                    break
        return placed

    def _plan_ticks(self) -> tuple[list[tuple[str, float, int, tuple[int, int, int]]], list[int]]:
        if self.bw < 100:
            return [], []
        font, labels, occupied = self.fonts.small, [], []
        start_txt, end_txt = fmt_clock(0.0), fmt_clock(self.total_task_s)
        sw, ew = _text_width(font, start_txt), _text_width(font, end_txt)
        if sw + ew + 12 < self.bw:
            occupied += [(float(self.bx0), float(self.bx0) + sw), (float(self.bx1) - ew, float(self.bx1))]
            labels += [
                (start_txt, float(self.bx0), self.bx0, TEXT_PRIMARY),
                (end_txt, float(self.bx1) - ew, self.bx1, TEXT_PRIMARY),
            ]
        transitions: list[tuple[float, float]] = []
        for idx, move in enumerate(self.moves):
            prev_gap = move["preceding"]["duration_s"] if move["preceding"] else 0.0
            next_gap = (
                (self.moves[idx + 1]["preceding"] or {"duration_s": 0.0})["duration_s"]
                if idx + 1 < len(self.moves)
                else max(0.0, self.total_task_s - move["end_s"])
            )
            if move["start_s"] > 0.5:
                transitions.append((max(prev_gap, move["duration_s"]), move["start_s"]))
            if self.total_task_s - move["end_s"] > 0.5:
                transitions.append((next_gap, move["end_s"]))
        for _, rel_s in sorted(transitions, key=lambda item: (-item[0], item[1])):
            tx, text = self.x_at(rel_s), fmt_clock(rel_s)
            tw = _text_width(font, text)
            lx = max(float(self.bx0), min(float(self.bx1) - tw, tx - tw / 2.0))
            if not _collides(occupied, lx, lx + tw, 12.0):
                occupied.append((lx, lx + tw))
                labels.append((text, lx, tx, TEXT_SECONDARY))
        minor: list[int] = []
        if self.total_task_s >= 15.0:
            step = next(
                (s for s in (5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 900.0, 1800.0)
                 if self.total_task_s / s <= 10.0),
                1800.0,
            )
            grid = step
            while grid < self.total_task_s - 0.5 * step:
                tx, text = self.x_at(grid), fmt_clock(grid)
                minor.append(tx)
                tw = _text_width(font, text)
                lx = max(float(self.bx0), min(float(self.bx1) - tw, tx - tw / 2.0))
                if not _collides(occupied, lx, lx + tw, 16.0):
                    occupied.append((lx, lx + tw))
                    labels.append((text, lx, tx, TEXT_MUTED))
                grid += step
        return labels, minor

    def _hatch_layer(self, current_start_s: float) -> Image.Image:
        """THINK intervals with diagonal hatching, drawn at 2x and box-downsampled (shapes only)."""
        scale = 2
        h2 = (self.bar_y1 - self.bar_y0 - 2) * scale
        layer = Image.new("RGB", (self.width * scale, h2), TRACK_BG)
        for iv in self.intervals:
            if iv["kind"] != "thinking":
                continue
            x0, x1 = self.x_at(iv["start_s"]) * scale, self.x_at(iv["end_s"]) * scale
            if x1 <= x0:
                continue
            fill = THINK_FILL_PAST if iv["end_s"] <= current_start_s + 1e-4 else THINK_FILL_FUTURE
            layer.paste(_hatched(x1 - x0, h2, fill, 16, 3, phase=x0), (x0, 0))
        return layer.resize((self.width, h2 // scale), Image.Resampling.BOX)

    def _draw_legend(self, image: Image.Image, draw: ImageDraw.ImageDraw, min_left_x: float) -> None:
        fonts = self.fonts
        sw_w, sw_h, gap_icon = 22, 13, 8
        for move_lbl, think_lbl, gap_group in (
            ("Actual Motion", "Skipped Thinking", 20),
            ("Motion", "Thinking", 14),
        ):
            legend = ((move_lbl, MOVE_TEXT, "move"), (think_lbl, THINK_TEXT, "think"))
            legend_w = sum(sw_w + gap_icon + _text_width(fonts.row2, t) for t, _, _ in legend) + gap_group
            legend_left = self.bx1 - legend_w
            if legend_left >= min_left_x:
                x = legend_left
                for text, color, kind in legend:
                    ix = int(round(x))
                    if kind == "move":
                        draw.rectangle((ix, 37, ix + sw_w, 37 + sw_h), fill=MOVE_PLAYED, outline=MOVE_ACTIVE_GLOW)
                    else:
                        swatch = _hatched((sw_w + 1) * 2, (sw_h + 1) * 2, THINK_FILL_PAST, 12, 3)
                        image.paste(swatch.resize((sw_w + 1, sw_h + 1), Image.Resampling.BOX), (ix, 37))
                        draw.rectangle((ix, 37, ix + sw_w, 37 + sw_h), outline=THINK_BORDER)
                    x += sw_w + gap_icon
                    draw.text((x, 49), text, font=fonts.row2, fill=color, anchor="ls")
                    x += _text_width(fonts.row2, text) + gap_group
                return

    def _segment_layer(self, segment_index: int) -> tuple[Image.Image, tuple[int, int] | None]:
        if self._cached_seg_idx == segment_index and self._cached_base_img is not None:
            return self._cached_base_img, self._cached_active_box

        fonts = self.fonts
        move = self.by_segment.get(segment_index)
        current_start = move["start_s"] if move else 0.0
        current_number = move["move_number"] if move else 0
        image = Image.new("RGB", (self.width, self.height), BG_HUD)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, self.width - 1, 1), fill=BORDER_TOP)
        draw.rectangle((self.bx0, 30, self.bx1, 30), fill=DIVIDER)

        # Row 2: active move + preceding skipped thinking + collision-free right-aligned legend.
        if move is not None and self.bw >= 60:
            pre = move["preceding"]
            pre_dur = float(pre["duration_s"]) if pre else 0.0
            has_pre = pre is not None and pre_dur >= 0.05
            pre_color = THINK_TEXT if has_pre else TEXT_SECONDARY
            short_aid = short_action_id(move.get("action_id"))
            num_moves = max(1, len(self.moves))
            full_pre_text = (
                f"Preceding Skipped Thinking: {fmt_clock(pre['start_s'], tenths=True)} - "
                f"{fmt_clock(pre['end_s'], tenths=True)} ({pre_dur:.1f}s)"
                if has_pre
                else "Preceding Skipped Thinking: none (0.0s)"
            )
            full_spans = [
                (
                    f"ACTIVE MOVE #{move['move_number']}/{num_moves} ({short_aid}): "
                    f"{fmt_clock(move['start_s'], tenths=True)} - {fmt_clock(move['end_s'], tenths=True)} "
                    f"({move['duration_s']:.1f}s)",
                    MOVE_TEXT,
                ),
                ("   |   ", TEXT_MUTED),
                (full_pre_text, pre_color),
            ]
            med_spans = [
                (
                    f"MOVE #{move['move_number']}/{num_moves} ({short_aid}): "
                    f"{fmt_clock(move['start_s'], tenths=True)}-{fmt_clock(move['end_s'], tenths=True)} "
                    f"({move['duration_s']:.1f}s)",
                    MOVE_TEXT,
                ),
                (" | ", TEXT_MUTED),
                (
                    f"Prev Think: {fmt_clock(pre['start_s'])}-{fmt_clock(pre['end_s'])} ({pre_dur:.1f}s)"
                    if has_pre
                    else "Prev Think: 0.0s",
                    pre_color,
                ),
            ]
            compact_spans = [
                (
                    f"#{move['move_number']}/{num_moves} ({short_aid[:10]}) "
                    f"{fmt_clock(move['start_s'], tenths=True)}-{fmt_clock(move['end_s'], tenths=True)}",
                    MOVE_TEXT,
                ),
                (" | ", TEXT_MUTED),
                (f"Skip {pre_dur:.1f}s", pre_color),
            ]
            tiny_spans = [
                (f"M{move['move_number']}/{num_moves} {move['duration_s']:.1f}s", MOVE_TEXT),
                (" ", TEXT_MUTED),
                (f"T{pre_dur:.0f}s", pre_color),
            ]

            draw.ellipse((self.bx0, 38, self.bx0 + 10, 48), fill=MOVE_ACTIVE_FILL, outline=MOVE_ACTIVE_GLOW)
            text_x0 = self.bx0 + 18
            avail_w = max(1, self.bx1 - text_x0)
            full_legend_w = (
                22 + 8 + _text_width(fonts.row2, "Actual Motion")
                + 20 + 22 + 8 + _text_width(fonts.row2, "Skipped Thinking")
            )
            tiers = (
                (fonts.row2, full_spans, avail_w - full_legend_w - 16),
                (fonts.small, full_spans, avail_w - full_legend_w - 16),
                (fonts.small, med_spans, avail_w),
                (fonts.tag, compact_spans, avail_w),
                (fonts.micro, tiny_spans, avail_w),
            )
            chosen_font, chosen_spans = tiers[-1][0], tiers[-1][1]
            for cand_font, cand_spans, limit_w in tiers:
                if limit_w > 0 and _spans_width(cand_spans, cand_font) <= limit_w:
                    chosen_font, chosen_spans = cand_font, cand_spans
                    break
            left_end = _draw_spans(draw, text_x0, 49, chosen_spans, chosen_font, max_x=float(self.bx1))
            self._draw_legend(image, draw, left_end + 16)
        elif self.bw >= 120:
            self._draw_legend(image, draw, float(self.bx0))

        # Row 3a: skip callouts above THINK intervals.
        for text, lx, xc in self.skip_labels:
            draw.text((lx, 71), text, font=fonts.small, fill=THINK_TEXT, anchor="ls")
            draw.rectangle((xc, 74, xc, 78), fill=THINK_BORDER)

        # Row 3b: hatched THINK layer, MOVE boxes, border.
        image.paste(self._hatch_layer(current_start), (0, self.bar_y0 + 1))
        draw.rectangle((0, self.bar_y0, self.bx0 - 1, self.bar_y1), fill=BG_HUD)
        draw.rectangle((self.bx1 + 1, self.bar_y0, self.width - 1, self.bar_y1), fill=BG_HUD)
        for iv in self.intervals:
            if iv["kind"] != "thinking":
                continue
            x0, x1 = self.x_at(iv["start_s"]), self.x_at(iv["end_s"])
            if x1 <= x0:
                continue
            draw.rectangle((x0, self.bar_y0 + 1, x1, self.bar_y0 + 1), fill=THINK_BORDER)
            draw.rectangle((x0, self.bar_y1 - 1, x1, self.bar_y1 - 1), fill=THINK_BORDER)
            tw = _text_width(fonts.tag, "THINK")
            if x1 - x0 >= tw + 40:
                tx = (x0 + x1 - tw) / 2.0
                draw.rectangle((tx - 5, self.bar_y0 + 5, tx + tw + 5, self.bar_y1 - 5), fill=THINK_PILL_BG)
                draw.text((tx, self.bar_y1 - 8), "THINK", font=fonts.tag, fill=THINK_PILL_TEXT, anchor="ls")

        active_box = None
        for m in self.moves:
            x0, x1 = self.move_box(m)
            if m["move_number"] < current_number:
                draw.rectangle((x0, self.bar_y0 + 1, x1, self.bar_y1 - 1), fill=MOVE_PLAYED, outline=MOVE_PLAYED_BORDER)
                draw.rectangle((x0, self.bar_y0 + 1, x1, self.bar_y0 + 2), fill=MOVE_PLAYED_TOP)
            elif m["move_number"] > current_number:
                draw.rectangle((x0, self.bar_y0 + 1, x1, self.bar_y1 - 1), fill=MOVE_UPCOMING, outline=MOVE_UPCOMING_BORDER)
            else:
                active_box = (x0, x1)
                draw.rectangle((x0, self.bar_y0 + 1, x1, self.bar_y1 - 1), fill=MOVE_ACTIVE_REM)
            if m["move_number"] != current_number:
                self._move_tag(draw, x0, x1)
        draw.rectangle((self.bx0, self.bar_y0, self.bx1, self.bar_y1), outline=TRACK_BORDER)

        # Row 4: ticks and wall-clock labels.
        for tx in self.minor_ticks:
            draw.rectangle((tx, self.bar_y1 + 1, tx, self.bar_y1 + 4), fill=TRACK_BORDER)
        for m in self.moves:
            for tx in self.move_box(m):
                draw.rectangle((tx, self.bar_y1 + 1, tx, self.bar_y1 + 5), fill=MOVE_PLAYED_BORDER)
        for text, lx, tx, color in self.tick_labels:
            draw.rectangle((tx, self.bar_y1 + 1, tx, self.bar_y1 + 6), fill=color)
            draw.text((lx, 124), text, font=fonts.small, fill=color, anchor="ls")

        self._cached_seg_idx = segment_index
        self._cached_base_img = image
        self._cached_active_box = active_box
        return image, active_box

    def _move_tag(self, draw: ImageDraw.ImageDraw, x0: int, x1: int) -> None:
        tw = _text_width(self.fonts.tag, "MOVE")
        if x1 - x0 >= tw + 12:
            draw.text(
                ((x0 + x1 - tw) / 2.0, self.bar_y1 - 8),
                "MOVE",
                font=self.fonts.tag,
                fill=TEXT_PRIMARY,
                anchor="ls",
                stroke_width=2,
                stroke_fill=NEEDLE_HALO,
            )

    def render(self, segment_index: int, task_s: float, clip_s: float) -> np.ndarray:
        base, active_box = self._segment_layer(segment_index)
        image = base.copy()
        draw = ImageDraw.Draw(image)
        fonts = self.fonts
        task_s = min(self.total_task_s, max(0.0, float(task_s)))
        nx = self.x_at(task_s)

        # Row 1: live task elapsed + motion / skipped-thinking / clip breakdown (width-adaptive).
        if self.bw >= 60:
            cur_clock = fmt_clock(task_s, tenths=True)
            tot_clock = fmt_clock(self.total_task_s, tenths=True)
            row1_tiers = (
                (
                    fonts.row,
                    [
                        ("Task Elapsed: ", TEXT_SECONDARY),
                        (f"{cur_clock} / {tot_clock} ({self.total_task_s:.1f}s)", TEXT_PRIMARY),
                    ],
                ),
                (
                    fonts.small,
                    [
                        ("Task: ", TEXT_SECONDARY),
                        (f"{cur_clock}/{tot_clock}", TEXT_PRIMARY),
                    ],
                ),
                (fonts.tag, [(f"{cur_clock}/{tot_clock}", TEXT_PRIMARY)]),
            )
            r1_font, left_spans = row1_tiers[-1]
            for cand_font, cand_spans in row1_tiers:
                if _spans_width(cand_spans, cand_font) <= self.bw:
                    r1_font, left_spans = cand_font, cand_spans
                    break
            left_end = _draw_spans(draw, self.bx0, 22, left_spans, r1_font, max_x=float(self.bx1))

            groups = [
                (f"Motion: {self.total_motion_s:.1f}s ({self.move_pct:.1f}%)", MOVE_TEXT),
                (f"Skipped Thinking: {self.total_thinking_s:.1f}s ({self.think_pct:.1f}%)", THINK_TEXT),
                (f"Clip: {fmt_clock(clip_s, tenths=True)} / {fmt_clock(self.total_clip_s, tenths=True)}", TEXT_SECONDARY),
            ]
            placed = False
            for keep in (3, 2, 1):
                spans: list[tuple[str, tuple[int, int, int]]] = []
                for idx, item in enumerate(groups[:keep]):
                    if idx:
                        spans.append(("   |   ", TEXT_MUTED))
                    spans.append(item)
                rw = _spans_width(spans, r1_font)
                if self.bx1 - rw >= left_end + 20:
                    _draw_spans(draw, self.bx1 - rw, 22, spans, r1_font)
                    placed = True
                    break
            if not placed:
                compact_right = [
                    (f"M:{self.move_pct:.0f}%", MOVE_TEXT),
                    (" ", TEXT_MUTED),
                    (f"T:{self.think_pct:.0f}%", THINK_TEXT),
                ]
                rw = _spans_width(compact_right, fonts.tag)
                if self.bx1 - rw >= left_end + 10:
                    _draw_spans(draw, self.bx1 - rw, 22, compact_right, fonts.tag)

        # Active MOVE swept fill & glowing outline.
        if active_box is not None:
            x0, x1 = active_box
            sweep = max(x0 + 1, min(x1, nx))
            draw.rectangle((x0, self.bar_y0 + 1, sweep, self.bar_y1 - 1), fill=MOVE_ACTIVE_FILL)
            self._move_tag(draw, x0, x1)
            draw.rectangle((x0, self.bar_y0, x1, self.bar_y1), outline=MOVE_ACTIVE_GLOW, width=2)

        # Live playhead needle (halo + bright line + top/bottom triangles).
        draw.rectangle((nx - 2, self.bar_y0 - 4, nx + 2, self.bar_y1 + 4), fill=NEEDLE_HALO)
        draw.rectangle((nx - 1, self.bar_y0 - 4, nx, self.bar_y1 + 4), fill=NEEDLE_LINE)
        for tip_y, base_y in ((self.bar_y0 + 1, self.bar_y0 - 7), (self.bar_y1 - 1, self.bar_y1 + 7)):
            draw.polygon([(nx - 6, base_y), (nx + 6, base_y), (nx, tip_y)], fill=NEEDLE_CAP, outline=NEEDLE_HALO)
        return np.asarray(image)

    def render_into(
        self,
        canvas: np.ndarray,
        y_offset: int,
        segment_index: int,
        display_monotonic_s: float,
        clip_pts_s: float,
    ) -> None:
        task_s = max(0.0, float(display_monotonic_s) - self.task_start_s)
        canvas[y_offset:y_offset + self.height, :] = self.render(segment_index, task_s, clip_pts_s)
