"""Two-view top height of an object standing on the table: ``contact_top_z`` of the object-table cross-view check.

Assumption (stated with every result): the object (or the stack of rows under it, down to the table) is a vertical
prism standing on the table plane, i.e. it occupies footprint x [table_z, top_z] with vertical sides and a horizontal
top, and its outline in each view is convex. Under that assumption two calibrated views fix the top height without
any monocular depth and without a footprint guess:

1. Each view's outline (the union of the masks' outer contours, ``contour_px``, undistorted to the pinhole image of
   its ``K``) is replaced by its convex hull ``H_v``.
2. Columns: a grid over the table plane where both hulls' back-projections overlap. A column is alive at height z when
   its vertical segment from the table up to z projects inside both hulls; the alive columns at z are the largest
   footprint both outlines allow for a prism of top z.
3. For each z (steps of one grid cell), the predicted outline in each view is the convex hull of that footprint's
   corners projected at the table and at z; ``fit_iou(z)`` is the mean over the two views of IoU(predicted, ``H_v``).
   A prism that is too low leaves the far parts of the outlines (the parallax of its top) unexplained; one that is too
   high has a footprint smaller than the object; the best z maximises the mean IoU. ``contact_top_z`` is that z.
4. ``contact_top_z_range``: the same fit with both outlines dilated and eroded by 1 px (the outline quantisation),
   [lowest, highest] of the three tops; null when a changed outline cannot be fitted (too small). Calibration error
   is not included: ``ray_gap_m`` of the same cross-view row measures how consistent the two calibrations are.

Grid cell (m): the coarser of the two views' ground sample distances at the region (depth / focal length), and at
least 1 / ``MAX_COLUMNS`` of the region's larger side, so a grid has at most ``MAX_COLUMNS`` ^ 2 columns. Search
ceiling: ``max_height_m`` above the table (features.object_table_max_height_m).

Synthetic check (exact outlines of boxes 20-80 mm x 20-50 mm x 15-75 mm on the SO-101 fixed camera calibration, a
wrist-like camera 0.15-0.25 m away at 30-85 deg elevation, 1.5 px rms pose error on the second view; 16 cases): top
error p50 0.6 mm, p90 2.8 mm, largest at parallax below 15 deg; the 1 px range contained the truth in 13 of 16; about
0.1 s per row. Not yet measured on the real arm (scripts/so101_perception_bench.py).

Reasons instead of a value: ``mask_at_image_border:<view>`` (an outline reaches the outermost ``BORDER_PX`` px of
the image, so part of the object may be outside the view), ``outline_missing:<view>``, ``views_do_not_overlap_on_table`` (the two outlines'
table footprints do not overlap), ``no_consistent_footprint`` (no column is inside both outlines at table height).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

#: Largest number of grid columns per side of the search region (cost bound; see module docstring).
MAX_COLUMNS = 100
#: Outline pixels this close to the image edge (px) count as touching it: a contour mapped back from a distorted
#: image is rounded to whole pixels, so an outline cut by the edge can end one pixel inside it.
BORDER_PX = 2
ASSUMPTION = ("contact_top_z: two-view fit of a vertical prism standing on the table (bottom at the table plane, "
              "vertical sides, horizontal top, convex outlines) to the two masks' outlines; for a row on other rows "
              "the stack down to the table is fitted. contact_top_z_range is the fit with the outlines grown and "
              "shrunk by 1 px; calibration error is not included (see ray_gap_m). contact_fit_iou: mean IoU of the "
              "fitted prism's outlines with the two outlines (1 = the prism explains both). A tilted, overhanging or "
              "partly hidden object breaks the assumption.")


def pinhole_polygon(camera, contour_px):
    """(N, 2) float pixels of ``contour_px`` (original image) in the pinhole image of ``camera["K"]`` (undistorted
    with ``camera["D"]``; unchanged without distortion)."""
    points = np.asarray(contour_px, dtype=np.float64).reshape(-1, 2)
    D = np.asarray(camera.get("D", []), dtype=np.float64).reshape(-1)
    if not len(points) or not D.size or not np.any(np.abs(D) > 1e-12):
        return points
    K = np.asarray(camera["K"], dtype=np.float64)
    return cv2.undistortPoints(points.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)


def outline_mask(polygons, width, height):
    """bool (height, width): the filled union of ``polygons`` (each (N, 2) px, N >= 3)."""
    canvas = np.zeros((int(height), int(width)), np.uint8)
    for polygon in polygons:
        if len(polygon) >= 3:
            cv2.fillPoly(canvas, [np.rint(np.asarray(polygon) * 4).astype(np.int32).reshape(-1, 1, 2)], 1,
                         lineType=cv2.LINE_8, shift=2)
    return canvas.astype(bool)


def _hull_mask(mask):
    ys, xs = np.nonzero(mask)
    out = np.zeros(mask.shape, np.uint8)
    if len(xs) >= 3:
        cv2.fillConvexPoly(out, cv2.convexHull(np.column_stack([xs, ys]).astype(np.int32)), 1)
    return out.astype(bool)


def _project(K, T, points):
    local = (points - T[:3, 3]) @ T[:3, :3]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.stack([K[0, 0] * local[:, 0] / local[:, 2] + K[0, 2],
                         K[1, 1] * local[:, 1] / local[:, 2] + K[1, 2]], 1), local[:, 2]


def _inside(mask, uv, depth):
    u, v = np.floor(uv[:, 0]), np.floor(uv[:, 1])
    ok = (depth > 0) & (u >= 0) & (u < mask.shape[1]) & (v >= 0) & (v < mask.shape[0])
    out = np.zeros(len(uv), bool)
    out[ok] = mask[v[ok].astype(int), u[ok].astype(int)]
    return out


def _table_points(K, T, mask, table_z_m):
    """xy (m) where the rays of the mask's hull vertices meet the table plane (None if one misses it)."""
    ys, xs = np.nonzero(mask)
    hull = cv2.convexHull(np.column_stack([xs, ys]).astype(np.float32)).reshape(-1, 2) + 0.5
    rays = np.column_stack([(hull[:, 0] - K[0, 2]) / K[0, 0], (hull[:, 1] - K[1, 2]) / K[1, 1],
                            np.ones(len(hull))]) @ T[:3, :3].T
    s = (table_z_m - T[2, 3]) / rays[:, 2]
    if not (s > 0).all():
        return None
    return T[:2, 3] + s[:, None] * rays[:, :2]


def _box(hull):
    """(hull, [x0, y0, x1, y1]): the hull mask and its pixel bounding box (x1, y1 exclusive)."""
    ys, xs = np.nonzero(hull)
    return hull, np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])


def _iou(predicted_uv, boxed):
    """IoU of the convex hull of ``predicted_uv`` (px) with a ``_box`` hull mask, on their joint bounding box."""
    hull, box = boxed
    low = np.maximum(np.floor(np.minimum(predicted_uv.min(0), box[:2])).astype(int) - 1, 0)
    high = np.minimum(np.ceil(np.maximum(predicted_uv.max(0), box[2:])).astype(int) + 1, [hull.shape[1], hull.shape[0]])
    if (high <= low).any():
        return 0.0
    canvas = np.zeros((high[1] - low[1], high[0] - low[0]), np.uint8)
    # predicted_uv is continuous (pixel i spans [i, i + 1)); the hull mask is indexed by pixel: shift by half a pixel
    cv2.fillConvexPoly(canvas, cv2.convexHull(np.rint((predicted_uv - 0.5 - low) * 4).astype(np.int32)), 1,
                       lineType=cv2.LINE_8, shift=2)
    predicted, target = canvas.astype(bool), hull[low[1]:high[1], low[0]:high[0]]
    union = int((predicted | target).sum())
    return float((predicted & target).sum()) / union if union else 0.0


def _fit(views, hulls, table_z_m, max_height_m):
    """(top_z, fit_iou) of one pair of hull masks, or a reason (str)."""
    shadows = []
    for (K, T, _), hull in zip(views, hulls):
        if hull.sum() < 3:
            return "outline_missing"
        xy = _table_points(K, T, hull, table_z_m)
        if xy is None:
            return "views_do_not_overlap_on_table"
        shadows.append(xy)
    low = np.maximum(shadows[0].min(0), shadows[1].min(0))
    high = np.minimum(shadows[0].max(0), shadows[1].max(0))
    if (high <= low).any():
        return "views_do_not_overlap_on_table"
    centre = np.array([*(low + high) / 2, table_z_m])
    sample = []
    for K, T, _ in views:
        _, depth = _project(K, T, centre[None])
        sample.append(float(depth[0]) / float(min(K[0, 0], K[1, 1])))
    cell = max(max(sample), float((high - low).max()) / MAX_COLUMNS)
    xs = np.arange(low[0] + cell / 2, high[0], cell)
    ys = np.arange(low[1] + cell / 2, high[1], cell)
    columns = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
    alive = np.ones(len(columns), bool)
    boxed = [_box(hull) for hull in hulls]
    best = None
    for k in range(int(math.floor(max_height_m / cell)) + 1):
        z = table_z_m + k * cell
        points = np.column_stack([columns[alive], np.full(int(alive.sum()), z)])
        keep = np.ones(len(points), bool)
        for (K, T, _), hull in zip(views, hulls):
            keep &= _inside(hull, *_project(K, T, points))
        alive[np.flatnonzero(alive)[~keep]] = False
        if not alive.any():
            break
        if k == 0:
            continue
        footprint = columns[alive]
        if len(footprint) >= 3:
            corners = footprint[cv2.convexHull(footprint.astype(np.float32), returnPoints=False).reshape(-1)]
        else:
            corners = footprint
        corners = np.concatenate([corners + [[cell / 2, cell / 2]], corners - [[cell / 2, cell / 2]],
                                  corners + [[cell / 2, -cell / 2]], corners + [[-cell / 2, cell / 2]]])
        prism = np.concatenate([np.column_stack([corners, np.full(len(corners), table_z_m)]),
                                np.column_stack([corners, np.full(len(corners), z)])])
        score = float(np.mean([_iou(_project(K, T, prism)[0], item) for (K, T, _), item in zip(views, boxed)]))
        if best is None or score > best[1]:
            best = (z, score)
    if best is None:
        return "no_consistent_footprint"
    return best


def contact_top(views, table_z_m, max_height_m):
    """``dict(contact_top_z, contact_top_z_range, contact_fit_iou)`` (m, m, IoU 0..1) of two views of one object, or
    ``dict(contact_reason=...)`` (module docstring).

    ``views``: two ``(K, T, polygons)``: pinhole ``K`` (3x3 px), ``T`` (4x4 T_world_camera, m) and the object's
    outlines in that pinhole image (list of (N, 2) px arrays, ``pinhole_polygon``), each with ``(width, height)`` of
    the image as a 4th item. ``table_z_m``: table plane (m). ``max_height_m``: search ceiling above the table (m).
    Example: ``contact_top([(K1, T1, [poly1], (640, 480)), (K2, T2, [poly2], (640, 480))], -0.016, 0.3)``."""
    masks = []
    for index, (K, T, polygons, size) in enumerate(views):
        mask = outline_mask(polygons, *size)
        if not mask.any():
            return dict(contact_reason="outline_missing:" + ("source" if index == 0 else "target"))
        # the outermost BORDER_PX rows/columns: an outline mapped back from a distorted image is rounded to pixels
        if (mask[:BORDER_PX].any() or mask[-BORDER_PX:].any() or mask[:, :BORDER_PX].any()
                or mask[:, -BORDER_PX:].any()):
            return dict(contact_reason="mask_at_image_border:" + ("source" if index == 0 else "target"))
        masks.append(mask)
    plain = [(np.asarray(K, float), np.asarray(T, float), None) for K, T, _, _ in views]
    kernel = np.ones((3, 3), np.uint8)
    best, tops = None, []
    for change in (0, 1, -1):
        changed = [mask if change == 0 else
                   (cv2.dilate if change > 0 else cv2.erode)(mask.astype(np.uint8), kernel).astype(bool)
                   for mask in masks]
        found = _fit(plain, [_hull_mask(mask) for mask in changed], float(table_z_m), float(max_height_m))
        if change == 0:
            if isinstance(found, str):
                return dict(contact_reason=found)
            best = found
        elif isinstance(found, str):   # an outline 1 px larger or smaller cannot be fitted: no range, not [z, z]
            tops = None
            break
        tops.append(found[0])
    return dict(contact_top_z=round(best[0], 3),
                contact_top_z_range=None if tops is None else [round(min(tops), 3), round(max(tops), 3)],
                contact_fit_iou=round(best[1], 2))
