#!/usr/bin/env python3
# coding=utf-8
"""
costmap_adapter.py
------------------
CostmapAdapter — a thin drop-in replacement for the RRT planner's original
`OccupancyMap`. It exposes the EXACT method/attribute surface that
`QuadRRTPlanner` calls on `self.map`, but backed by the live Nav2 global
costmap (a `nav_msgs/OccupancyGrid`) instead of a self-built /scan map.

Because the surface matches, the planner code in planner_core.py needs zero
changes.

Surface used by the planner (from grep of rrt_virtual_mover.py):
    world_to_cell(x, y)      -> (cx, cy)
    cell_to_world(cx, cy)    -> (x, y)   world coord of cell centre
    in_bounds(cx, cy)        -> bool
    inflated_mask()          -> 2D bool array, True = blocked, indexed
                                [cy, cx] = [row, col] = [y, x] (row-major)
    _bresenham(x0,y0,x1,y1)  -> generator of (cx, cy) cells along a line
    origin_x, origin_y       -> map origin in world frame
    res                      -> metres / cell
    w, h                     -> width / height in cells

CRITICAL — no double inflation:
    The original OccupancyMap.inflated_mask() dilated obstacles by INFLATION_M.
    The Nav2 costmap ALREADY contains an inflation layer, so we must NOT inflate
    again. Here inflated_mask() is a pure binary threshold of the costmap — no
    dilation. Double-inflating would make the robot think everything is blocked.
"""

import numpy as np


# Nav2 publishes /global_costmap/costmap as nav_msgs/OccupancyGrid with values
# scaled to 0..100 (plus -1 = unknown). Nav2's cost_translation maps the raw
# costmap's inscribed-inflated value (253) -> 99 and lethal (254) -> 100.
# So thresholding OccupancyGrid at >= 99 is equivalent to raw cost >= 253.
LETHAL_THRESHOLD = 99


class CostmapAdapter:
    def __init__(self, lethal_threshold=LETHAL_THRESHOLD,
                 unknown_is_blocked=False):
        # Metadata (populated on first costmap message)
        self.res      = None
        self.w        = 0
        self.h        = 0
        self.origin_x = 0.0
        self.origin_y = 0.0

        self._lethal_threshold  = lethal_threshold
        self._unknown_is_blocked = unknown_is_blocked

        # Cost grid stored as int16 in the natural ROS row-major layout,
        # shape (h, w), indexed self._cost[cy, cx] = [row, col] = [y, x].
        # (No transpose at the source — the planner's readers use [cy, cx].)
        self._cost = None
        self._mask = None   # cached inflated_mask() result, invalidated on update

    # ── ingest ──────────────────────────────────────────────────
    def update_from_msg(self, msg):
        """Store the latest nav_msgs/OccupancyGrid. Cheap; called on each msg."""
        info = msg.info
        self.res      = info.resolution
        self.w        = info.width
        self.h        = info.height
        self.origin_x = info.origin.position.x
        self.origin_y = info.origin.position.y

        # OccupancyGrid.data is row-major: index = row * width + col.
        # Keep that layout: reshape -> [row, col] == [cy, cx] == [y, x].
        # Readers in planner_core.py index the mask as obs[cy, cx] to match.
        data = np.asarray(msg.data, dtype=np.int16).reshape(self.h, self.w)
        self._cost = data
        self._mask = None   # invalidate cache

    def ready(self):
        """True once at least one costmap has been received."""
        return self._cost is not None and self.res is not None

    # ── planner surface (mirrors OccupancyMap) ──────────────────
    def world_to_cell(self, wx, wy):
        cx = int((wx - self.origin_x) / self.res)
        cy = int((wy - self.origin_y) / self.res)
        return cx, cy

    def cell_to_world(self, cx, cy):
        wx = self.origin_x + (cx + 0.5) * self.res
        wy = self.origin_y + (cy + 0.5) * self.res
        return wx, wy

    def in_bounds(self, cx, cy):
        return 0 <= cx < self.w and 0 <= cy < self.h

    def inflated_mask(self):
        """
        Binary blocked mask, True = blocked. Indexed [cy, cx] = [row, col]
        = [y, x] (row-major — same layout as the source OccupancyGrid).

        Pure threshold of the ALREADY-inflated Nav2 costmap — no extra dilation
        (see module docstring: avoid double inflation).
        """
        if self._mask is not None:
            return self._mask

        blocked = self._cost >= self._lethal_threshold
        if self._unknown_is_blocked:
            blocked |= (self._cost < 0)   # -1 == unknown
        self._mask = blocked
        return self._mask

    def _bresenham(self, x0, y0, x1, y1):
        """Cells along a grid line — copied verbatim from OccupancyMap."""
        dx, dy = abs(x1-x0), abs(y1-y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            yield x0, y0
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy; x0 += sx
            if e2 < dx:
                err += dx; y0 += sy
