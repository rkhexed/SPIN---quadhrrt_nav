#!/usr/bin/env python3
# coding=utf-8
"""
planner_core.py
---------------
QuadHRRT* sampling-based planner, copied almost verbatim from
passion-project-aab/rrt_virtual_mover.py.

Contains: QuadTreeNode, QuadTree, RRTNode, QuadRRTPlanner.

NOTE: their `OccupancyMap` is intentionally NOT copied. The planner reaches the
map only through `self.map`, whose method surface is provided instead by
`CostmapAdapter` (see costmap_adapter.py), which is backed by the live Nav2
costmap. The planner code below is unchanged apart from dropping the
`OccupancyMap` type annotation on `QuadRRTPlanner.__init__`.
"""

import math
import time
import random

import numpy as np


# ───────────────────────── CONFIG ─────────────────────────────
# RRT parameters
MAX_ITERATIONS   = 5000
STEP_SIZE        = 0.30
GOAL_BIAS        = 0.10    # kept only as fallback
GOAL_TOLERANCE   = 0.5
USE_RRT_STAR     = True
RRT_STAR_RADIUS  = 1.0

# HMA-RRT* parameters
CORRIDOR_FACTOR      = 0.4
MIN_CORRIDOR_WIDTH   = 2.0
GOAL_REGION_RADIUS   = 1.5
CORRIDOR_SAMPLE_RATE = 0.80
GOAL_SAMPLE_RATE     = 0.15
RANDOM_SAMPLE_RATE   = 0.05

# Grid-based sampling (Component 1 upgrade — Section 3.1)
GRID_N           = 10      # 10×10 grid over entire map
GRID_DELTA       = 0.1     # attenuation factor for selection count
GRID_P_BASE      = 0.05    # base probability every cell starts with
# ──────────────────────────────────────────────────────────────


# ============================================================
# Quad RRT
# ============================================================
class QuadTreeNode:
    def __init__(self, x, y, node_index):
        self.x = x
        self.y = y
        self.node_index = node_index


class QuadTree:
    MAX_POINTS = 8

    def __init__(self, x, y, w, h, depth=0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.depth = depth

        self.points = []

        self.divided = False

        self.nw = None
        self.ne = None
        self.sw = None
        self.se = None

    def contains(self, x, y):
        return (
            self.x <= x < self.x + self.w and
            self.y <= y < self.y + self.h
        )

    def subdivide(self):
        hw = self.w / 2
        hh = self.h / 2

        self.nw = QuadTree(self.x, self.y, hw, hh, self.depth + 1)
        self.ne = QuadTree(self.x + hw, self.y, hw, hh, self.depth + 1)
        self.sw = QuadTree(self.x, self.y + hh, hw, hh, self.depth + 1)
        self.se = QuadTree(self.x + hw, self.y + hh, hw, hh, self.depth + 1)

        self.divided = True

    def insert(self, point):
        if not self.contains(point.x, point.y):
            return False

        if len(self.points) < self.MAX_POINTS:
            self.points.append(point)
            return True

        if not self.divided:
            self.subdivide()

        return (
            self.nw.insert(point) or
            self.ne.insert(point) or
            self.sw.insert(point) or
            self.se.insert(point)
        )

    def query_radius(self, x, y, radius, found=None):
        if found is None:
            found = []

        if not self._intersects_circle(x, y, radius):
            return found

        r2 = radius * radius

        for p in self.points:
            dx = p.x - x
            dy = p.y - y
            if dx*dx + dy*dy <= r2:
                found.append(p.node_index)

        if self.divided:
            self.nw.query_radius(x, y, radius, found)
            self.ne.query_radius(x, y, radius, found)
            self.sw.query_radius(x, y, radius, found)
            self.se.query_radius(x, y, radius, found)

        return found

    def nearest(self, x, y, best=None):

    # Check points in current node
        for p in self.points:
            d = (p.x - x)**2 + (p.y - y)**2

            if best is None or d < best[0]:
             best = (d, p.node_index)

    # If subdivided → search children intelligently
        if self.divided:

            children = [self.nw, self.ne, self.sw, self.se]

            # Sort children by proximity to query point
            children.sort(
                key=lambda child:
                child.distance_to_boundary(x, y)
            )

            for child in children:

                # If this quadrant cannot contain closer point → skip
                if best is not None:
                    min_possible_dist = child.distance_to_boundary(x, y)

                    if min_possible_dist > best[0]:
                        continue

                best = child.nearest(x, y, best)

        return best

    def _intersects_circle(self, x, y, r):
        nearest_x = max(self.x, min(x, self.x + self.w))
        nearest_y = max(self.y, min(y, self.y + self.h))

        dx = x - nearest_x
        dy = y - nearest_y

        return dx*dx + dy*dy <= r*r

    def distance_to_boundary(self, x, y):
        """
        Minimum squared distance from point (x,y)
        to this QuadTree region.
        """

        dx = 0.0
        dy = 0.0

        if x < self.x:
            dx = self.x - x
        elif x > self.x + self.w:
            dx = x - (self.x + self.w)

        if y < self.y:
            dy = self.y - y
        elif y > self.y + self.h:
            dy = y - (self.y + self.h)

        return dx*dx + dy*dy


# ══════════════════════════════════════════════════════════════
#  RRT / RRT* Planner
# ══════════════════════════════════════════════════════════════
class RRTNode:
    __slots__ = ("x", "y", "parent", "cost")

    def __init__(self, x, y, parent=None, cost=0.0):
        self.x      = x
        self.y      = y
        self.parent = parent   # index into node list
        self.cost   = cost     # RRT* path cost from root


class QuadRRTPlanner:

    def __init__(self, occ_map):
        self.map = occ_map
        self.quadtree = None
        self.latest_nodes = []
        print("[INFO] QuadRRTPlanner initialized")
        # HMA-RRT* metrics
        self.plan_time = 0.0
        self.total_nodes = 0

        self.raw_path_length = 0.0
        self.smoothed_path_length = 0.0

        self.goal_samples = 0
        self.corridor_samples = 0
        self.global_samples = 0

        self.rewire_count = 0
        # Grid sampling state — reset each plan() call
        self.grid_probs  = {}   # normalized probability per (i,j) cell
        self.grid_counts = {}   # how many times each cell has been sampled
        self.grid_nc     = 0    # obstacle count intersecting start-goal line




    # ══════════════════════════════════════════════════════
#  Component 1 — Grid-Based Dynamic Sampling (Paper §3.1)
# ══════════════════════════════════════════════════════

    def _point_to_line_dist(self, px, py, x1, y1, x2, y2):
        """
        Perpendicular distance from point (px,py) to the
        finite line segment (x1,y1)→(x2,y2).
        Used in Equation 9 to compute d_ij for every grid cell.
        """
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return math.hypot(px - x1, py - y1)
        t = ((px - x1)*dx + (py - y1)*dy) / (dx*dx + dy*dy)
        t = max(0.0, min(1.0, t))
        return math.hypot(px - (x1 + t*dx), py - (y1 + t*dy))


    def _obstacle_ratio_in_cell(self, cx, cy, cw, ch, obs_mask):
        """
        Fraction of a grid cell covered by obstacle pixels.
        Returns A_ij used in Equation 11.
        cx, cy = world-coord bottom-left corner of cell
        cw, ch = cell width and height in metres
        """
        # Convert cell world corners to map pixel indices
        x0 = int((cx - self.map.origin_x) / self.map.res)
        y0 = int((cy - self.map.origin_y) / self.map.res)
        x1 = int((cx + cw - self.map.origin_x) / self.map.res)
        y1 = int((cy + ch - self.map.origin_y) / self.map.res)

        # Clamp to map bounds
        x0 = max(0, x0);  y0 = max(0, y0)
        x1 = min(obs_mask.shape[0] - 1, x1)
        y1 = min(obs_mask.shape[1] - 1, y1)

        if x1 <= x0 or y1 <= y0:
            return 0.0

        region = obs_mask[x0:x1, y0:y1]
        return float(np.sum(region)) / float(region.size + 1e-6)


    def _count_line_obstacle_intersections(self, sx, sy, gx, gy, obs_mask):
        """
        Count how many distinct obstacle clusters the straight
        start→goal line passes through.
        This gives n_c in Equation 7, used to compute σ_k (Eq. 8).
        """
        steps = int(math.hypot(gx - sx, gy - sy) / self.map.res)
        steps = max(steps, 1)
        prev_hit = False
        nc = 0
        for k in range(steps + 1):
            t  = k / steps
            wx = sx + t * (gx - sx)
            wy = sy + t * (gy - sy)
            cx, cy = self.map.world_to_cell(wx, wy)
            if self.map.in_bounds(cx, cy) and obs_mask[cx, cy]:
                if not prev_hit:
                    nc += 1       # entering a new obstacle cluster
                prev_hit = True
            else:
                prev_hit = False
        return nc

    def _init_sampling_grid(self, sx, sy, gx, gy, obs_mask):
        """
        Build the full n×n probability table once per plan() call.
        Implements Equations 7–13 from Section 3.1 of the paper.

        After this call, self.grid_probs[(i,j)] holds the normalized
        probability of sampling from cell (i,j).
        self.grid_counts is reset to zero for all cells.
        """
        n     = GRID_N
        ox    = self.map.origin_x          # world x of map left edge
        oy    = self.map.origin_y          # world y of map bottom edge
        mw    = self.map.w * self.map.res  # total map width  in metres
        mh    = self.map.h * self.map.res  # total map height in metres
        cw    = mw / n                     # single cell width  in metres
        ch    = mh / n                     # single cell height in metres

        L = math.hypot(gx - sx, gy - sy)  # start-goal line length

        # ── Eq. 7: count obstacle clusters on direct line ──
        nc    = self._count_line_obstacle_intersections(sx, sy, gx, gy, obs_mask)
        nmax  = max(1, nc)             # at least 1 to avoid div-by-zero

        # ── Eq. 8: steepness of distance-bias function ──
        # When nc is large (many obstacles on line), sigma_k → 0
        # meaning the distance falloff is gentle → samples spread wide
        # When nc is 0 (clear line), sigma_k = 1 → tight corridor
        sigma_k = 1.0 if nc == 0 else (1.0 - nc / (nmax + 1))

        raw_probs = {}
        self.grid_counts = {}

        for i in range(n):
            for j in range(n):
                # World coord of this cell's centre
                cell_cx = ox + (i + 0.5) * cw
                cell_cy = oy + (j + 0.5) * ch

                # ── Eq. 9: normalised perpendicular distance to line ──
                d_ij = self._point_to_line_dist(
                    cell_cx, cell_cy, sx, sy, gx, gy) / (L / 2.0 + 1e-6)

                # ── Eq. 10: distance-bias probability ──
                # Sigmoid: cells near the line get ~1.0, far cells get ~0.0
                P_line = 1.0 / (1.0 + math.exp(sigma_k * (d_ij - 1.0)))

                # ── Eq. 11: obstacle-area penalty ──
                # High obstacle ratio → low probability
                A_ij  = self._obstacle_ratio_in_cell(
                    ox + i*cw, oy + j*ch, cw, ch, obs_mask)
                P_area = math.exp(-A_ij)

                # ── Eq. 12: combined probability (no prior selections yet) ──
                # C_ij = 0 at start, so e^(-delta*0) = 1 → no attenuation yet
                C_ij  = 0
                P_ij  = (GRID_P_BASE + P_line * math.exp(-GRID_DELTA * C_ij)) \
                        * P_area

                raw_probs[(i, j)]    = max(P_ij, 1e-9)
                self.grid_counts[(i, j)] = 0

        # ── Eq. 13: normalise so all cells sum to 1.0 ──
        total = sum(raw_probs.values())
        self.grid_probs = {k: v / total for k, v in raw_probs.items()}

    def _sample_from_grid(self):
    # ── Explicit goal bias FIRST (replaces old GOAL_BIAS) ──
        if random.random() < GOAL_SAMPLE_RATE:   # 15%
            theta  = random.uniform(0, 2 * math.pi)
            radius = random.uniform(0, GOAL_REGION_RADIUS)
            self.goal_samples += 1
            return (
                self._plan_gx + radius * math.cos(theta),
                self._plan_gy + radius * math.sin(theta)
            )

        # ── Remaining 85% → grid-based sampling ──
        n  = GRID_N
        ox = self.map.origin_x
        oy = self.map.origin_y
        cw = (self.map.w * self.map.res) / n
        ch = (self.map.h * self.map.res) / n

        keys    = list(self.grid_probs.keys())
        weights = [self.grid_probs[k] for k in keys]
        chosen  = random.choices(keys, weights=weights, k=1)[0]
        i, j    = chosen

        self.grid_counts[(i, j)] += 1
        self.grid_probs[(i, j)]  *= math.exp(-GRID_DELTA)

        total = sum(self.grid_probs.values())
        if total < 0.5:
            self.grid_probs = {k: v/total for k, v in self.grid_probs.items()}

        # Check if this cell is near goal — count accordingly
        cell_cx = ox + (i + 0.5) * cw
        cell_cy = oy + (j + 0.5) * ch
        if math.hypot(cell_cx - self._plan_gx,
                    cell_cy - self._plan_gy) <= GOAL_REGION_RADIUS:
            self.goal_samples += 1
        else:
            self.corridor_samples += 1

        rx = ox + (i + random.random()) * cw
        ry = oy + (j + random.random()) * ch
        return rx, ry

    # ── public API ─────────────────────────────────────────────

    def adaptive_sample(
            self,
        sx, sy,
        gx, gy,
        wx_min, wx_max,
        wy_min, wy_max,
        nodes
        ):
        """
        HMA-RRT* Dynamic Region-Based Sampling

        Samples mostly inside a corridor between
        start and goal.
        """

        r = random.random()

        # ===================================================
        # Goal Region Sampling
        # ===================================================
        if r < GOAL_SAMPLE_RATE:

            theta = random.uniform(
                0.0,
                2.0 * math.pi
            )

            radius = random.uniform(
                0.0,
                GOAL_REGION_RADIUS
            )
            self.goal_samples += 1

            return (
                gx + radius * math.cos(theta),
                gy + radius * math.sin(theta)
            )

        # ===================================================
        # Corridor Sampling
        # ===================================================
        elif r < GOAL_SAMPLE_RATE + CORRIDOR_SAMPLE_RATE:

            dx = gx - sx
            dy = gy - sy

            dist = math.hypot(dx, dy)

            corridor_width = max(
                MIN_CORRIDOR_WIDTH,
                dist * CORRIDOR_FACTOR
            )

            t = random.random()

            base_x = sx + t * dx
            base_y = sy + t * dy

            if dist > 0.001:

                nx = -dy / dist
                ny = dx / dist

                offset = random.uniform(
                    -corridor_width / 2.0,
                    corridor_width / 2.0
                )

                sample_x = base_x + offset * nx
                sample_y = base_y + offset * ny

            else:

                sample_x = sx
                sample_y = sy

            sample_x = max(
                wx_min,
                min(sample_x, wx_max)
            )

            sample_y = max(
                wy_min,
                min(sample_y, wy_max)
            )
            self.corridor_samples += 1

            return sample_x, sample_y

        # ===================================================
        # Global Random Sampling
        # ===================================================
        else:
            self.global_samples += 1
            return (
                random.uniform(wx_min, wx_max),
                random.uniform(wy_min, wy_max)
            )

    def path_length(self, path):
        if len(path) < 2:
            return 0.0

        total = 0.0

        for i in range(len(path)-1):
            total += math.hypot(
                path[i+1][0] - path[i][0],
                path[i+1][1] - path[i][1]
            )

        return total


    def plan(self, start_world, goal_world):
        """
        Run RRT (or RRT* if USE_RRT_STAR=True).
        Returns list of (x, y) world-coord waypoints, or [] on failure.
        """
        self.goal_samples = 0
        self.corridor_samples = 0
        self.global_samples = 0

        self.rewire_count = 0

        self.plan_time = 0.0
        self.total_nodes = 0

        self.raw_path_length = 0.0
        self.smoothed_path_length = 0.0

        obs = self.map.inflated_mask()

        start_time = time.time()

        # Map bounds in world coordinates
        wx_min = self.map.origin_x
        wx_max = self.map.origin_x + self.map.w * self.map.res

        wy_min = self.map.origin_y
        wy_max = self.map.origin_y + self.map.h * self.map.res

        # Create QuadTree
        self.quadtree = QuadTree(
            wx_min,
            wy_min,
            wx_max - wx_min,
            wy_max - wy_min
        )

        sx, sy = start_world
        gx, gy = goal_world

        self._plan_gx = gx
        self._plan_gy = gy

        # Validate start / goal
        scx, scy = self.map.world_to_cell(sx, sy)
        gcx, gcy = self.map.world_to_cell(gx, gy)

        if not self.map.in_bounds(scx, scy):
            return []
        if not self.map.in_bounds(gcx, gcy) or obs[gcx, gcy]:
            gcx, gcy = self._free_near(gcx, gcy, obs)
            if gcx is None:
                return []
            gx, gy = self.map.cell_to_world(gcx, gcy)

        # Map bounds in world coords
        wx_min = self.map.origin_x
        wx_max = self.map.origin_x + self.map.w * self.map.res
        wy_min = self.map.origin_y
        wy_max = self.map.origin_y + self.map.h * self.map.res

        nodes = [RRTNode(sx, sy, parent=None, cost=0.0)]
        self.quadtree.insert(
        QuadTreeNode(sx, sy, 0)
        )
        goal_node_idx = None
        # ── Component 1: build grid probability table once ──
        self._init_sampling_grid(sx, sy, gx, gy, obs)


        for _ in range(MAX_ITERATIONS):
            # Sample

           # Sample — Component 1: grid-based dynamic sampling
            rx, ry = self._sample_from_grid()

            # Nearest node
            nearest_idx = self.quadtree.nearest(rx, ry)[1]
            nearest     = nodes[nearest_idx]

            # Steer
            nx, ny = self._steer(nearest.x, nearest.y, rx, ry)

            # Collision check
            if not self._collision_free(nearest.x, nearest.y, nx, ny, obs):
                continue

            new_cost = nearest.cost + math.hypot(nx - nearest.x,
                                                  ny - nearest.y)

            if USE_RRT_STAR:
                # RRT*: find neighbours and choose best parent
                new_node, parent_idx = self._choose_parent(
                    nodes, nx, ny, new_cost, obs)
                new_idx = len(nodes)
                nodes.append(new_node)
                self.quadtree.insert(
                QuadTreeNode(nx, ny, len(nodes)-1)
                )
                # Rewire
                self._rewire(nodes, new_idx, obs)
            else:
                new_node = RRTNode(nx, ny, parent=nearest_idx, cost=new_cost)
                nodes.append(new_node)
                self.quadtree.insert(
                QuadTreeNode(nx, ny, len(nodes)-1)
                )


            # Goal check
            if ( math.hypot(nx-gx, ny-gy) <= GOAL_TOLERANCE and self._collision_free(nx, ny, gx, gy, obs)):
                goal_node_idx = len(nodes) - 1
                break

        if goal_node_idx is None:
            return []


        path = self._extract_path(nodes, goal_node_idx)
        raw_length = self.path_length(path)
        # Step 1: averaging smooth
        path = self.smooth_path(path)

        # Step 2: shortcut smooth (NEW)
        path = self.shortcut_smooth(path)
        self.latest_nodes = nodes
        elapsed = time.time() - start_time
        smooth_length = self.path_length(path)
        self.plan_time = elapsed
        self.total_nodes = len(nodes)

        self.raw_path_length = raw_length
        self.smoothed_path_length = smooth_length
        print(f"[RRT*] Time: {elapsed:.3f}s | Nodes: {len(nodes)}")

        print(
            f"[Sampling Stats] "
            f"Goal={self.goal_samples}, "
            f"Corridor={self.corridor_samples}, "
            f"Global={self.global_samples}"
        )
        print(
        f"[Optimization] "
        f"{raw_length:.2f}m -> {smooth_length:.2f}m"
        )
        print(f"[RRT*] Rewires: {self.rewire_count}")

        return path

    def get_tree_edges(self, nodes):
        """Return list of (x0,y0,x1,y1) for RViz tree visualisation."""
        edges = []
        for i, node in enumerate(nodes):
            if node.parent is not None:
                p = nodes[node.parent]
                edges.append((p.x, p.y, node.x, node.y))
        return edges

    # ── internal helpers ───────────────────────────────────────

    def _nearest(self, nodes, rx, ry):
        dists = [(math.hypot(n.x - rx, n.y - ry), i)
                 for i, n in enumerate(nodes)]
        return min(dists)[1]

    def _steer(self, fx, fy, tx, ty):
        d = math.hypot(tx - fx, ty - fy)
        if d <= STEP_SIZE:
            return tx, ty
        ratio = STEP_SIZE / d
        return fx + ratio * (tx - fx), fy + ratio * (ty - fy)



    def _collision_free(self, x0, y0, x1, y1, obs):
        """Check line segment for collisions using Bresenham."""
        cx0, cy0 = self.map.world_to_cell(x0, y0)
        cx1, cy1 = self.map.world_to_cell(x1, y1)
        for cx, cy in self.map._bresenham(cx0, cy0, cx1, cy1):
            if not self.map.in_bounds(cx, cy):
                return False
            if obs[cx, cy]:
                return False
        return True

    def _choose_parent(self, nodes, nx, ny, default_cost, obs):
        """RRT*: pick parent that gives lowest cost."""
        best_parent = None
        best_cost   = float('inf')
        nearby = self.quadtree.query_radius(
        nx, ny, RRT_STAR_RADIUS
        )

        radius = max(0.5,min(RRT_STAR_RADIUS,2.0 * math.sqrt(math.log(len(nodes)+1)/(len(nodes)+1))))
        for i in nearby:
            node = nodes[i]
            d = math.hypot(node.x - nx, node.y - ny)

            if d > radius:
                continue
            if not self._collision_free(node.x, node.y, nx, ny, obs):
                continue
            c = node.cost + d
            if c < best_cost:
                best_cost   = c
                best_parent = i
        if best_parent is None:
            # Fall back to nearest
            best_parent = self._nearest(nodes, nx, ny)
            p = nodes[best_parent]
            best_cost = p.cost + math.hypot(p.x - nx, p.y - ny)
        return RRTNode(nx, ny, parent=best_parent, cost=best_cost), best_parent

    def _rewire(self, nodes, new_idx, obs):
        """RRT*: check if routing through new_node shortens neighbour costs."""
        new_node = nodes[new_idx]
        nearby = self.quadtree.query_radius(
        new_node.x,
        new_node.y,
        RRT_STAR_RADIUS
        )

        radius = max(0.5,min(RRT_STAR_RADIUS,2.0 * math.sqrt(math.log(len(nodes)+1)/(len(nodes)+1))))
        for i in nearby:
            node = nodes[i]
            if i == new_idx or i == new_node.parent:
                continue
            d = math.hypot(node.x - new_node.x, node.y - new_node.y)

            if d > radius:
                continue
            new_cost = new_node.cost + d
            if new_cost < node.cost and \
               self._collision_free(new_node.x, new_node.y,
                                    node.x, node.y, obs):
                node.parent = new_idx
                self.rewire_count += 1
                node.cost   = new_cost
                self._update_children_costs(nodes, i)

    def _update_children_costs(self, nodes, parent_idx):
        """
        Recursively update costs of all descendants
        after a rewire operation.
        """

        parent = nodes[parent_idx]

        for i, node in enumerate(nodes):

            if node.parent == parent_idx:

                edge_cost = math.hypot(
                    node.x - parent.x,
                    node.y - parent.y
                )

                node.cost = parent.cost + edge_cost

                # Recursively update grandchildren
                self._update_children_costs(nodes, i)

    def _extract_path(self, nodes, goal_idx):
        path = []
        idx  = goal_idx
        while idx is not None:
            path.append((nodes[idx].x, nodes[idx].y))
            idx = nodes[idx].parent
        path.reverse()
        return path

    def _free_near(self, gcx, gcy, obs, search=20):
        """Find nearest free cell to a blocked goal cell."""
        for r in range(1, search):
            for ddx in range(-r, r+1):
                for ddy in range(-r, r+1):
                    nx, ny = gcx+ddx, gcy+ddy
                    if self.map.in_bounds(nx, ny) and not obs[nx, ny]:
                        return nx, ny
        return None, None

    def smooth_path(self, waypoints, iterations=50):
        """Collision-aware smoothing: average neighbours, but reject any move
        that would push a segment through an obstacle (keeps the raw point)."""
        if len(waypoints) < 3:
            return waypoints
        obs = self.map.inflated_mask()
        pts = list(waypoints)
        for _ in range(iterations):
            for i in range(1, len(pts) - 1):
                cand = (
                    (pts[i-1][0] + pts[i][0] + pts[i+1][0]) / 3.0,
                    (pts[i-1][1] + pts[i][1] + pts[i+1][1]) / 3.0,
                )
                if (self._collision_free(pts[i-1][0], pts[i-1][1], cand[0], cand[1], obs)
                        and self._collision_free(cand[0], cand[1], pts[i+1][0], pts[i+1][1], obs)):
                    pts[i] = cand
        return pts

    def shortcut_smooth(self, path, iterations=50):
        """Remove unnecessary waypoints by connecting distant nodes directly."""
        if len(path) < 3:
            return path

        import random
        new_path = list(path)

        obs = self.map.inflated_mask()
        for _ in range(iterations):
            i = random.randint(0, len(new_path) - 2)
            j = random.randint(i + 1, len(new_path) - 1)

            x1, y1 = new_path[i]
            x2, y2 = new_path[j]
            if self._collision_free(x1, y1, x2, y2, obs):
                new_path = new_path[:i+1] + new_path[j:]
        return new_path
