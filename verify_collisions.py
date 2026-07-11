#!/usr/bin/env python3
"""
verify_collisions.py — verify the QuadHRRT* planner produces collision-free paths.

Runs WITHOUT ROS2 (pure numpy/PIL/yaml). Loads a map .pgm/.yaml straight into the
CostmapAdapter, plans a path, and reports collisions at each stage (raw / smoothed /
final) using an INDEPENDENT dense checker.

ACCEPTANCE CRITERION:
  Stage "1. RAW (from RRT* tree)" MUST report 0 collisions.
  If RAW has collisions, the tree-building collision check (_collision_free / obs
  indexing) is still wrong — keep fixing.

USAGE (from the repo root, or adjust --repo and --map):
  python3 verify_collisions.py \
      --repo   /path/to/SPIN---quadhrrt_nav \
      --map    /path/to/ros2_capstone_SPI/src/turtlebot4/turtlebot4_navigation/maps/maze.yaml \
      --start -8 -8 --goal 8 8

The independent checker below indexes the mask as mask[row, col] = mask[y, x],
which is the CORRECT row-major convention for an OccupancyGrid built height x width.
"""
import argparse, math, os, sys
import numpy as np
import yaml
from PIL import Image


def load_mask_and_meta(map_yaml):
    y = yaml.safe_load(open(map_yaml))
    img = y['image']
    if not os.path.isabs(img):
        img = os.path.join(os.path.dirname(map_yaml), img)
    px = np.array(Image.open(img).convert('L'))
    h, w = px.shape
    res = float(y['resolution']); ox, oy = map(float, y['origin'][:2])
    occ_th = float(y.get('occupied_thresh', 0.65))
    free_th = float(y.get('free_thresh', 0.196))
    negate = int(y.get('negate', 0))
    p = px.astype(np.float32) / 255.0
    if negate:
        p = 1.0 - p
    prob_occ = 1.0 - p
    occ = np.full((h, w), -1, np.int8)
    occ[prob_occ >= occ_th] = 100
    occ[prob_occ <= free_th] = 0
    occ = np.flipud(occ)                     # ROS bottom-left origin
    data = occ.flatten().astype(np.int8).tolist()

    class I: pass
    class M: pass
    m = M(); m.info = I()
    m.info.resolution = res; m.info.width = w; m.info.height = h
    m.info.origin = type('O', (), {})(); m.info.origin.position = type('P', (), {})()
    m.info.origin.position.x = ox; m.info.origin.position.y = oy
    m.data = data
    return m, (w, h, res, ox, oy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True, help='path to SPIN---quadhrrt_nav')
    ap.add_argument('--map', required=True, help='path to map .yaml')
    ap.add_argument('--start', nargs=2, type=float, default=[-8.0, -8.0])
    ap.add_argument('--goal',  nargs=2, type=float, default=[8.0, 8.0])
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.repo))
    from quadhrrt_nav.costmap_adapter import CostmapAdapter
    from quadhrrt_nav.planner_core import QuadRRTPlanner

    msg, (w, h, res, ox, oy) = load_mask_and_meta(args.map)
    a = CostmapAdapter(); a.update_from_msg(msg)

    # INDEPENDENT ground-truth mask, indexed [row=y, col=x]
    gt = np.array(a.inflated_mask())
    # Make an unambiguous [y,x] ground truth directly from the message too:
    grid = np.array(msg.data, dtype=np.int8).reshape(h, w)   # row-major [y,x]
    gt_yx = grid >= 99                                        # True = blocked, [y,x]

    def count_bad(path, label):
        bad = tot = 0
        for i in range(len(path) - 1):
            x0, y0 = path[i]; x1, y1 = path[i + 1]
            d = math.hypot(x1 - x0, y1 - y0); n = max(2, int(d / 0.02))
            for t in range(n + 1):
                x = x0 + (x1 - x0) * t / n
                yy = y0 + (y1 - y0) * t / n
                cx, cyy = a.world_to_cell(x, yy)
                if 0 <= cyy < h and 0 <= cx < w:
                    tot += 1
                    if gt_yx[cyy, cx]:
                        bad += 1
        print(f"  {label}: {bad}/{tot} collisions, {len(path)} waypoints")
        return bad

    p = QuadRRTPlanner(a)
    orig_smooth = p.smooth_path
    orig_short = p.shortcut_smooth
    stages = {}

    def cap_smooth(wp, *A, **K):
        stages['raw'] = list(wp)
        out = orig_smooth(wp, *A, **K)
        stages['after_smooth'] = list(out)
        return out

    def cap_short(wp, *A, **K):
        out = orig_short(wp, *A, **K)
        stages['after_shortcut'] = list(out)
        return out

    p.smooth_path = cap_smooth
    p.shortcut_smooth = cap_short

    path = p.plan(tuple(args.start), tuple(args.goal))
    print("Collisions at each stage (independent [y,x] ground truth):")
    raw_bad = count_bad(stages.get('raw', []), "1. RAW (from RRT* tree)")
    count_bad(stages.get('after_smooth', []), "2. after smooth_path")
    count_bad(stages.get('after_shortcut', path), "3. after shortcut_smooth (FINAL)")

    print()
    if raw_bad == 0:
        print("PASS: raw path is collision-free. Tree-building collision check is correct.")
    else:
        print(f"FAIL: raw path has {raw_bad} collisions. The obs[...] indexing in "
              "_collision_free / planner_core.py is still wrong (should be [cy,cx], "
              "matching row-major [y,x]).")


if __name__ == '__main__':
    main()
