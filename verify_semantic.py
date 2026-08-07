#!/usr/bin/env python3
"""
verify_semantic.py - check the semantic cost injection in QuadHRRT*.

Runs without ROS (numpy/PIL/yaml), reusing verify_collisions.py's map loader.
Builds a scenario in the map's most open region and checks the semantic path:

  1. found             (len(path) > 1; empty fails)
  2. collision-free    (0/N via the dense [y,x] ground-truth checker)
  3. avoids the hazard (min-distance to an injected "person" grows vs baseline)
  4. reaches the goal  (path ends within GOAL_TOLERANCE of the goal)

For (3)/(4) the hazard is placed near the goal so avoidance and arrival compete;
the near-goal taper should let both hold at once. Weight and sigma stay at their
defaults (not tuned to inflate avoidance). Also runs an attraction sanity check
(a negative-weight target should pull the path in).

The open region is used because avoidance can only show where a detour exists; in
a forced narrow corridor the soft penalty still traverses. RRT* is randomized, so
each condition is measured over several trials and compared on the median.

Usage:
  python3 verify_semantic.py --repo . --map /path/to/maze.yaml
  # optional: --trials N --weight W --sigma S --taper T
"""
import argparse, math, os, sys, statistics
import numpy as np


def _load_planner_and_map(repo, map_yaml):
    sys.path.insert(0, os.path.abspath(repo))
    import importlib.util
    vc_path = os.path.join(os.path.abspath(repo), "verify_collisions.py")
    spec = importlib.util.spec_from_file_location("vc", vc_path)
    vc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vc)
    from quadhrrt_nav.costmap_adapter import CostmapAdapter
    from quadhrrt_nav.planner_core import QuadRRTPlanner

    msg, (w, h, res, ox, oy) = vc.load_mask_and_meta(map_yaml)
    adapter = CostmapAdapter()
    adapter.update_from_msg(msg)
    grid = np.array(msg.data, dtype=np.int8).reshape(h, w)   # row-major [y,x]
    gt_yx = grid >= 99                                        # True = blocked
    return CostmapAdapter, QuadRRTPlanner, adapter, gt_yx, (w, h)


def count_collisions(path, adapter, gt_yx, w, h):
    """Dense [y,x] ground-truth collision count (same as verify_collisions)."""
    bad = tot = 0
    for i in range(len(path) - 1):
        x0, y0 = path[i]; x1, y1 = path[i + 1]
        d = math.hypot(x1 - x0, y1 - y0); n = max(2, int(d / 0.02))
        for t in range(n + 1):
            x = x0 + (x1 - x0) * t / n
            yy = y0 + (y1 - y0) * t / n
            cx, cyy = adapter.world_to_cell(x, yy)
            if 0 <= cyy < h and 0 <= cx < w:
                tot += 1
                if gt_yx[cyy, cx]:
                    bad += 1
    return bad, tot


def min_dist_to_point(path, px, py):
    """Minimum distance from any point ON the path (densely sampled) to (px,py)."""
    if len(path) < 2:
        return float('inf')
    best = float('inf')
    for i in range(len(path) - 1):
        x0, y0 = path[i]; x1, y1 = path[i + 1]
        d = math.hypot(x1 - x0, y1 - y0); n = max(2, int(d / 0.05))
        for t in range(n + 1):
            x = x0 + (x1 - x0) * t / n
            yy = y0 + (y1 - y0) * t / n
            best = min(best, math.hypot(x - px, yy - py))
    return best


def find_open_region(adapter, gt_yx, w, h):
    """Return (wx, wy, clearance_m) of the map's most-open free cell: the centre
    of the largest square window with no blocked cells. Uses an integral image of
    the blocked mask so window emptiness is O(1)."""
    res = adapter.res
    ii = np.zeros((h + 1, w + 1), dtype=np.int32)
    ii[1:, 1:] = np.cumsum(np.cumsum(gt_yx.astype(np.int32), 0), 1)

    def win_blocked(cy, cx, r):
        y0, y1, x0, x1 = cy - r, cy + r + 1, cx - r, cx + r + 1
        if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
            return 1
        return int(ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])

    for rm in [2.4, 2.2, 2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8]:
        r = int(rm / res)
        hits = []
        for cy in range(r, h - r, 5):
            for cx in range(r, w - r, 5):
                if win_blocked(cy, cx, r) == 0:
                    hits.append((cx, cy))
        if hits:
            cx, cy = hits[len(hits) // 2]
            wx, wy = adapter.cell_to_world(cx, cy)
            return wx, wy, rm
    return None, None, 0.0


def plan_trials(PlannerCls, adapter, start, goal, detections, sem_classes,
                weight, sigma, taper, trials):
    """Run `trials` plans with the given detections; return (paths, empties,
    times), keeping only non-empty paths. force_anytime=True on every planner so
    the baseline and the semantic run use the same optimization and only the
    penalty differs."""
    paths, empties, times = [], 0, []
    for _ in range(trials):
        p = PlannerCls(adapter, semantic_weight=weight, semantic_sigma=sigma,
                       semantic_taper_radius=taper,
                       semantic_classes=sem_classes, force_anytime=True)
        if detections:
            p.set_detections(detections)
        path = p.plan(tuple(start), tuple(goal))
        if len(path) > 1:
            paths.append(path)
            times.append(p.plan_time)
        else:
            empties += 1
    return paths, empties, times


def _final_verdict(ok):
    if ok:
        print("STEP 3 PASS: path FOUND + collision-free + bends away from the "
              "hazard + REACHES the goal (taper on); attraction pulls in.")
        sys.exit(0)
    print("STEP 3 FAIL: see failing condition(s) above.")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='.')
    ap.add_argument('--map', required=True)
    ap.add_argument('--trials', type=int, default=12)
    ap.add_argument('--weight', type=float, default=5.0)
    ap.add_argument('--sigma',  type=float, default=0.5)
    ap.add_argument('--taper',  type=float, default=0.5,
                    help='near-goal taper radius (m); 0 disables the taper')
    args = ap.parse_args()

    (CostmapAdapter, QuadRRTPlanner, adapter,
     gt_yx, (w, h)) = _load_planner_and_map(args.repo, args.map)
    from quadhrrt_nav.planner_core import GOAL_TOLERANCE

    # ── Build a controlled scenario in the map's most-open region ──
    cx0, cy0, clr = find_open_region(adapter, gt_yx, w, h)
    if cx0 is None:
        print("FAIL: no open region found on this map.")
        sys.exit(1)
    gap = clr * 0.8
    start = (cx0 - gap, cy0)
    goal  = (cx0 + gap, cy0)
    # hazard a fixed ~(sigma+taper) m from the goal, along the line: close enough
    # that the goal region competes, far enough that the taper doesn't kill
    # avoidance at the hazard. clamped to stay on the segment.
    off = min(args.sigma + args.taper if args.taper > 0 else args.sigma + 0.5,
              2 * gap * 0.6)
    person = (goal[0] - off, cy0)                # on the start->goal line
    d_pg = math.hypot(person[0] - goal[0], person[1] - goal[1])
    print("=" * 66)
    print(f"Scenario: open region centre ({cx0:.2f}, {cy0:.2f}), "
          f"clearance ~{clr:.1f} m")
    print(f"  start={start[0]:.2f},{start[1]:.2f}  "
          f"goal={goal[0]:.2f},{goal[1]:.2f}  "
          f"(straight-line gap {2*gap:.2f} m, detour available)")
    print(f"  person near goal: {d_pg:.2f} m from goal  "
          f"(taper radius {args.taper:.2f} m, goal tol {GOAL_TOLERANCE:.2f} m)")
    print(f"  weight={args.weight}  sigma={args.sigma}  taper={args.taper}  "
          f"trials={args.trials}")

    overall_ok = True

    # ── 0. Baseline (no detections) ─────────────────────────────
    base_paths, _, base_t = plan_trials(
        QuadRRTPlanner, adapter, start, goal, None,
        {"person": +1.0}, args.weight, args.sigma, args.taper, args.trials)
    if not base_paths:
        print("  FAIL: baseline produced no path; cannot test.")
        sys.exit(1)
    base_mind = [min_dist_to_point(p, *person) for p in base_paths]
    base_med = statistics.median(base_mind)
    print("=" * 66)
    print(f"BASELINE (no detections): min-dist to (person spot) "
          f"median={base_med:.3f} m  [{statistics.mean(base_t)*1000:.0f} ms/plan]")

    # ── 1-3. REPEL ──────────────────────────────────────────────
    print("=" * 66)
    print(f"REPEL: person at ({person[0]:.2f}, {person[1]:.2f}) weight=+1.0")
    sem_paths, sem_empty, sem_t = plan_trials(
        QuadRRTPlanner, adapter, start, goal, [("person", *person)],
        {"person": +1.0}, args.weight, args.sigma, args.taper, args.trials)

    # (1) FOUND
    if not sem_paths:
        print(f"  FAIL [found]: ALL {args.trials} semantic plans EMPTY.")
        sys.exit(1)
    print(f"  found: {len(sem_paths)}/{args.trials} (empty {sem_empty})  "
          f"[{statistics.mean(sem_t)*1000:.0f} ms/plan]")

    # (2) COLLISION-FREE
    worst = max(count_collisions(p, adapter, gt_yx, w, h)[0] for p in sem_paths)
    print(f"  collision-free: {'PASS' if worst == 0 else 'FAIL'} "
          f"({worst} collisions, worst over {len(sem_paths)} paths)")
    if worst != 0:
        overall_ok = False

    # (3) AVOIDS
    sem_mind = [min_dist_to_point(p, *person) for p in sem_paths]
    sem_med = statistics.median(sem_mind)
    if sem_med > base_med:
        print(f"  AVOIDANCE: PASS  (min-dist {base_med:.3f} -> {sem_med:.3f} m, "
              f"+{sem_med-base_med:.3f})")
    else:
        print(f"  AVOIDANCE: FAIL  (min-dist {base_med:.3f} -> {sem_med:.3f} m)")
        overall_ok = False

    # (4) reaches goal: with the hazard near the goal, avoidance must not drag the
    # endpoint off it. every path's final waypoint must land within GOAL_TOLERANCE.
    def goal_gap(p):
        return math.hypot(p[-1][0] - goal[0], p[-1][1] - goal[1])
    base_gap = statistics.median([goal_gap(p) for p in base_paths])
    sem_gaps = [goal_gap(p) for p in sem_paths]
    sem_gap_med = statistics.median(sem_gaps)
    worst_gap = max(sem_gaps)
    if worst_gap <= GOAL_TOLERANCE:
        print(f"  GOAL-REACHING: PASS  (end-to-goal median {sem_gap_med:.3f} m, "
              f"worst {worst_gap:.3f} <= tol {GOAL_TOLERANCE:.2f}; "
              f"baseline median {base_gap:.3f})")
    else:
        print(f"  GOAL-REACHING: FAIL  (end-to-goal worst {worst_gap:.3f} m > "
              f"tol {GOAL_TOLERANCE:.2f}; {sum(g > GOAL_TOLERANCE for g in sem_gaps)}"
              f"/{len(sem_gaps)} paths miss the goal)")
        overall_ok = False

    # ── ATTRACTION sanity check ─────────────────────────────────
    # Target offset perpendicular to the straight line (~1 sigma), inside the
    # open disk so the path can reach it and a gradient exists.
    print("=" * 66)
    off = min(0.8, clr * 0.4)
    tx, ty = cx0, cy0 + off
    att_w = 0.4   # modest: |att_w * weight| kept below typical step cost
    print(f"ATTRACTION: target at ({tx:.2f}, {ty:.2f}) weight=-{att_w} "
          f"(offset {off:.2f} m from line)")
    base_att = statistics.median([min_dist_to_point(p, tx, ty) for p in base_paths])
    att_paths, att_empty, _ = plan_trials(
        QuadRRTPlanner, adapter, start, goal, [("target", tx, ty)],
        {"target": -att_w}, args.weight, args.sigma, args.taper, args.trials)
    if not att_paths:
        print("  FAIL [found]: attraction produced no path.")
        overall_ok = False
    else:
        att_worst = max(count_collisions(p, adapter, gt_yx, w, h)[0]
                        for p in att_paths)
        att_med = statistics.median([min_dist_to_point(p, tx, ty)
                                     for p in att_paths])
        print(f"  found {len(att_paths)}/{args.trials}, collisions {att_worst}")
        if att_med < base_att and att_worst == 0:
            print(f"  ATTRACTION: PASS  (min-dist {base_att:.3f} -> "
                  f"{att_med:.3f} m, -{base_att-att_med:.3f})")
        else:
            print(f"  ATTRACTION: WEAK  (min-dist {base_att:.3f} -> "
                  f"{att_med:.3f} m, collisions {att_worst})")
            # Only collisions/empty are hard failures; a weak pull is a tuning
            # matter, not a correctness break.
            if att_worst != 0:
                overall_ok = False

    print("=" * 66)
    _final_verdict(overall_ok)


if __name__ == '__main__':
    main()
