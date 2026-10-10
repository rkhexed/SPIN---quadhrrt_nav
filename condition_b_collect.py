#!/usr/bin/env python3
"""
condition_b_collect.py - Condition B (our QuadHRRT* / RRT* planner) evaluation.

Planner-quality data for the A/B comparison table. For each matched start-goal
pair it plans `trials` times and reports:
  - path length and planning time as mean +/- std (RRT* is randomized, so the
    spread matters and bare means are not enough),
  - aggregate goal-reach rate and total collisions across the whole trial set.

Semantics are OFF here. This measures the base planner against the NavFn baseline
(Condition A), which must be run on the IDENTICAL goal set on the VM. The goal set
is shared via eval_goalsets.py so the two conditions cannot drift apart.

Runs WITHOUT ROS (numpy/PIL/yaml). Reuses verify_collisions.load_mask_and_meta as
the map loader (the OccupancyGrid-from-.pgm helper) and its dense [y,x] collision
check. Map is a command-line argument, so one flag switches maps rather than three
edited files.

Usage:
  python3 condition_b_collect.py --repo . --map /path/to/maze.yaml
  # optional: --trials N  --map-name maze  --csv out.csv  --force-anytime  --verbose

Every number is printed with its map, trial count, and goal set. Results also go
to a CSV (one row per pair plus an aggregate row) for mechanical table assembly.
"""
import argparse, contextlib, io, math, os, statistics, sys


def _sd(xs):
    return statistics.stdev(xs) if len(xs) >= 2 else 0.0


def _mean(xs):
    return statistics.mean(xs) if xs else 0.0


def path_length(path):
    return sum(math.hypot(path[i + 1][0] - path[i][0],
                          path[i + 1][1] - path[i][1])
               for i in range(len(path) - 1))


def load_env(repo, map_yaml):
    """Load the planner, the adapter, and a dense [y,x] ground-truth mask."""
    sys.path.insert(0, os.path.abspath(repo))
    import importlib.util
    import numpy as np
    vc_path = os.path.join(os.path.abspath(repo), "verify_collisions.py")
    spec = importlib.util.spec_from_file_location("vc", vc_path)
    vc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vc)
    from quadhrrt_nav.costmap_adapter import CostmapAdapter
    from quadhrrt_nav.planner_core import QuadRRTPlanner, GOAL_TOLERANCE

    msg, (w, h, res, ox, oy) = vc.load_mask_and_meta(map_yaml)
    adapter = CostmapAdapter()
    adapter.update_from_msg(msg)
    grid = np.array(msg.data, dtype=np.int8).reshape(h, w)   # row-major [y,x]
    gt_yx = grid >= 99                                        # True = blocked
    return (QuadRRTPlanner, adapter, gt_yx, (w, h, res, ox, oy),
            GOAL_TOLERANCE)


def count_collisions(path, adapter, gt_yx, w, h):
    """Dense [y,x] ground-truth collision count (same basis as verify_*)."""
    bad = 0
    for i in range(len(path) - 1):
        x0, y0 = path[i]; x1, y1 = path[i + 1]
        d = math.hypot(x1 - x0, y1 - y0); n = max(2, int(d / 0.02))
        for t in range(n + 1):
            x = x0 + (x1 - x0) * t / n
            yy = y0 + (y1 - y0) * t / n
            cx, cyy = adapter.world_to_cell(x, yy)
            if 0 <= cyy < h and 0 <= cx < w and gt_yx[cyy, cx]:
                bad += 1
    return bad


def run_pair(PlannerCls, adapter, gt_yx, wh, start, goal, trials,
             goal_tol, force_anytime, verbose):
    """Plan `trials` times for one pair. Returns a metrics dict.

    Length/time are aggregated over the trials that REACH the goal (the usable
    plans); found/reached rates and collisions are reported separately so a
    partial-success pair is not silently dropped."""
    w, h, *_ = wh
    lens, times = [], []
    n_found = n_reached = collisions = 0
    for _ in range(trials):
        if verbose:
            p = PlannerCls(adapter, force_anytime=force_anytime)
            path = p.plan(tuple(start), tuple(goal))
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                p = PlannerCls(adapter, force_anytime=force_anytime)
                path = p.plan(tuple(start), tuple(goal))
        if len(path) <= 1:
            continue
        n_found += 1
        collisions += count_collisions(path, adapter, gt_yx, w, h)
        end_gap = math.hypot(path[-1][0] - goal[0], path[-1][1] - goal[1])
        if end_gap <= goal_tol:
            n_reached += 1
            lens.append(path_length(path))
            times.append(p.plan_time * 1000.0)   # ms
    return {
        'n_trials': trials, 'n_found': n_found, 'n_reached': n_reached,
        'collisions': collisions,
        'len_mean': _mean(lens), 'len_std': _sd(lens),
        'ms_mean': _mean(times), 'ms_std': _sd(times),
        'lens': lens, 'times': times,
    }


CSV_COLS = ['map', 'map_name', 'trials', 'semantics', 'force_anytime',
            'start_x', 'start_y', 'goal_x', 'goal_y', 'navfn_confirmed',
            'n_trials', 'n_found', 'n_reached', 'found_rate', 'reach_rate',
            'collisions', 'path_len_mean', 'path_len_std',
            'plan_ms_mean', 'plan_ms_std']


def csv_row(map_name, map_base, trials, force_anytime, start, goal,
            navfn_confirmed, m):
    rate = lambda a, b: (a / b) if b else 0.0
    sx, sy = (start if start else ('', ''))
    gx, gy = (goal if goal else ('', ''))
    return [map_base, map_name, trials, 'off', int(force_anytime),
            sx, sy, gx, gy, ('' if navfn_confirmed is None else int(navfn_confirmed)),
            m['n_trials'], m['n_found'], m['n_reached'],
            f"{rate(m['n_found'], m['n_trials']):.3f}",
            f"{rate(m['n_reached'], m['n_trials']):.3f}",
            m['collisions'],
            f"{m['len_mean']:.4f}", f"{m['len_std']:.4f}",
            f"{m['ms_mean']:.2f}", f"{m['ms_std']:.2f}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='.')
    ap.add_argument('--map', required=True, help='path to the map .yaml')
    ap.add_argument('--map-name', default=None,
                    help='goal-set key; defaults to the .yaml basename')
    ap.add_argument('--trials', type=int, default=10)
    ap.add_argument('--force-anytime', action='store_true',
                    help='run the planner with anytime optimization forced on')
    ap.add_argument('--csv', default=None,
                    help='write results CSV here (default condition_b_<map>.csv '
                         'next to this script)')
    ap.add_argument('--verbose', action='store_true',
                    help='let the planner print its per-plan chatter')
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.repo))
    from eval_goalsets import get_goalset, pairs_needing_condition_a, fmt_pair

    map_base = os.path.splitext(os.path.basename(args.map))[0]
    map_name = args.map_name or map_base
    goalset = get_goalset(map_name)

    (PlannerCls, adapter, gt_yx, meta, goal_tol) = load_env(args.repo, args.map)
    w, h, res, ox, oy = meta

    print("=" * 70)
    print("CONDITION B - QuadHRRT* / RRT* planner (this repo)")
    print(f"  map          : {map_base}  ({w}x{h} cells, res {res} m, "
          f"origin ({ox:g},{oy:g}))")
    print(f"  trials/pair  : {args.trials}   (RRT* randomized; mean +/- std)")
    print(f"  semantics    : OFF   force_anytime={args.force_anytime}")
    print(f"  goal tol     : {goal_tol} m   hazard: none (planner comparison)")
    print(f"  goal set     : {len(goalset)} matched pairs on '{map_name}'")
    for p in goalset:
        print("      " + fmt_pair(p))
    print("=" * 70)

    hdr = (f"{'start':>12} {'goal':>12}  {'found':>6} {'reach':>6} "
           f"{'coll':>4}  {'len(m) mean+/-std':>20}  {'time(ms) mean+/-std':>20}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    all_lens, all_times = [], []
    tot_trials = tot_found = tot_reached = tot_coll = 0
    for p in goalset:
        start, goal = p['start'], p['goal']
        m = run_pair(PlannerCls, adapter, gt_yx, meta, start, goal,
                     args.trials, goal_tol, args.force_anytime, args.verbose)
        rows.append(csv_row(map_name, map_base, args.trials, args.force_anytime,
                            start, goal, p['navfn_confirmed'], m))
        all_lens += m['lens']; all_times += m['times']
        tot_trials += m['n_trials']; tot_found += m['n_found']
        tot_reached += m['n_reached']; tot_coll += m['collisions']
        flag = '' if p['navfn_confirmed'] else ' *'
        print(f"({start[0]:5g},{start[1]:5g}) ({goal[0]:5g},{goal[1]:5g})  "
              f"{m['n_found']:>2}/{m['n_trials']:<3} {m['n_reached']:>2}/{m['n_trials']:<3} "
              f"{m['collisions']:>4}  "
              f"{m['len_mean']:8.3f} +/- {m['len_std']:5.3f}  "
              f"{m['ms_mean']:8.2f} +/- {m['ms_std']:6.2f}{flag}")

    agg = {
        'n_trials': tot_trials, 'n_found': tot_found, 'n_reached': tot_reached,
        'collisions': tot_coll,
        'len_mean': _mean(all_lens), 'len_std': _sd(all_lens),
        'ms_mean': _mean(all_times), 'ms_std': _sd(all_times),
    }
    print("-" * len(hdr))
    print(f"{'AGGREGATE':>25}  {tot_found:>2}/{tot_trials:<3} "
          f"{tot_reached:>2}/{tot_trials:<3} {tot_coll:>4}  "
          f"{agg['len_mean']:8.3f} +/- {agg['len_std']:5.3f}  "
          f"{agg['ms_mean']:8.2f} +/- {agg['ms_std']:6.2f}")
    print(f"  goal-reach rate: {tot_reached}/{tot_trials} "
          f"({100.0*tot_reached/tot_trials:.1f}%)   collisions: {tot_coll}")
    rows.append(csv_row(map_name, map_base, args.trials, args.force_anytime,
                        None, None, None, agg))

    need = pairs_needing_condition_a(map_name)
    if need:
        print("=" * 70)
        print(f"Condition A (NavFn, VM) still needed for {len(need)} pair(s) "
              f"on '{map_name}' (marked * above):")
        for p in need:
            print("      " + fmt_pair(p))
        print("Run condition_a_collect.py on the VM for the SAME goal set, then "
              "drop any NavFn rejects from both conditions before reporting.")

    csv_path = args.csv or os.path.join(
        os.path.abspath(args.repo), f"condition_b_{map_base}.csv")
    with open(csv_path, 'w') as f:
        f.write(','.join(CSV_COLS) + '\n')
        for r in rows:
            f.write(','.join(str(x) for x in r) + '\n')
    print("=" * 70)
    print(f"CSV written: {csv_path}")


if __name__ == '__main__':
    main()
