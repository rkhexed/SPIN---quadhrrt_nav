#!/usr/bin/env python3
"""
condition_a_collect.py - Condition A (Nav2 NavFn baseline) evaluation.

The A side of the A/B planner comparison. For each matched start-goal pair it asks
the live Nav2 planner server for a NavFn path `trials` times and reports:
  - path length and planning time as mean +/- std (path length is deterministic
    for NavFn, so its std is ~0; planning time varies),
  - aggregate goal-reach rate and collision count across the trial set.

The goal set is imported from eval_goalsets.py, the SAME source condition_b_collect
uses, so Conditions A and B are on an identical set by construction.

WHERE THIS RUNS: the VirtualBox VM only, with the full live Nav2 stack up (the
5-terminal bringup: map_server, amcl/localization, planner_server + controller,
bt_navigator, rviz or the nav2 lifecycle). It talks to the running
planner_server over the ComputePathToPose action. It CANNOT run on WSL (no ROS).
On WSL use --selftest to print the goal set and sanity-check the shared module.

NavFn rejects goals whose cell is near a wall or in unknown space. Any pair NavFn
rejects here must be dropped from BOTH conditions before the comparison table.
The pairs still needing this confirmation are listed in the header at run time.

Usage on the VM (Nav2 already running):
  ros2 run ... no - this is a plain script, run it with python3 in a sourced ws:
  python3 condition_a_collect.py --map-name maze --trials 10
  # optional: --planner-id GridBased  --frame map  --csv out.csv
  #           --goal-tolerance 0.5     --timeout 15.0

Usage on WSL (no ROS):
  python3 condition_a_collect.py --selftest --map-name maze
"""
import argparse, math, os, sys, time

# Match planner_core.GOAL_TOLERANCE. Kept as a literal so this script has no hard
# dependency on numpy/the planner package when it runs beside Nav2 on the VM.
DEFAULT_GOAL_TOLERANCE = 0.5

# Column order is IDENTICAL to condition_b_collect.CSV_COLS so both CSVs stack
# into one table. 'semantics' is n/a for NavFn; 'force_anytime' is left blank.
CSV_COLS = ['map', 'map_name', 'trials', 'semantics', 'force_anytime',
            'start_x', 'start_y', 'goal_x', 'goal_y', 'navfn_confirmed',
            'n_trials', 'n_found', 'n_reached', 'found_rate', 'reach_rate',
            'collisions', 'path_len_mean', 'path_len_std',
            'plan_ms_mean', 'plan_ms_std']

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionClient
    from nav2_msgs.action import ComputePathToPose
    from geometry_msgs.msg import PoseStamped
    _HAVE_ROS = True
except ImportError:
    _HAVE_ROS = False


def _sd(xs):
    import statistics
    return statistics.stdev(xs) if len(xs) >= 2 else 0.0


def _mean(xs):
    import statistics
    return statistics.mean(xs) if xs else 0.0


def path_length(poses):
    """Length of a nav_msgs/Path pose list, in metres."""
    d = 0.0
    for i in range(len(poses) - 1):
        a = poses[i].pose.position
        b = poses[i + 1].pose.position
        d += math.hypot(b.x - a.x, b.y - a.y)
    return d


if _HAVE_ROS:
    class NavFnCollector(Node):
        def __init__(self, frame, planner_id, timeout):
            super().__init__('condition_a_collect')
            self.frame = frame
            self.planner_id = planner_id
            self.timeout = timeout
            self._client = ActionClient(self, ComputePathToPose,
                                        'compute_path_to_pose')

        def _pose(self, x, y):
            p = PoseStamped()
            p.header.frame_id = self.frame
            p.header.stamp = self.get_clock().now().to_msg()
            p.pose.position.x = float(x)
            p.pose.position.y = float(y)
            p.pose.orientation.w = 1.0
            return p

        def plan_once(self, start, goal):
            """One ComputePathToPose call. Returns (poses or None, elapsed_s).
            poses is None on rejection/timeout/empty path."""
            g = ComputePathToPose.Goal()
            g.start = self._pose(*start)
            g.goal = self._pose(*goal)
            g.use_start = True
            g.planner_id = self.planner_id

            t0 = time.perf_counter()
            send = self._client.send_goal_async(g)
            rclpy.spin_until_future_complete(self, send, timeout_sec=self.timeout)
            gh = send.result()
            if gh is None or not gh.accepted:
                return None, time.perf_counter() - t0
            res = gh.get_result_async()
            rclpy.spin_until_future_complete(self, res, timeout_sec=self.timeout)
            outcome = res.result()
            elapsed = time.perf_counter() - t0
            if outcome is None:
                return None, elapsed
            poses = outcome.result.path.poses
            return (poses if len(poses) > 1 else None), elapsed

        def wait_for_server(self, timeout=10.0):
            return self._client.wait_for_server(timeout_sec=timeout)


def run_pair(node, start, goal, trials, goal_tol):
    """Plan `trials` times for one pair. Length/time over reached trials; found/
    reached rates reported separately. NavFn paths are collision-free on the
    inflated costmap by construction, so collisions is 0 (kept for column parity
    with Condition B)."""
    lens, times = [], []
    n_found = n_reached = 0
    for _ in range(trials):
        poses, elapsed = node.plan_once(start, goal)
        if poses is None:
            continue
        n_found += 1
        end = poses[-1].pose.position
        if math.hypot(end.x - goal[0], end.y - goal[1]) <= goal_tol:
            n_reached += 1
            lens.append(path_length(poses))
            times.append(elapsed * 1000.0)   # ms
    return {
        'n_trials': trials, 'n_found': n_found, 'n_reached': n_reached,
        'collisions': 0,
        'len_mean': _mean(lens), 'len_std': _sd(lens),
        'ms_mean': _mean(times), 'ms_std': _sd(times),
        'lens': lens, 'times': times,
    }


def csv_row(map_name, trials, start, goal, navfn_confirmed, m):
    rate = lambda a, b: (a / b) if b else 0.0
    sx, sy = (start if start else ('', ''))
    gx, gy = (goal if goal else ('', ''))
    return [map_name, map_name, trials, 'n/a', '',
            sx, sy, gx, gy,
            ('' if navfn_confirmed is None else int(navfn_confirmed)),
            m['n_trials'], m['n_found'], m['n_reached'],
            f"{rate(m['n_found'], m['n_trials']):.3f}",
            f"{rate(m['n_reached'], m['n_trials']):.3f}",
            m['collisions'],
            f"{m['len_mean']:.4f}", f"{m['len_std']:.4f}",
            f"{m['ms_mean']:.2f}", f"{m['ms_std']:.2f}"]


def _load_goalset(repo, map_name):
    sys.path.insert(0, os.path.abspath(repo))
    from eval_goalsets import get_goalset, pairs_needing_condition_a, fmt_pair
    return get_goalset(map_name), pairs_needing_condition_a, fmt_pair


def selftest(repo, map_name):
    goalset, need_fn, fmt = _load_goalset(repo, map_name)
    print(f"goal set '{map_name}': {len(goalset)} pairs")
    for p in goalset:
        print("  " + fmt(p))
    need = need_fn(map_name)
    print(f"{len(need)} pair(s) need Condition A (NavFn) data on the VM.")
    print("selftest OK (shared goal set imports; ROS not required for this check)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='.')
    ap.add_argument('--map-name', default='maze',
                    help='goal-set key: maze | depot | warehouse')
    ap.add_argument('--trials', type=int, default=10)
    ap.add_argument('--planner-id', default='GridBased',
                    help="Nav2 planner plugin id (NavFn default: GridBased)")
    ap.add_argument('--frame', default='map')
    ap.add_argument('--goal-tolerance', type=float, default=DEFAULT_GOAL_TOLERANCE)
    ap.add_argument('--timeout', type=float, default=15.0,
                    help='seconds to wait per action call')
    ap.add_argument('--csv', default=None)
    ap.add_argument('--selftest', action='store_true',
                    help='no ROS: print the goal set and exit')
    args = ap.parse_args()

    if args.selftest:
        selftest(args.repo, args.map_name)
        return

    if not _HAVE_ROS:
        print("ERROR: ROS 2 (rclpy / nav2_msgs) not importable. This collector "
              "runs on the VM with the live Nav2 stack. Use --selftest on WSL.",
              file=sys.stderr)
        sys.exit(2)

    goalset, need_fn, fmt = _load_goalset(args.repo, args.map_name)

    print("=" * 70)
    print("CONDITION A - Nav2 NavFn baseline (live planner_server)")
    print(f"  map / goal set : {args.map_name}   ({len(goalset)} matched pairs)")
    print(f"  planner_id     : {args.planner_id}   frame: {args.frame}")
    print(f"  trials/pair    : {args.trials}   goal tol: {args.goal_tolerance} m")
    print(f"  note           : NavFn path length is deterministic (std ~0); "
          f"time varies")
    for p in goalset:
        print("      " + fmt(p))
    print("=" * 70)

    rclpy.init()
    node = NavFnCollector(args.frame, args.planner_id, args.timeout)
    if not node.wait_for_server(timeout=10.0):
        print("ERROR: compute_path_to_pose action server not available. Is the "
              "Nav2 planner_server up and its lifecycle active?", file=sys.stderr)
        node.destroy_node(); rclpy.shutdown(); sys.exit(3)

    hdr = (f"{'start':>12} {'goal':>12}  {'found':>6} {'reach':>6} "
           f"{'coll':>4}  {'len(m) mean+/-std':>20}  {'time(ms) mean+/-std':>20}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    all_lens, all_times = [], []
    tot_trials = tot_found = tot_reached = tot_coll = 0
    rejected = []
    for p in goalset:
        start, goal = p['start'], p['goal']
        m = run_pair(node, start, goal, args.trials, args.goal_tolerance)
        rows.append(csv_row(args.map_name, args.trials, start, goal,
                            p['navfn_confirmed'], m))
        all_lens += m['lens']; all_times += m['times']
        tot_trials += m['n_trials']; tot_found += m['n_found']
        tot_reached += m['n_reached']; tot_coll += m['collisions']
        if m['n_found'] == 0:
            rejected.append(p)
        print(f"({start[0]:5g},{start[1]:5g}) ({goal[0]:5g},{goal[1]:5g})  "
              f"{m['n_found']:>2}/{m['n_trials']:<3} {m['n_reached']:>2}/{m['n_trials']:<3} "
              f"{m['collisions']:>4}  "
              f"{m['len_mean']:8.3f} +/- {m['len_std']:5.3f}  "
              f"{m['ms_mean']:8.2f} +/- {m['ms_std']:6.2f}")

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
          f"({100.0*tot_reached/max(1,tot_trials):.1f}%)")
    rows.append(csv_row(args.map_name, args.trials, None, None, None, agg))

    if rejected:
        print("=" * 70)
        print(f"NavFn REJECTED {len(rejected)} pair(s) (no path in any trial). "
              f"Drop these from BOTH conditions before the comparison table:")
        for p in rejected:
            print("      " + fmt(p))

    node.destroy_node()
    rclpy.shutdown()

    csv_path = args.csv or os.path.join(
        os.path.abspath(args.repo), f"condition_a_{args.map_name}.csv")
    with open(csv_path, 'w') as f:
        f.write(','.join(CSV_COLS) + '\n')
        for r in rows:
            f.write(','.join(str(x) for x in r) + '\n')
    print("=" * 70)
    print(f"CSV written: {csv_path}")


if __name__ == '__main__':
    main()
