#!/usr/bin/env python3
"""
eval_goalsets.py - the matched start/goal pairs used by BOTH condition scripts.

condition_a_collect.py (Nav2 NavFn baseline, VM) and condition_b_collect.py (our
RRT* planner, WSL) import the SAME pairs from here so the A/B comparison is on an
identical set. Do not hardcode goals in either script.

Each pair carries `navfn_confirmed`:
  True  - the pair is known to be solvable by BOTH planners (safe to report).
  False - RRT* solves it reliably, but NavFn has NOT been confirmed on it yet.
          Run Condition A on the VM and drop any NavFn rejects from both sides
          before putting the pair in the paper.

RRT* reliability of every pair below was measured on the real maps (median over
6 trials, goal-reach within GOAL_TOLERANCE): all reach >= 5/6. maze pairs 1-4 are
the ones previously confirmed on NavFn too; the rest await Condition A.

Coordinates are world (map) frame metres. Map spans, for reference:
  maze:      x[-10.1, 10.0]  y[-10.0, 10.1]   (res 0.03)
  depot:     x[-15.1, 15.1]  y[ -7.7,  7.5]   (res 0.04)
  warehouse: x[-15.1, 15.1]  y[-25.0, 25.2]   (res 0.03)
"""

# One dict per pair: start (sx,sy), goal (gx,gy), navfn_confirmed.
GOAL_SETS = {
    'maze': [
        {'start': (-8.0, -8.0), 'goal': (5.0, 5.0), 'navfn_confirmed': True},
        {'start': (-8.0, -8.0), 'goal': (8.0, 8.0), 'navfn_confirmed': True},
        {'start': (-8.0, -8.0), 'goal': (0.0, 0.0), 'navfn_confirmed': True},
        {'start': (0.0, 0.0),   'goal': (8.0, 8.0), 'navfn_confirmed': True},
        {'start': (0.0, 0.0),   'goal': (-8.0, -8.0), 'navfn_confirmed': False},
        {'start': (8.0, 8.0),   'goal': (0.0, 0.0), 'navfn_confirmed': False},
        {'start': (5.0, 5.0),   'goal': (8.0, 8.0), 'navfn_confirmed': False},
        {'start': (-8.0, 0.0),  'goal': (8.0, 0.0), 'navfn_confirmed': False},
    ],
    'depot': [
        {'start': (-3.0, -3.0), 'goal': (3.0, 3.0),  'navfn_confirmed': False},
        {'start': (-4.0, 0.0),  'goal': (4.0, 0.0),  'navfn_confirmed': False},
        {'start': (-4.0, -4.0), 'goal': (4.0, 4.0),  'navfn_confirmed': False},
        {'start': (3.0, 3.0),   'goal': (-3.0, -3.0),'navfn_confirmed': False},
        {'start': (0.0, 0.0),   'goal': (4.0, 4.0),  'navfn_confirmed': False},
        {'start': (-4.0, 4.0),  'goal': (4.0, -4.0), 'navfn_confirmed': False},
        {'start': (-2.0, -2.0), 'goal': (2.0, 2.0),  'navfn_confirmed': False},
    ],
    'warehouse': [
        {'start': (-3.0, -3.0), 'goal': (3.0, 3.0),  'navfn_confirmed': False},
        {'start': (-5.0, 0.0),  'goal': (5.0, 0.0),  'navfn_confirmed': False},
        {'start': (0.0, -4.0),  'goal': (0.0, 4.0),  'navfn_confirmed': False},
        {'start': (-5.0, -5.0), 'goal': (5.0, 5.0),  'navfn_confirmed': False},
        {'start': (5.0, 5.0),   'goal': (-5.0, -5.0),'navfn_confirmed': False},
        {'start': (0.0, 0.0),   'goal': (5.0, 5.0),  'navfn_confirmed': False},
        {'start': (-2.0, -2.0), 'goal': (2.0, 2.0),  'navfn_confirmed': False},
    ],
}


def get_goalset(map_name):
    """Return the list of pair dicts for a map. KeyError with the valid names
    if the map is unknown."""
    if map_name not in GOAL_SETS:
        raise KeyError(f"no goal set for '{map_name}'; "
                       f"known maps: {sorted(GOAL_SETS)}")
    return GOAL_SETS[map_name]


def pairs_needing_condition_a(map_name):
    """Pairs whose NavFn side is not yet confirmed. These are the goals the VM
    Condition A run must cover, and any NavFn reject among them must be dropped
    from both conditions before reporting."""
    return [p for p in get_goalset(map_name) if not p['navfn_confirmed']]


def fmt_pair(p):
    s, g = p['start'], p['goal']
    tag = '' if p['navfn_confirmed'] else '  [needs Condition A]'
    return f"({s[0]:g},{s[1]:g}) -> ({g[0]:g},{g[1]:g}){tag}"


if __name__ == '__main__':
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else 'maze'
    print(f"goal set '{name}' ({len(get_goalset(name))} pairs):")
    for p in get_goalset(name):
        print("  " + fmt_pair(p))
    need = pairs_needing_condition_a(name)
    print(f"{len(need)} pair(s) still need Condition A (NavFn) data.")
