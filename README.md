# quadhrrt_nav

A ROS2 (`ament_python`) package with a QuadHRRT* sampling-based global planner that
plans over the live Nav2 costmap. Two additions on top of the base planner:

1. Semantic cost injection: detections (e.g. a `person`) bend the planned path
   around hazards (repel) or toward targets (attract), as a soft penalty on the
   RRT* edge cost. A near-goal taper stops a hazard next to the goal from pushing
   the path off the goal.
2. An LLM goal front-end (`das_goal_node`): turns a natural-language instruction
   into a map-frame goal pose.

Pipeline:

```
instruction (text)
  -> das_goal_node (one LLM call) -> /goal_pose
  -> quadhrrt_node (plan over the costmap) -> FollowPath -> Nav2 controller
```

`quadhrrt_node` is a standalone planning node, not a Nav2 global-planner plugin.

## Python version

Developed and tested on Python 3.12.3. The live nodes run inside a sourced
ROS2 + Nav2 workspace. Python 3.12 matches ROS2 Jazzy (Ubuntu 24.04); use whatever
distro provides your Nav2 / turtlebot4 stack.

## What needs ROS2/Nav2 and what does not

The package is split so the planner and its verification run on a plain laptop,
and only the live nodes need the full stack.

| Component | Command | Needs ROS2/Nav2? | Extra needs |
|---|---|---|---|
| Planner core | `import quadhrrt_nav.planner_core` | No (pure Python) | `numpy` |
| Collision verifier | `python3 verify_collisions.py --map ...` | No | `numpy pillow pyyaml` + a map `.pgm/.yaml` |
| Semantic verifier | `python3 verify_semantic.py --map ...` | No | `numpy pillow pyyaml` + a map `.pgm/.yaml` |
| Das parse self-test | `python3 quadhrrt_nav/das_goal_node.py --selftest` | No | stdlib only |
| Planner node | `ros2 run quadhrrt_nav quadhrrt_node` | Yes | a running Nav2 (costmap, TF, FollowPath) |
| Das node (live) | `ros2 run quadhrrt_nav das_goal_node` | Yes | `requests` + network + `OPENROUTER_API_KEY` |

Installable with pip and works outside ROS2/Nav2 (`pip install -r requirements.txt`):
the planner math, both offline verifiers, and the Das parse self-test.

Does not work without ROS2/Nav2: `quadhrrt_node` (it subscribes to the costmap and
TF and drives the robot via the `FollowPath` action) and the live Das node (it
needs `rclpy`/`std_msgs` to subscribe/publish, plus network and an API key for the
LLM call). The pure-Python parsing inside the Das node is covered by `--selftest`;
the network call and the ROS publish are not.

## Installation

### 1. Pure-Python dependencies (pip, no ROS)

```bash
pip install -r requirements.txt
```

Covers the planner and the offline verifiers.

### 2. ROS2 / Nav2 dependencies (for the live nodes, not pip)

Declared in `package.xml` (`rclpy`, `nav_msgs`, `geometry_msgs`, `nav2_msgs`,
`std_msgs`, `tf2_ros`) and installed from ROS, not pip:

```bash
cd <your_ros2_ws>            # this package lives under src/
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select quadhrrt_nav
source install/setup.bash
```

## Running

### Planner node (needs a running Nav2)

```bash
ros2 run quadhrrt_nav quadhrrt_node
```

Publish a goal (or let the Das node do it) and the planner plans over the costmap
and hands the path to `FollowPath`.

### Das LLM goal front-end (needs ROS2 + network + API key)

The API key is read from an environment variable at runtime and is not stored in
the code. Export it before launching:

```bash
export OPENROUTER_API_KEY=...        # required; do not commit this
ros2 run quadhrrt_nav das_goal_node
# then, from another terminal:
ros2 topic pub --once /instruction std_msgs/msg/String "{data: 'bring me a bottle of water'}"
```

The node makes one LLM call per instruction, resolves it to a goal, and publishes
a `PoseStamped` on `/goal_pose`. An unparseable or malformed result is logged and
dropped, never published.

### Offline verification (no ROS)

```bash
python3 verify_collisions.py --repo . --map /path/to/maze.yaml   # raw path is collision-free
python3 verify_semantic.py   --repo . --map /path/to/maze.yaml   # found + collision-free + avoids + reaches goal
python3 quadhrrt_nav/das_goal_node.py --selftest                 # instruction/LLM-output -> goal xy
```

## Key parameters (`quadhrrt_node`)

| Parameter | Default | Meaning |
|---|---|---|
| `plan_time_budget` | `1.0` | wall-clock seconds per plan (bounds latency) |
| `semantic_weight` | `5.0` | global scale of the semantic penalty |
| `semantic_sigma` | `0.5` | metres; Gaussian falloff width of a detection |
| `semantic_taper_radius` | `0.5` | metres; penalty ramps to 0 within this distance of the goal |
| `force_anytime` | `false` | force anytime optimization on even with no detections |
| `detections` | `['']` | static test detections, `"class:x:y"` world coords, e.g. `['person:1.0:2.0']` |
| `metrics_csv` | `''` | if set, append one CSV row per plan |

## Notes

The LLM API key is read only from `OPENROUTER_API_KEY`; do not hardcode or commit
keys. `.gitignore` excludes Python/ROS build artifacts.
