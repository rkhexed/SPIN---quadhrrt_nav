#!/usr/bin/env python3
# coding=utf-8
"""
das_goal_node.py

LLM goal front-end. Takes a natural-language instruction on /instruction, asks an
LLM (OpenRouter) for a goal, and publishes it on /goal_pose for the planner node.

The LLM is called once per instruction inside the subscription callback - not in a
loop and not per planning cycle. A bad/empty result is logged and dropped.

The API key is read from OPENROUTER_API_KEY at runtime (never hardcoded). Export it
before launching:
    export OPENROUTER_API_KEY=...

Run `python3 das_goal_node.py --selftest` to test the parsing without ROS/network.
"""
import json
import math
import os
import re

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# item -> location-name hints
DEFAULT_HINTS = {
    "water": "table",
    "bottle of water": "table",
    "panadol": "foldable chair",
    "pain relief": "foldable chair",
    "paracetamol": "foldable chair",
}


def build_system_prompt(locations, hints):
    p = "You are a robot navigation assistant in a warehouse environment.\n\n"
    if locations:
        p += "Here are the known locations:\n"
        for name, c in locations.items():
            p += f"- {name}: ({c[0]}, {c[1]})\n"
    if hints:
        p += "\nHere are item-to-location hints:\n"
        for item, loc in hints.items():
            p += f"- {item} -> {loc}\n"
    p += '\nRespond only in JSON like: {"goal": [x, y]}'
    return p


def parse_goal(message, hints, locations):
    """LLM message -> goal [x, y], or None. Tries direct JSON, then JSON embedded
    in text, then a bare item name via the hints."""
    if not message:
        return None
    try:
        parsed = json.loads(message)
        if isinstance(parsed, dict) and "goal" in parsed:
            return parsed["goal"]
    except (json.JSONDecodeError, TypeError):
        pass
    m = re.search(r"\{.*\}", message, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, dict) and "goal" in parsed:
                return parsed["goal"]
        except json.JSONDecodeError:
            pass
    key = message.strip().lower()
    if key in hints and hints[key] in locations:
        return locations[hints[key]]
    return None


def goal_to_xy(goal):
    """Validate a goal into finite (x, y) floats, or None."""
    try:
        x, y = float(goal[0]), float(goal[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


# ROS node - only defined when rclpy is importable, so the parsing above stays
# usable (and testable) without ROS.
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from geometry_msgs.msg import PoseStamped
    _HAVE_ROS = True
except ImportError:
    _HAVE_ROS = False


if _HAVE_ROS:
    import requests

    class DasGoalNode(Node):
        def __init__(self):
            super().__init__('das_goal_node')

            self.declare_parameter('instruction_topic', '/instruction')
            self.declare_parameter('goal_topic', '/goal_pose')
            self.declare_parameter('global_frame', 'map')
            self.declare_parameter('model', 'openai/gpt-4o')
            self.declare_parameter('locations_file', '')
            self.declare_parameter('request_timeout', 20.0)

            self.goal_topic = self.get_parameter('goal_topic').value
            self.global_frame = self.get_parameter('global_frame').value
            self.model = self.get_parameter('model').value
            self.request_timeout = float(self.get_parameter('request_timeout').value)

            self.api_key = os.environ.get('OPENROUTER_API_KEY')
            if not self.api_key:
                self.get_logger().error(
                    "OPENROUTER_API_KEY not set - export it before launching. "
                    "Instructions will be dropped until it is set.")

            self.locations = self._load_locations(
                self.get_parameter('locations_file').value)
            self.hints = DEFAULT_HINTS
            self.system_prompt = build_system_prompt(self.locations, self.hints)

            self.goal_pub = self.create_publisher(
                PoseStamped, self.goal_topic, 10)
            instr_topic = self.get_parameter('instruction_topic').value
            self.create_subscription(String, instr_topic, self.on_instruction, 10)

            self.get_logger().info(
                f"das_goal_node up. instruction='{instr_topic}' -> LLM "
                f"({self.model}) -> goal='{self.goal_topic}' "
                f"(frame '{self.global_frame}'). "
                f"{len(self.locations)} known location(s).")

        def _load_locations(self, path):
            # explicit param path wins; else try the llm_nav_interface bundled
            # file; else empty (the LLM can still return raw coordinates)
            if not path:
                try:
                    from ament_index_python.packages import \
                        get_package_share_directory
                    path = os.path.join(
                        get_package_share_directory('llm_nav_interface'),
                        'warehouse_locations.json')
                except Exception:
                    return {}
            try:
                with open(os.path.expanduser(path)) as f:
                    return json.load(f)
            except Exception as e:
                self.get_logger().warn(f"Could not load locations '{path}': {e}")
                return {}

        def on_instruction(self, msg):
            text = (msg.data or "").strip()
            if not text:
                return
            self.get_logger().info(f"Instruction: {text!r}")

            goal = self._resolve_goal(text)     # one LLM call
            if goal is None:
                self.get_logger().warn("No usable goal from LLM - dropping.")
                return
            xy = goal_to_xy(goal)
            if xy is None:
                self.get_logger().warn(f"Malformed goal {goal!r} - dropping.")
                return

            pose = PoseStamped()
            pose.header.frame_id = self.global_frame
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x, pose.pose.position.y = xy
            pose.pose.orientation.w = 1.0
            self.goal_pub.publish(pose)
            self.get_logger().info(
                f"Published goal ({xy[0]:.2f}, {xy[1]:.2f}) to {self.goal_topic}.")

        def _resolve_goal(self, command):
            if not self.api_key:
                self.get_logger().error("No OPENROUTER_API_KEY - cannot resolve.")
                return None
            headers = {"Authorization": f"Bearer {self.api_key}",
                       "Content-Type": "application/json"}
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": command},
                ],
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
            }
            try:
                r = requests.post(OPENROUTER_URL, headers=headers, json=payload,
                                  timeout=self.request_timeout)
            except requests.RequestException as e:
                self.get_logger().error(f"LLM request failed: {e}")
                return None
            if r.status_code != 200:
                self.get_logger().error(f"LLM API {r.status_code}: {r.text[:200]}")
                return None
            try:
                message = r.json()["choices"][0]["message"]["content"]
            except (KeyError, IndexError, ValueError) as e:
                self.get_logger().error(f"Unexpected LLM response: {e}")
                return None
            goal = parse_goal(message, self.hints, self.locations)
            if goal is None:
                self.get_logger().error(f"No goal parsed from LLM: {message!r}")
            return goal

    def main(args=None):
        rclpy.init(args=args)
        node = DasGoalNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
else:
    def main(args=None):
        raise SystemExit("rclpy not available. Use --selftest to test parsing.")


def _selftest():
    locs = {"table": [1.0, 2.0], "foldable chair": [-3.5, 0.25]}
    hints = DEFAULT_HINTS
    assert parse_goal('{"goal": [1.5, -2.0]}', hints, locs) == [1.5, -2.0]
    assert parse_goal('Sure! {"goal": [0, 0]} ok', hints, locs) == [0, 0]
    assert parse_goal('water', hints, locs) == [1.0, 2.0]
    assert parse_goal('PARACETAMOL', hints, locs) == [-3.5, 0.25]
    assert parse_goal('no idea', hints, locs) is None
    assert parse_goal('', hints, locs) is None
    assert goal_to_xy([1, 2]) == (1.0, 2.0)
    assert goal_to_xy(["3", "4"]) == (3.0, 4.0)
    assert goal_to_xy([1]) is None
    assert goal_to_xy("nope") is None
    assert goal_to_xy([float('nan'), 0]) is None
    print("selftest OK")


if __name__ == '__main__':
    import sys
    if '--selftest' in sys.argv:
        _selftest()
    else:
        main()
