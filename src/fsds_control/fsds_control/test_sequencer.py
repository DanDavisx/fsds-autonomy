#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from fs_msgs.msg import ControlCommand  

import yaml


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class Step:
    name: str
    duration_s: float
    throttle: float
    steering: float
    brake: float


class TestSequencer(Node):
    """
    This runs a series of commands from test_plan.yaml for testing purposes.
    """

    def __init__(self):
        super().__init__("test_sequencer")

        # Params
        self.declare_parameter("plan_path", "")
        plan_path = str(self.get_parameter("plan_path").value).strip()
        if not plan_path:
            raise RuntimeError("Missing required parameter: plan_path")

        # Load YAML plan
        plan = self._load_plan(plan_path)

        self.rate_hz: float = float(plan.get("rate_hz", 20))
        self.topic_control: str = str(plan.get("topic_control", "/fsds/control_command"))
        self.topic_odom: str = str(plan.get("topic_odom", "/fsds/testing_only/odom"))

        self.log_csv: bool = bool(plan.get("log_csv", True))
        self.csv_path: str = str(plan.get("csv_path", os.path.expanduser("~/fsds_test_log.csv")))

        self.steps: List[Step] = self._parse_steps(plan.get("steps", []))
        if not self.steps:
            raise RuntimeError("Plan has no steps. Add steps in the YAML file.")

        #  ROS subs and pubs
        self.pub = self.create_publisher(ControlCommand, self.topic_control, 10)
        self.sub_odom = self.create_subscription(Odometry, self.topic_odom, self._on_odom, 50)

        period = 1.0 / max(1.0, self.rate_hz)
        self.timer = self.create_timer(period, self._tick)

        # State
        self.start_wall = time.time()
        self.step_index = 0
        self.step_start_wall = time.time()

        self.last_odom: Optional[Odometry] = None

        # CSV file
        self.csv_file = None
        self.csv_writer = None
        if self.log_csv:
            os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
            self.csv_file = open(self.csv_path, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                "wall_elapsed_s",
                "step_index",
                "step_name",
                "cmd_throttle",
                "cmd_steering",
                "cmd_brake",
                "odom_time_s",
                "speed_mps",
                "x",
                "y",
            ])
            self.csv_file.flush()

        self.get_logger().info(f"Loaded plan: {plan_path}")
        self.get_logger().info(f"Publishing to: {self.topic_control} @ {self.rate_hz} Hz")
        self.get_logger().info(f"Subscribing to: {self.topic_odom}")
        if self.log_csv:
            self.get_logger().info(f"Logging CSV to: {self.csv_path}")
        self.get_logger().info(f"Steps: {len(self.steps)} (starting now)")

        self._announce_step()

    def destroy_node(self):
        # Stops the car on shutdown
        try:
            self._publish_cmd(0.0, 0.0, 1.0)  # Brake when finished.
        except Exception:
            pass

        if self.csv_file:
            try:
                self.csv_file.flush()
                self.csv_file.close()
            except Exception:
                pass
        super().destroy_node()

    def _load_plan(self, path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise RuntimeError("YAML plan must be a mapping/object.")
        return data

    def _parse_steps(self, raw_steps: Any) -> List[Step]:
        if not isinstance(raw_steps, list):
            raise RuntimeError("YAML steps must be a list.")
        steps: List[Step] = []
        for i, s in enumerate(raw_steps):
            if not isinstance(s, dict):
                raise RuntimeError(f"Step {i} is not a mapping.")
            steps.append(
                Step(
                    name=str(s.get("name", f"step_{i}")),
                    duration_s=float(s.get("duration_s", 0.0)),
                    throttle=float(s.get("throttle", 0.0)),
                    steering=float(s.get("steering", 0.0)),
                    brake=float(s.get("brake", 0.0)),
                )
            )

        for st in steps:
            if st.duration_s <= 0.0:
                raise RuntimeError(f"Step '{st.name}' has impossible duration_s: {st.duration_s}")
        return steps

    def _on_odom(self, msg: Odometry):
        self.last_odom = msg

    def _odom_time_s(self, msg: Odometry) -> float:
        return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

    def _current_step(self) -> Step:
        return self.steps[self.step_index]

    def _announce_step(self):
        st = self._current_step()
        self.get_logger().info(
            f"[{self.step_index+1}/{len(self.steps)}] {st.name}: "
            f"{st.duration_s:.2f}s | thr={st.throttle:.2f} steer={st.steering:.2f} brk={st.brake:.2f}"
        )

    def _publish_cmd(self, throttle: float, steering: float, brake: float):
        msg = ControlCommand()
        msg.header.frame_id = "fsds/FSCar"  

        msg.throttle = float(clamp(throttle, 0.0, 1.0))
        msg.steering = float(clamp(steering, -1.0, 1.0))
        msg.brake = float(clamp(brake, 0.0, 1.0))

        self.pub.publish(msg)

    def _log_row(self, st: Step):
        if not self.csv_writer:
            return

        wall_elapsed = time.time() - self.start_wall

        odom_t = ""
        speed = ""
        x = ""
        y = ""

        if self.last_odom is not None:
            odom_t = f"{self._odom_time_s(self.last_odom):.9f}"
            speed = f"{self.last_odom.twist.twist.linear.x:.6f}"
            x = f"{self.last_odom.pose.pose.position.x:.6f}"
            y = f"{self.last_odom.pose.pose.position.y:.6f}"

        self.csv_writer.writerow([
            f"{wall_elapsed:.6f}",
            self.step_index,
            st.name,
            f"{st.throttle:.3f}",
            f"{st.steering:.3f}",
            f"{st.brake:.3f}",
            odom_t,
            speed,
            x,
            y,
        ])
        self.csv_file.flush()

    def _tick(self):
        # Publish current step command. 
        st = self._current_step()
        self._publish_cmd(st.throttle, st.steering, st.brake)
        self._log_row(st)

        # advance if step duration elapsed.
        if (time.time() - self.step_start_wall) >= st.duration_s:
            self.step_index += 1
            if self.step_index >= len(self.steps):
                self.get_logger().info("Plan complete. Stopping vehicle.")
                self._publish_cmd(0.0, 0.0, 1.0)
                rclpy.shutdown()
                return

            self.step_start_wall = time.time()
            self._announce_step()


def main():
    rclpy.init()
    node = TestSequencer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()