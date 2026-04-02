#!/usr/bin/env python3
from __future__ import annotations

import math
import time
from collections import deque
import numpy as np

import json
from std_msgs.msg import String

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import TwistWithCovarianceStamped
from fs_msgs.msg import ControlCommand

from .car_model import KinematicBicycleModel
from .mpc_optimiser import KinematicBicycleMPC

"""
- CONTROLLER -
This is the ROS2 node that brings everything together. 

This is your main entry point to running the model predictive control in the sim.

With your simulator, trajectory publisher, and ROS bridge all running, do:

    ros2 run fsds_control mpc_controller

Parameters can be overriden at launch, for example:

    ros2 run fsds_control mpc_controller --ros-args -p horizon:=20 -p dt:=0.1 -p target_speed:=7.0  

- INPUT OUTPUT-
It subscribes to: 
    
    - /reference_path from the trajectory publisher
    - /fsds/odom from the sim
    - /fsds/gss from the sim

It then runs the MPC at a fixed rate, publishing control commands to the car.

It outputs this as:

    - /fsds/control_command - steering, throttle, brake.

- CONTROL LOOP -
At each tick, the controller:
    
    Localises on the path, finding the nearest waypoint using a heading aware search.

    Builds a reference trajectory. For each step in the MPC horizon it computes a
    reference position, heading, curvature, and target speed. 

    Checks safety bounds like cross-track error and heading error.

    Solves the MPC by passing the reference trajectory and current state to the 
    optimiser.

    Applies post-solve hard limits like a hard steering rate limit before
    publishing.

- CURVATURE ESTIMATION -
It estimates curvature using Menger curvature, fitting a circle between three
points and returning 1/radius.

- BRAKE AND THROTTLE - 
The controller applies two layers of longitudinal control on top of MPC output.
A curvature and steering aware brake ceiling blends between straight_brake_max
and corner_brake_max using a smooth S curve, which prevents heavy breaking mid
corner which can destabilise the car. THen, slew rate limits on both throttle 
and break prevent sudden actuator changes in spite of what the solver 
requests.

"""

# - HELPER FUNCTIONS -

def quaternion_to_yaw(q) -> float:
    # Extracts yaw from ROS quaternion.
    # Only need yaw as the circuits are flat.
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def wrap_to_pi(angle: float) -> float:
    # Wrap an angle to [-pi, pi]
    # Particularly important as if you subtract normally you'd get error = 179 - (-179) = 358° for example.
    # In reality the difference would be 2 degrees. 
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

def clamp(x: float, lo: float, hi: float) -> float:
    # Clamps used for steering, throttle, speeds, and so on.
    # Needed to clamp values to a set limit.
    return max(lo, min(hi, x))

def smoothstep01(x: float) -> float:
    # This smooths the transition over a value of 0 to 1.
    # Smooth Hermite interpolation.
    # Used when changing from straight to cornering behaviour.
    x = clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

def find_closest_index(path_xy: np.ndarray, x: float, y: float, yaw: float) -> int:
    # Finds the closest index on the trajectory path.
    # A basic distance search was problematic because it often latched to an index pointing the wrong way or behind the car.
    # So, this filters out any candidates that are beyond 100 degrees of the car's front.
    dx = path_xy[:, 0] - x
    dy = path_xy[:, 1] - y
    d2 = dx * dx + dy * dy

    candidate_count = min(20, len(path_xy)) # Candidates limited to 20 for per tick performance.
    candidate_idxs = np.argsort(d2)[:candidate_count]

    best_idx = int(candidate_idxs[0])
    best_score = float('inf')

    for idx in candidate_idxs:
        path_yaw = get_path_tangent_yaw(path_xy, int(idx))
        heading_diff = abs(wrap_to_pi(path_yaw - yaw))

        if heading_diff > math.radians(100.0): # Ignore any that are behind the car.
            continue

        score = d2[int(idx)] + 2.0 * (heading_diff ** 2)

        if score < best_score:
            best_score = score
            best_idx = int(idx)

    return best_idx

def get_path_tangent_yaw(path_xy: np.ndarray, idx: int, smooth: int = 1) -> float:
    # Estimate the path heading at a given waypoint using a finite difference.
    # Smooth parameter sets how many waypoints on either side of the path to span.
    # A higher value will impact precision of tight corners.
    n = len(path_xy)
    i_prev = (idx - smooth) % n
    i_next = (idx + smooth) % n
    dx = path_xy[i_next, 0] - path_xy[i_prev, 0]
    dy = path_xy[i_next, 1] - path_xy[i_prev, 1]
    return math.atan2(dy, dx)

def get_path_curvature(path_xy: np.ndarray, idx: int, smooth: int = 2) -> float:
    # Estimates the path curvature at a waypoint by using MEnger curvature.
    # Menger curvature fits a circle through three points and returns 1/radius.

    n = len(path_xy)
    i_a = (idx - smooth) % n
    i_b = idx % n
    i_c = (idx + smooth) % n

    ax, ay = path_xy[i_a]
    bx, by = path_xy[i_b]
    cx, cy = path_xy[i_c]

    # The triangle side lengths
    lab = math.hypot(bx - ax, by - ay)
    lbc = math.hypot(cx - bx, cy - by)
    lac = math.hypot(cx - ax, cy - ay)

    # Signed area of the triangle via cross product.
    # Positive = left turn, negative = right turn.
    area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    denom = lab * lbc * lac
    if abs(denom) < 1e-9 or abs(area) < 1e-9:
        return 0.0

    kappa = 2.0 * area / denom

    return kappa

def curvature_to_speed(curvature: float, v_min: float, v_max: float, a_lat_max: float) -> float:
    # Caclulates the max safe speed for any given path curvature.

    kappa = abs(curvature)

    if kappa < 1e-6:
        return v_max

    v_curve = math.sqrt(a_lat_max / kappa)
    return clamp(v_curve, v_min, v_max)


# - ROS2 NODE -

class MPCController(Node):

    def __init__(self):
        super().__init__('mpc_controller')

        self.declare_parameter('target_speed', 16.0) # straight line target speed (m/s)
        self.declare_parameter('min_speed', 4.0) # minimum speed for curvature_to_speed (m/s)
        self.declare_parameter('max_speed', 25.0) # maximumum speed for curvature_to_speed (m/s)
        self.declare_parameter('corner_curvature_threshold', 0.20) # 1/m, for brake blending
        self.declare_parameter('a_lat_max', 2.5) # lateral acceleration limit (m/s^2)
        self.declare_parameter('speed_lookahead', 25) # number of waypoints to look ahead for speed planning
        self.declare_parameter('horizon', 25) # MPC prediction horizon steps
        self.declare_parameter('dt', 0.1) # MPC timestep (s)
        self.declare_parameter('rate_hz', 20.0) # control loop rate (hz)

        self.declare_parameter('topic_path', '/reference_path') 
        self.declare_parameter('topic_odom', '/fsds/testing_only/odom')
        self.declare_parameter('topic_gss', '/fsds/gss')
        self.declare_parameter('topic_control', '/fsds/control_command')

        self.declare_parameter('estop_cte_m', 1.5) # cross track error emergency stop threshold (m)
        self.declare_parameter('estop_heading_deg', 60.0) # heading error emergency stop threshold (degs)
        self.declare_parameter('estop_stop_steer', 0.0) # to fix steering on stop

        self.declare_parameter('straight_brake_max', 1.0) # max braking on straight
        self.declare_parameter('corner_brake_max', 0.15) # max brake mid corner
        self.declare_parameter('brake_rate_up', 0.04) # brake application rate per tick
        self.declare_parameter('brake_rate_down', 0.12) # brake release rate per tick
        self.declare_parameter('throttle_rate_up', 0.08) # throttle application rate per tick
        self.declare_parameter('throttle_rate_down', 0.12) # throttle release rate per tick
        self.declare_parameter('steer_turn_threshold', 0.18) # steering magnitude per tick
        self.declare_parameter('brake_curvature_deadband', 0.085) # deadband for brake on straight vs brake on corner

        self.declare_parameter('timing_log_period_sec', 1.0) 
        self.declare_parameter('timing_window_size', 100) 
        self.declare_parameter('warn_on_deadline_miss', True) # warn if solve time exceeds control budget
        self.declare_parameter('topic_debug', '/mpc_debug')

        target_speed = self.get_parameter('target_speed').value
        min_speed = self.get_parameter('min_speed').value
        max_speed = self.get_parameter('max_speed').value
        corner_kappa_thresh = self.get_parameter('corner_curvature_threshold').value
        a_lat_max = self.get_parameter('a_lat_max').value
        speed_lookahead = self.get_parameter('speed_lookahead').value
        estop_cte_m = self.get_parameter('estop_cte_m').value
        estop_heading_deg = self.get_parameter('estop_heading_deg').value
        estop_stop_steer = self.get_parameter('estop_stop_steer').value

        N = self.get_parameter('horizon').value
        dt = self.get_parameter('dt').value
        rate_hz = self.get_parameter('rate_hz').value

        timing_log_period_sec = self.get_parameter('timing_log_period_sec').value
        timing_window_size = self.get_parameter('timing_window_size').value
        warn_on_deadline_miss = self.get_parameter('warn_on_deadline_miss').value
        topic_debug = self.get_parameter('topic_debug').value

        topic_path = self.get_parameter('topic_path').value
        topic_odom = self.get_parameter('topic_odom').value
        topic_gss = self.get_parameter('topic_gss').value
        topic_control = self.get_parameter('topic_control').value

        self.get_logger().info(
            f"MPC: N={N}, dt={dt}, target_speed={target_speed} m/s, rate={rate_hz} Hz"
        )

        self.model = KinematicBicycleModel(dt=dt)
        self.mpc = KinematicBicycleMPC(model=self.model, N=N, dt=dt, target_speed=target_speed)
        self.N = N
        self.dt = dt
        self.rate_hz = float(rate_hz)
        self.control_period_sec = 1.0 / self.rate_hz
        self.control_budget_ms = self.control_period_sec * 1000.0
        self.base_target_speed = float(target_speed)
        self.min_speed = float(min_speed)
        self.max_speed = float(max_speed)
        self.corner_curvature_threshold = float(corner_kappa_thresh)
        self.a_lat_max = float(a_lat_max)
        self.speed_lookahead = int(speed_lookahead)

        self.straight_brake_max = float(self.get_parameter('straight_brake_max').value)
        self.corner_brake_max = float(self.get_parameter('corner_brake_max').value)
        self.brake_rate_up = float(self.get_parameter('brake_rate_up').value)
        self.brake_rate_down = float(self.get_parameter('brake_rate_down').value)
        self.throttle_rate_up = float(self.get_parameter('throttle_rate_up').value)
        self.throttle_rate_down = float(self.get_parameter('throttle_rate_down').value)
        self.steer_turn_threshold = float(self.get_parameter('steer_turn_threshold').value)
        self.brake_curvature_deadband = float(self.get_parameter('brake_curvature_deadband').value)

        self.estop_cte_m = float(estop_cte_m)
        self.estop_heading_rad = math.radians(float(estop_heading_deg))
        self.estop_stop_steer = float(estop_stop_steer)
        self.estop_active = False

        self.path_xy: np.ndarray | None = None
        self.odom: Odometry | None = None
        self.gss: TwistWithCovarianceStamped | None = None
        self.u_prev = [0.0, 0.25, 0.0] 

        self.timing_log_period_sec = float(timing_log_period_sec)
        self.warn_on_deadline_miss = bool(warn_on_deadline_miss)

        window_size = int(timing_window_size)
        self.solve_times_ms = deque(maxlen=window_size)
        self.tick_times_ms = deque(maxlen=window_size)

        self.total_solves = 0
        self.solve_failures = 0
        self.deadline_misses = 0
        self.max_solve_time_ms = 0.0
        self.max_tick_time_ms = 0.0
        self.last_timing_log_time = time.perf_counter()

        path_qos = QoSProfile(depth=1)
        path_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(Path, topic_path, self._path_cb, path_qos)
        self.create_subscription(Odometry, topic_odom, self._odom_cb, 10)
        self.create_subscription(TwistWithCovarianceStamped, topic_gss, self._gss_cb, 10)

        self.pub = self.create_publisher(ControlCommand, topic_control, 10)
        self.debug_pub = self.create_publisher(String, topic_debug, 10)
        self.create_timer(1.0 / rate_hz, self._tick)

        self.get_logger().info("MPC controller ready. Waiting for path and odometry...")

    def _publish_debug(
        self,
        timestamp_sec: float,
        waypoint_idx: int,
        x: float,
        y: float,
        yaw: float,
        speed: float,
        ref_speed: float,
        cte: float,
        heading_error_rad: float,
        steer_cmd: float,
        throttle: float,
        brake: float,
        solve_time_ms: float | None,
        tick_time_ms: float | None,
        solver_success: bool,
        solver_status: str,
        solver_iters: int | None,
        deadline_miss: bool,
    ):
        msg = String()
        payload = {
            'timestamp_sec': float(timestamp_sec),
            'waypoint_idx': int(waypoint_idx),
            'x': float(x),
            'y': float(y),
            'yaw_rad': float(yaw),
            'yaw_deg': float(math.degrees(yaw)),
            'speed_mps': float(speed),
            'ref_speed_mps': float(ref_speed),
            'cte_m': float(cte),
            'path_length_points': 0 if self.path_xy is None else int(len(self.path_xy)),
            'heading_error_rad': float(heading_error_rad),
            'heading_error_deg': float(math.degrees(heading_error_rad)),
            'steer_cmd': float(steer_cmd),
            'throttle_cmd': float(throttle),
            'brake_cmd': float(brake),
            'solve_time_ms': None if solve_time_ms is None else float(solve_time_ms),
            'tick_time_ms': None if tick_time_ms is None else float(tick_time_ms),
            'solver_success': bool(solver_success),
            'solver_status': str(solver_status),
            'solver_iters': None if solver_iters is None else int(solver_iters),
            'deadline_miss': bool(deadline_miss),
            'estop_active': bool(self.estop_active),
            
        }
        msg.data = json.dumps(payload)
        self.debug_pub.publish(msg)

    def _path_cb(self, msg: Path):
        if not msg.poses:
            self.get_logger().warn("Received empty path.")
            return
        self.path_xy = np.array([[ps.pose.position.x, ps.pose.position.y] for ps in msg.poses])
        self.get_logger().info(f"Path received: {len(self.path_xy)} waypoints.")

    def _odom_cb(self, msg: Odometry):
        self.odom = msg

    def _gss_cb(self, msg: TwistWithCovarianceStamped):
        self.gss = msg

    def _trigger_estop(self, reason: str):
        self.estop_active = True
        self.u_prev = [0.0, 0.0, 1.0]
        self.get_logger().error(f"EMERGENCY STOP: {reason}")
        self._publish(self.estop_stop_steer, 0.0, 1.0)

    def _hold_estop(self):
        self._publish(self.estop_stop_steer, 0.0, 1.0)

        now_msg = self.get_clock().now().nanoseconds / 1e9
        self._publish_debug(
            timestamp_sec=now_msg,
            waypoint_idx=-1,
            x=0.0 if self.odom is None else float(self.odom.pose.pose.position.x),
            y=0.0 if self.odom is None else float(self.odom.pose.pose.position.y),
            yaw=0.0 if self.odom is None else float(quaternion_to_yaw(self.odom.pose.pose.orientation)),
            speed=0.0 if self.gss is None else float(self.gss.twist.twist.linear.x),
            ref_speed=0.0,
            cte=0.0,
            heading_error_rad=0.0,
            steer_cmd=self.estop_stop_steer,
            throttle=0.0,
            brake=1.0,
            solve_time_ms=None,
            tick_time_ms=None,
            solver_success=False,
            solver_status='ESTOP_HOLD',
            solver_iters=None,
            deadline_miss=False,
        )

    # - BRAKE LIMIT FOR UPCOMING CORNER -

    def _compute_brake_limit(self, upcoming_curvature: float, steer_cmd_guess: float) -> float:
        # Calculate the max brake command allowed given an upcoming corner.
        # Heavy braking locks the wheels and restricts turning which will destablish the car.
        # This brake limit blends with straight_brake_max and corner_brake_max as curvature and steering increase.

        kappa_mag = abs(upcoming_curvature)
        curvature_mix = smoothstep01(kappa_mag / max(self.corner_curvature_threshold, 1e-6))
        steer_mix = smoothstep01(abs(steer_cmd_guess) / max(self.steer_turn_threshold, 1e-6))
        turn_mix = max(curvature_mix, steer_mix)
        return (1.0 - turn_mix) * self.straight_brake_max + turn_mix * self.corner_brake_max

    def _slew(self, cmd: float, prev: float, up_step: float, down_step: float) -> float:
        # Rate limit any command to prevent abrupt changes.
        if cmd > prev:
            return min(cmd, prev + up_step)
        return max(cmd, prev - down_step)

    def _record_solver_stats(self):
        stats = getattr(self.mpc, 'last_solve_stats', None)
        if not stats:
            return

        solve_time_ms = stats.get('solve_time_ms', None)
        success = bool(stats.get('success', False))
        return_status = stats.get('return_status', 'UNKNOWN')
        iter_count = stats.get('iter_count', None)

        if solve_time_ms is not None:
            self.solve_times_ms.append(float(solve_time_ms))
            self.max_solve_time_ms = max(self.max_solve_time_ms, float(solve_time_ms))

            if solve_time_ms > self.control_budget_ms:
                self.deadline_misses += 1
                if self.warn_on_deadline_miss:
                    self.get_logger().warn(
                        f"MPC deadline miss: solve_time={solve_time_ms:.2f} ms > "
                        f"budget={self.control_budget_ms:.2f} ms "
                        f"(status={return_status}, iters={iter_count})",
                        throttle_duration_sec=0.5,
                    )

        self.total_solves += 1
        if not success:
            self.solve_failures += 1

    def _log_timing_summary_if_due(self):
        now = time.perf_counter()
        if now - self.last_timing_log_time < self.timing_log_period_sec:
            return

        self.last_timing_log_time = now

        avg_solve = sum(self.solve_times_ms) / len(self.solve_times_ms) if self.solve_times_ms else 0.0
        avg_tick = sum(self.tick_times_ms) / len(self.tick_times_ms) if self.tick_times_ms else 0.0
        failure_rate = (100.0 * self.solve_failures / self.total_solves) if self.total_solves > 0 else 0.0
        miss_rate = (100.0 * self.deadline_misses / self.total_solves) if self.total_solves > 0 else 0.0

        last_stats = getattr(self.mpc, 'last_solve_stats', {})
        last_status = last_stats.get('return_status', 'UNKNOWN')
        last_iters = last_stats.get('iter_count', None)

        self.get_logger().info(
            f"MPC TIMING: avg_solve={avg_solve:.2f} ms, "
            f"max_solve={self.max_solve_time_ms:.2f} ms, "
            f"avg_tick={avg_tick:.2f} ms, "
            f"max_tick={self.max_tick_time_ms:.2f} ms, "
            f"budget={self.control_budget_ms:.2f} ms, "
            f"solves={self.total_solves}, "
            f"failures={self.solve_failures} ({failure_rate:.1f}%), "
            f"deadline_misses={self.deadline_misses} ({miss_rate:.1f}%), "
            f"last_status={last_status}, "
            f"last_iters={last_iters}",
        )

    # - MAIN CONTROL LOOP -

    def _tick(self):

        tick_t0 = time.perf_counter()

        if self.estop_active:
            self._hold_estop()
            return
        if self.path_xy is None:
            self.get_logger().warn("Waiting for reference path...", throttle_duration_sec=5.0)
            return
        if self.odom is None:
            self.get_logger().warn("Waiting for odometry...", throttle_duration_sec=5.0)
            return
        if self.gss is None:
            self.get_logger().warn("Waiting for GSS velocity...", throttle_duration_sec=5.0)
            return

        pose = self.odom.pose.pose
        x0 = pose.position.x
        y0 = pose.position.y
        yaw0 = quaternion_to_yaw(pose.orientation)
        v0 = float(self.gss.twist.twist.linear.x)

        idx = find_closest_index(self.path_xy, x0, y0, yaw0)
        n = len(self.path_xy)

        # - BUILD REFERENCE TRAJECTORY FOR MPC -

        ref_x = [] # position x
        ref_y = [] # position y
        ref_yaw = [] # heading 
        ref_v = [] # speed
        ref_kappa = [] # curvatures
        lookahead_curvatures = [] # waypoints further ahead than current horizon position

        for k in range(self.N):
            ref_idx = (idx + k) % n
            ref_x.append(float(self.path_xy[ref_idx, 0]))
            ref_y.append(float(self.path_xy[ref_idx, 1]))
            ref_yaw.append(get_path_tangent_yaw(self.path_xy, ref_idx))
            ref_kappa.append(get_path_curvature(self.path_xy, ref_idx))

            # Speed planning looks further ahead than the tracking horion so
            # the car brakes before the corner enters the MPC window.
            kappa_idx = (ref_idx + self.speed_lookahead) % n
            kappa = get_path_curvature(self.path_xy, kappa_idx)
            kappa_mag = abs(kappa)
            lookahead_curvatures.append(kappa)
            
            # Speed target
            if kappa_mag < self.brake_curvature_deadband:
                v_ref_k = self.base_target_speed
            else:
                v_ref_k = curvature_to_speed(
                    curvature=kappa,
                    v_min=self.min_speed,
                    v_max=self.max_speed,
                    a_lat_max=self.a_lat_max,
                )
                v_ref_k = min(v_ref_k, self.base_target_speed)

            ref_v.append(v_ref_k)

        # - SAFETY CHECKS -
        # Cross track and heading threshold
        ex0 = x0 - ref_x[0]
        ey0 = y0 - ref_y[0]
        cte0 = ex0 * math.sin(ref_yaw[0]) - ey0 * math.cos(ref_yaw[0])
        he0  = wrap_to_pi(yaw0 - ref_yaw[0])

        # - CROSS TRACK ERROR EMERGENCY STOP -
        if abs(cte0) > self.estop_cte_m:
            self._trigger_estop(
                f"CROSS-TRACK ERROR TOO HIGH: |{cte0:.3f}| METRES > {self.estop_cte_m:.3f} METRES. SAFETY STOP TRIGGERED."
            )
            return

        # - HEADING ERROR EMERGENCY STOP - 
        if abs(he0) > self.estop_heading_rad:
            self._trigger_estop(
                f"HEADING ERROR TOO HIGH: |{math.degrees(he0):.1f}| DEG > "
                f"{math.degrees(self.estop_heading_rad):.1f} DEG. SAFETY STOP TRIGGERED."
            )
            return

        self.get_logger().info(
            f"TRACKING INFO: waypoint={idx}, "
            f"target=({ref_x[0]:.1f},{ref_y[0]:.1f}), "
            f"vehicle=({x0:.1f},{y0:.1f}), "
            f"cross-track error={cte0:.3f} m, "
            f"heading error={math.degrees(he0):.1f}°, ",
            throttle_duration_sec=1.0,
        )

        # - BRAKE AND THROTTLE CAPS -
        # preview curvature is the worst curvature observed in the next 20 lookahead samples.
        # throttle and brake caps to limit throttle and brake entering and exiting corners
        preview_samples = sorted((abs(k) for k in lookahead_curvatures[:min(16, len(lookahead_curvatures))]), reverse=True)
        top_n = preview_samples[:4]
        preview_curvature = sum(top_n) / max(len(top_n), 1)
        brake_cap = self._compute_brake_limit(preview_curvature, self.u_prev[0])
        throttle_cap = 0.70 if brake_cap > 0.3 else 0.95

        steer_cmd, throttle, brake = self.mpc.solve(
            x0, y0, yaw0, v0,
            ref_x, ref_y, ref_yaw, ref_v, ref_kappa,
            a_lat_limit=self.a_lat_max,
            u_prev=self.u_prev,
            brake_ub=brake_cap,
            throttle_ub=throttle_cap,
        )

        self._record_solver_stats()

        # - POST SOLVE LIMITS -
        MAX_DSTEER = 0.30 # Hard limit on steering rate per tick.
        steer_cmd = clamp(steer_cmd, self.u_prev[0] - MAX_DSTEER, self.u_prev[0] + MAX_DSTEER)

        # Limit throttle and brake to prevent inputs that may upset car.
        throttle = self._slew(throttle, self.u_prev[1], self.throttle_rate_up, self.throttle_rate_down)
        brake = self._slew(brake, self.u_prev[2], self.brake_rate_up, self.brake_rate_down)

        # Prevent simultaneous throttle and brake with a small tolerance.
        if brake > 0.03:
            throttle = min(throttle, 0.05)
        if throttle > 0.12:
            brake = min(brake, 0.02)

        self.u_prev = [steer_cmd, throttle, brake]
        self._publish(steer_cmd, throttle, brake=brake)

        self.get_logger().info(
            f"VEHICLE STATE: position=({x0:.1f},{y0:.1f}), "
            f"heading={math.degrees(yaw0):.1f}°, "
            f"speed={v0:.2f} m/s, "
            f"ref_speed={ref_v[0]:.2f} m/s, "
            f"steering command={steer_cmd:.3f}, "
            f"throttle command={throttle:.3f}, "
            f"brake command={brake:.3f}, ",
            throttle_duration_sec=0.5,
        )

        tick_elapsed_ms = (time.perf_counter() - tick_t0) * 1000.0
        self.tick_times_ms.append(tick_elapsed_ms)
        self.max_tick_time_ms = max(self.max_tick_time_ms, tick_elapsed_ms)
        self._log_timing_summary_if_due()

        last_stats = getattr(self.mpc, 'last_solve_stats', {})
        solve_time_ms = last_stats.get('solve_time_ms', None)
        solver_success = bool(last_stats.get('success', False))
        solver_status = last_stats.get('return_status', 'UNKNOWN')
        solver_iters = last_stats.get('iter_count', None)
        deadline_miss = (
            solve_time_ms is not None and solve_time_ms > self.control_budget_ms
        )

        now_msg = self.get_clock().now().nanoseconds / 1e9

        self._publish_debug(
            timestamp_sec=now_msg,
            waypoint_idx=idx,
            x=x0,
            y=y0,
            yaw=yaw0,
            speed=v0,
            ref_speed=ref_v[0],
            cte=cte0,
            heading_error_rad=he0,
            steer_cmd=steer_cmd,
            throttle=throttle,
            brake=brake,
            solve_time_ms=solve_time_ms,
            tick_time_ms=tick_elapsed_ms,
            solver_success=solver_success,
            solver_status=solver_status,
            solver_iters=solver_iters,
            deadline_miss=deadline_miss,
        )

    # - PUBLISH CONTROLCOMMAND AFTER FINAL LIMITING AND CLEANUP -
    def _publish(self, steer: float, throttle: float, brake: float):
        msg = ControlCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'fsds/FSCar'

        steer = float(clamp(steer, -1.0, 1.0))
        throttle = float(clamp(throttle, 0.0, 1.0))
        brake = float(clamp(brake, 0.0, 1.0))

        if throttle < 1e-3:
            throttle = 0.0
        if brake < 1e-3:
            brake = 0.0

        if brake > 0.0:
            throttle = 0.0

        msg.steering = -steer
        msg.throttle = throttle
        msg.brake = brake
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = MPCController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish(0.0, 0.0, 1.0)
        if rclpy.ok():
            rclpy.shutdown()
        node.destroy_node()


if __name__ == '__main__':
    main()