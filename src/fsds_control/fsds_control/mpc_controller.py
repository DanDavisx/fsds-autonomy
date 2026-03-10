"""
Model Predictive Controller for the FSDS Formula Student Driverless Simulator.

Subscribes to:
    /reference_path          (nav_msgs/Path)                    - reference centreline
    /fsds/testing_only/odom  (nav_msgs/Odometry)                - vehicle pose
    /fsds/gss                (geometry_msgs/TwistWithCovarianceStamped) - vehicle velocity

Publishes to:
    /fsds/control_command    (fs_msgs/ControlCommand) - throttle, steering, brake

The MPC minimises over a horizon:
    - Cross-track error  (lateral deviation from path, computed from PREDICTED states)
    - Heading error      (yaw alignment with path tangent, from PREDICTED states)
    - Speed error        (deviation from target speed)
    - Control smoothness (penalise large control changes)

Key design: reference path waypoints are passed as NLP parameters so the cost
depends on the predicted trajectory X[:,k], not the current car position.
"""

#!/usr/bin/env python3
from __future__ import annotations

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import TwistWithCovarianceStamped
from fs_msgs.msg import ControlCommand

import casadi as ca


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def quaternion_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# Make this heading aware to stop it latching onto a wrong nearby segment on spawn.
def find_closest_index(path_xy: np.ndarray, x: float, y: float, yaw: float) -> int:
    dx = path_xy[:, 0] - x
    dy = path_xy[:, 1] - y
    d2 = dx * dx + dy * dy

    # Look at a small set of nearest candidates first
    candidate_count = min(20, len(path_xy))
    candidate_idxs = np.argsort(d2)[:candidate_count]

    best_idx = int(candidate_idxs[0])
    best_score = float('inf')

    for idx in candidate_idxs:
        path_yaw = get_path_tangent_yaw(path_xy, int(idx))
        heading_diff = abs(wrap_to_pi(path_yaw - yaw))

        # Reject candidates whose tangent points too far away from vehicle heading
        if heading_diff > math.radians(100.0):
            continue

        # Combined score: distance dominates, heading agreement helps
        score = d2[int(idx)] + 2.0 * (heading_diff ** 2)

        if score < best_score:
            best_score = score
            best_idx = int(idx)

    return best_idx


def get_path_tangent_yaw(path_xy: np.ndarray, idx: int) -> float:
    n = len(path_xy)
    next_idx = (idx + 1) % n
    dx = path_xy[next_idx, 0] - path_xy[idx, 0]
    dy = path_xy[next_idx, 1] - path_xy[idx, 1]
    return math.atan2(dy, dx)

def get_path_curvature(path_xy: np.ndarray, idx: int) -> float:
    """
    Approximate curvature at path index idx using change in tangent angle
    over local arc length. Returns signed curvature in 1/m.
    """
    n = len(path_xy)
    i_prev = (idx - 1) % n
    i_curr = idx % n
    i_next = (idx + 1) % n

    yaw_prev = get_path_tangent_yaw(path_xy, i_prev)
    yaw_curr = get_path_tangent_yaw(path_xy, i_curr)

    d_yaw = wrap_to_pi(yaw_curr - yaw_prev)

    dx = path_xy[i_next, 0] - path_xy[i_prev, 0]
    dy = path_xy[i_next, 1] - path_xy[i_prev, 1]
    ds = math.hypot(dx, dy)

    if ds < 1e-6:
        return 0.0

    return d_yaw / ds


def curvature_to_speed(curvature: float,
                       v_min: float,
                       v_max: float,
                       a_lat_max: float) -> float:
    """
    Convert path curvature to a safe reference speed using:
        v <= sqrt(a_lat_max / |curvature|)
    """
    kappa = abs(curvature)

    if kappa < 1e-6:
        return v_max

    v_curve = math.sqrt(a_lat_max / kappa)
    return clamp(v_curve, v_min, v_max)


# ---------------------------------------------------------------------------
# MPC Builder
# ---------------------------------------------------------------------------

class KinematicBicycleMPC:
    """
    Builds a CasADi NLP for a kinematic bicycle model MPC.

    State:    [x, y, yaw, v]
    Controls: [steer_cmd in [-1,1], throttle in [0,1]]

    Parameters passed to NLP each solve:
        p = [x0, y0, yaw0, v0,
             ref_x_0..N-1,   ref_y_0..N-1,
             ref_yaw_0..N-1]
        (4 + 3*N total)

    CTE and heading error are computed INSIDE the NLP from predicted states X[:,k],
    so the cost genuinely depends on the control decisions.
    """

    L            = 1.581872
    DELTA_MAX    = math.radians(25.0)
    V_MAX        = 25.346969892
    K_SPEED      = 0.158
    THROTTLE_EXP = 1.0

    def __init__(self, N: int = 10, dt: float = 0.1, target_speed: float = 3.0):
        self.N            = N
        self.dt           = dt
        self.target_speed = target_speed
        self._solver      = None
        self._build()

    def _build(self):
        N  = self.N
        dt = self.dt

        # ---- Symbolic state & control ----
        x_s   = ca.SX.sym('x')
        y_s   = ca.SX.sym('y')
        yaw_s = ca.SX.sym('yaw')
        v_s   = ca.SX.sym('v')
        state = ca.vertcat(x_s, y_s, yaw_s, v_s)

        steer_s    = ca.SX.sym('steer')
        throttle_s = ca.SX.sym('throttle')
        ctrl       = ca.vertcat(steer_s, throttle_s)

        # ---- Discrete dynamics ----
        delta   = steer_s * self.DELTA_MAX
        v_tgt   = self.V_MAX * (throttle_s ** self.THROTTLE_EXP)
        accel   = self.K_SPEED * (v_tgt - v_s)

        x_next   = x_s   + v_s * ca.cos(yaw_s) * dt
        y_next   = y_s   + v_s * ca.sin(yaw_s) * dt
        yaw_next = yaw_s + (v_s / self.L) * ca.tan(delta) * dt
        v_next   = ca.fmax(0.0, v_s + accel * dt)

        f = ca.Function('f', [state, ctrl],
                        [ca.vertcat(x_next, y_next, yaw_next, v_next)])

        # ---- NLP decision variables ----
        X = ca.SX.sym('X', 4, N + 1)
        U = ca.SX.sym('U', 2, N)

        # ---- Parameters: initial state + reference waypoints ----
        # p = [x0, y0, yaw0, v0, ref_x*N, ref_y*N, ref_yaw*N, ref_v*N]
        n_p = 4 + 4 * N
        p   = ca.SX.sym('p', n_p)

        ref_x   = p[4       : 4 +   N]
        ref_y   = p[4 +   N : 4 + 2*N]
        ref_yaw = p[4 + 2*N : 4 + 3*N]
        ref_v   = p[4 + 3*N : 4 + 4*N]

        # ---- Cost weights ----
        W_CTE       = 1.0
        W_HEADING   = 1.0
        W_SPEED     = 2.0
        W_STEER     = 0.5
        W_THROTTLE  = 0.1
        W_DSTEER    = 5.0
        W_DTHROTTLE = 0.5

        cost        = 0.0
        constraints = []

        # Initial state constraint
        constraints.append(X[:, 0] - p[:4])

        for k in range(N):
            st  = X[:, k]
            con = U[:, k]

            # Predicted position and heading
            px   = st[0]
            py   = st[1]
            pyaw = st[2]
            pv   = st[3]

            # Reference at this horizon step
            rx   = ref_x[k]
            ry   = ref_y[k]
            ryaw = ref_yaw[k]

            # Cross-track error: signed perpendicular distance
            # (positive = car is left of path)
            ex  = px - rx
            ey  = py - ry
            cte = ex * ca.sin(ryaw) - ey * ca.cos(ryaw)

            # Heading error, wrapped via atan2(sin,cos) for CasADi
            dh          = pyaw - ryaw
            heading_err = ca.atan2(ca.sin(dh), ca.cos(dh))

            # Stage cost
            cost += W_CTE * cte ** 2
            cost += W_HEADING * heading_err ** 2
            cost += W_SPEED * (pv - ref_v[k]) ** 2
            cost += W_STEER * con[0] ** 2
            cost += W_THROTTLE * con[1] ** 2

            if k > 0:
                cost += W_DSTEER    * (U[0, k] - U[0, k-1]) ** 2
                cost += W_DTHROTTLE * (U[1, k] - U[1, k-1]) ** 2

            # Dynamics constraint
            st_next = f(st, con)
            constraints.append(X[:, k+1] - st_next)

        # ---- Flatten for NLP ----
        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        g        = ca.vertcat(*constraints)

        nlp = {'f': cost, 'x': opt_vars, 'g': g, 'p': p}

        opts = {
            'ipopt.print_level': 0,
            'ipopt.max_iter':    150,
            'ipopt.tol':         1e-4,
            'print_time':        0,
        }

        self._solver = ca.nlpsol('mpc', 'ipopt', nlp, opts)
        self._n_p = n_p
        self._N = N
        self._f = f

        # ---- Bounds ----
        n_states   = 4 * (N + 1)
        self._lbx  = [-ca.inf] * n_states + [-1.0, 0.0] * N
        self._ubx  = [ ca.inf] * n_states + [ 1.0, 1.0] * N
        n_eq = 4 * (N + 1)
        self._lbg  = [0.0] * n_eq
        self._ubg  = [0.0] * n_eq

    def solve(self,
            x0: float, y0: float, yaw0: float, v0: float,
            ref_x: list[float], ref_y: list[float], ref_yaw: list[float], ref_v: list[float],
            u_prev: list[float] | None = None
            ) -> tuple[float, float]:
        """
        Solve the MPC. Returns (steer_cmd, throttle) for the first step.
        ref_x, ref_y, ref_yaw: lookahead reference waypoints, length N.
        """
        N = self._N

        p_val = [x0, y0, yaw0, v0] + list(ref_x) + list(ref_y) + list(ref_yaw) + list(ref_v)
        if u_prev is None:
            u_prev = [0.0, 0.3]

        # Warm-start: propagate initial state forward as guess
        x_init = []
        sx, sy, syaw, sv = x0, y0, yaw0, v0
        for i in range(N + 1):
            x_init += [sx, sy, syaw, sv]
            if i < N:
                sx  += sv * math.cos(syaw) * self.dt
                sy  += sv * math.sin(syaw) * self.dt
        u_init = list(u_prev) * N
        x0_nlp = x_init + u_init

        try:
            sol       = self._solver(
                x0=x0_nlp,
                lbx=self._lbx,
                ubx=self._ubx,
                lbg=self._lbg,
                ubg=self._ubg,
                p=p_val,
            )
            opt       = sol['x'].full().flatten()
            n_states  = 4 * (N + 1)
            steer_cmd = float(opt[n_states])
            throttle  = float(opt[n_states + 1])
            return clamp(steer_cmd, -1.0, 1.0), clamp(throttle, 0.0, 1.0)

        except Exception:
            return 0.0, 0.1


# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------

class MPCController(Node):

    def __init__(self):
        super().__init__('mpc_controller')

        self.declare_parameter('target_speed',3.0)
        self.declare_parameter('min_speed',2.0)
        self.declare_parameter('max_speed',8.0)
        self.declare_parameter('a_lat_max',4.0)
        self.declare_parameter('speed_lookahead',5)
        self.declare_parameter('horizon',20)
        self.declare_parameter('dt',0.1)
        self.declare_parameter('rate_hz',20.0)
        self.declare_parameter('topic_path','/reference_path')
        self.declare_parameter('topic_odom','/fsds/testing_only/odom')
        self.declare_parameter('topic_gss','/fsds/gss')
        self.declare_parameter('topic_control','/fsds/control_command')
        self.declare_parameter('estop_cte_m', 1.5)
        self.declare_parameter('estop_heading_deg', 60.0)
        self.declare_parameter('estop_stop_steer', 0.0)

        target_speed = self.get_parameter('target_speed').value
        min_speed = self.get_parameter('min_speed').value
        max_speed = self.get_parameter('max_speed').value
        a_lat_max = self.get_parameter('a_lat_max').value
        speed_lookahead = self.get_parameter('speed_lookahead').value
        estop_cte_m = self.get_parameter('estop_cte_m').value
        estop_heading_deg = self.get_parameter('estop_heading_deg').value
        estop_stop_steer = self.get_parameter('estop_stop_steer').value

        N = self.get_parameter('horizon').value
        dt = self.get_parameter('dt').value
        rate_hz = self.get_parameter('rate_hz').value
        topic_path = self.get_parameter('topic_path').value
        topic_odom = self.get_parameter('topic_odom').value
        topic_gss = self.get_parameter('topic_gss').value
        topic_control = self.get_parameter('topic_control').value

        self.get_logger().info(
            f"MPC: N={N}, dt={dt}, target_speed={target_speed} m/s, rate={rate_hz} Hz"
        )

        self.mpc = KinematicBicycleMPC(N=N, dt=dt, target_speed=target_speed)
        self.N = N
        self.dt = dt
        self.base_target_speed = target_speed
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.a_lat_max = a_lat_max
        self.speed_lookahead = speed_lookahead

        self.estop_cte_m = float(estop_cte_m)
        self.estop_heading_rad = math.radians(float(estop_heading_deg))
        self.estop_stop_steer = float(estop_stop_steer)
        self.estop_active = False

        self.path_xy: np.ndarray | None = None
        self.odom: Odometry | None = None
        self.gss: TwistWithCovarianceStamped | None = None
        self.u_prev = [0.0, 0.3]

        path_qos = QoSProfile(depth=1)
        path_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(Path, topic_path, self._path_cb, path_qos)
        self.create_subscription(Odometry, topic_odom, self._odom_cb, 10)
        self.create_subscription(
            TwistWithCovarianceStamped, topic_gss, self._gss_cb, 10)

        self.pub = self.create_publisher(ControlCommand, topic_control, 10)
        self.create_timer(1.0 / rate_hz, self._tick)

        self.get_logger().info("MPC controller ready. Waiting for path and odometry...")

    def _path_cb(self, msg: Path):
        if not msg.poses:
            self.get_logger().warn("Received empty path.")
            return
        self.path_xy = np.array(
            [[ps.pose.position.x, ps.pose.position.y] for ps in msg.poses]
        )
        self.get_logger().info(f"Path received: {len(self.path_xy)} waypoints.")

    def _odom_cb(self, msg: Odometry):
        self.odom = msg

    def _gss_cb(self, msg: TwistWithCovarianceStamped):
        self.gss = msg

    def _trigger_estop(self, reason: str):
        self.estop_active = True
        self.u_prev = [0.0, 0.0]
        self.get_logger().error(f"EMERGENCY STOP: {reason}")
        self._publish(self.estop_stop_steer, 0.0, 1.0)

    def _hold_estop(self):
        self._publish(self.estop_stop_steer, 0.0, 1.0)

    def _tick(self):
        if self.estop_active:
            self._hold_estop()
            return
        if self.path_xy is None:
            self.get_logger().warn(
                "Waiting for reference path...", throttle_duration_sec=5.0)
            return
        if self.odom is None:
            self.get_logger().warn(
                "Waiting for odometry...", throttle_duration_sec=5.0)
            return
        if self.gss is None:
            self.get_logger().warn(
                "Waiting for GSS velocity...", throttle_duration_sec=5.0)
            return

        # ---- Current state ----
        pose = self.odom.pose.pose
        x0 = pose.position.x
        y0 = pose.position.y
        yaw0 = quaternion_to_yaw(pose.orientation)
        v0 = float(self.gss.twist.twist.linear.x)

        # ---- Build lookahead reference waypoints ----
        idx = find_closest_index(self.path_xy, x0, y0, yaw0)
        n   = len(self.path_xy)

        ref_x = []
        ref_y = []
        ref_yaw = []
        ref_v = []

        for k in range(self.N):
            ref_idx = (idx + k) % n
            ref_x.append(float(self.path_xy[ref_idx, 0]))
            ref_y.append(float(self.path_xy[ref_idx, 1]))
            ref_yaw.append(get_path_tangent_yaw(self.path_xy, ref_idx))

            # Look a little ahead so we slow before the corner, not inside it
            kappa_idx = (ref_idx + self.speed_lookahead) % n
            kappa = get_path_curvature(self.path_xy, kappa_idx)

            v_ref_k = curvature_to_speed(
                curvature=kappa,
                v_min=self.min_speed,
                v_max=self.max_speed,
                a_lat_max=self.a_lat_max,
            )

            # Never ask for more than the user-set target speed cap
            v_ref_k = min(v_ref_k, self.base_target_speed)
            ref_v.append(v_ref_k)

        # ---- Debug log ----
        ex0 = x0 - ref_x[0]
        ey0 = y0 - ref_y[0]
        cte0 = ex0 * math.sin(ref_yaw[0]) - ey0 * math.cos(ref_yaw[0])
        he0  = wrap_to_pi(yaw0 - ref_yaw[0])

        if abs(cte0) > self.estop_cte_m:
            self._trigger_estop(
                f"CROSS-TRACK ERROR TOO HIGH: |{cte0:.3f}| METRES > {self.estop_cte_m:.3f} METRES. SAFETY STOP TRIGGERED."
            )
            return

        if abs(he0) > self.estop_heading_rad:
            self._trigger_estop(
                f"HEADING ERROR TOO HIGH: |{math.degrees(he0):.1f}| DEG > "
                f"{math.degrees(self.estop_heading_rad):.1f} DEG. SAFETY STOP TRIGGERED."
            )
            return

        #-TRACKING STATE LOGGING-
        self.get_logger().info(
            f"TRACKING INFO: waypoint={idx}, "
            f"target=({ref_x[0]:.1f},{ref_y[0]:.1f}), "
            f"vehicle=({x0:.1f},{y0:.1f}), "
            f"cross-track error={cte0:.3f} m, "
            f"heading error={math.degrees(he0):.1f}°, ",
            throttle_duration_sec=1.0,
        )

        # ---- Solve MPC ----
        steer_cmd, throttle = self.mpc.solve(
            x0, y0, yaw0, v0,
            ref_x, ref_y, ref_yaw, ref_v,
            u_prev=self.u_prev,
        )
        # Rate-limit steering to prevent violent oscillation
        MAX_DSTEER = 0.15  # max change per tick at 20 Hz
        steer_cmd = clamp(steer_cmd,
                        self.u_prev[0] - MAX_DSTEER,
                        self.u_prev[0] + MAX_DSTEER)
        self.u_prev = [steer_cmd, throttle]

        self._publish(steer_cmd, throttle, brake=0.0)

        # -VEHICLE STATE LOGGING-
        self.get_logger().info(
            f"VEHICLE STATE: position=({x0:.1f},{y0:.1f}), "
            f"heading={math.degrees(yaw0):.1f}°, "
            f"speed={v0:.2f} m/s, "
            f"steering command={steer_cmd:.3f} rad, "
            f"throttle command={throttle:.3f}, ",
            throttle_duration_sec=0.5,
        )
        
        # -STEERING CHECKS-
        self.get_logger().info(
            f"STEERING CHECKS: reference heading={math.degrees(ref_yaw[0]):.1f}°, "
            f"cross-track error={cte0:.3f} m, "
            f"commanded steering={steer_cmd:.3f} rad, ",
            throttle_duration_sec=0.5,
        )

    def _publish(self, steer: float, throttle: float, brake: float):
        msg                 = ControlCommand()
        msg.header.frame_id = 'fsds/FSCar'
        msg.steering        = -float(clamp(steer,    -1.0, 1.0))
        msg.throttle        = float(clamp(throttle,  0.0, 1.0))
        msg.brake           = float(clamp(brake,     0.0, 1.0))
        self.pub.publish(msg)


# ---------------------------------------------------------------------------

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