"""
The kinematic bicycle model.

Wheelbase: 1.582 m	
Max steer angle: 25 deg
Mass: 202 kg	
Drag coefficient: 0.3	
Wheel radius: 0.2525 m	
Brake torque: 350 FRONT / 180 REAR Nm	

"""

from dataclasses import dataclass
import math


@dataclass
class VehicleParams:
    # Lateral / geometry
    wheelbase_m: float = 1.581872
    delta_max_rad: float = math.radians(25.0)

    # Longitudinal
    v_max: float = 25.346969892 # m/s
    k_speed: float = 0.158 # 1/s (speed response rate)
    throttle_exponent: float = 0.5  # 0.5 = sqrt, 1.0 = linear

    # Braking: calibrated + simple saturation (matches your 0.3/0.6/1.0 tests)
    brake_max: float = 8.68 # m/s^2 (max decel)
    brake_sat: float = 0.37 # brake cmd where max decel is reached


@dataclass
class State:
    x: float
    y: float
    yaw: float
    v: float


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def wrap_to_pi(angle_rad: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def step(
    state: State,
    steer_cmd: float,
    throttle: float,
    brake: float,
    dt: float,
    params: VehicleParams
) -> State:
    """
    Advance the kinematic bicycle model one step with Euler discretisation.

    Lateral (kinematic bicycle):
        yaw_dot = (v/L) * tan(delta)

    Longitudinal (simple, good for varying throttle):
        v_target = v_max * throttle^p
        dv/dt    = k_speed * (v_target - v) - brake_max * brake_eff
        brake_eff = min(1, brake / brake_sat)  (simple saturation)
    """
    if dt <= 0.0:
        return state

    # Clamp normalised inputs
    steer_cmd = clamp(steer_cmd, -1.0, 1.0)
    throttle  = clamp(throttle, 0.0, 1.0)
    brake     = clamp(brake, 0.0, 1.0)

    # Map steering to physical steer angle (front wheel)
    delta = steer_cmd * params.delta_max_rad

    # Longitudinal model
    v_target = params.v_max * (throttle ** params.throttle_exponent)

    # Simple braking saturation () tests show 0.6 ~= 1.0 )
    if params.brake_sat <= 1e-9:
        brake_eff = brake
    else:
        brake_eff = min(1.0, brake / params.brake_sat)

    accel = params.k_speed * (v_target - state.v) - params.brake_max * brake_eff

    # Discrete update 
    x_next = state.x + state.v * math.cos(state.yaw) * dt
    y_next = state.y + state.v * math.sin(state.yaw) * dt
    yaw_next = state.yaw + (state.v / params.wheelbase_m) * math.tan(delta) * dt
    v_next = max(0.0, state.v + accel * dt)

    yaw_next = wrap_to_pi(yaw_next)

    return State(x=x_next, y=y_next, yaw=yaw_next, v=v_next)