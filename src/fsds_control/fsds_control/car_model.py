#!/usr/bin/env python3
from __future__ import annotations

import math
import casadi as ca

"""
- KINEMATIC BICYCLE MODEL -
This is the definition of the car model used by the MPC optimiser. It describes how
the car moves.

A real car has four wheels, suspension, tyre slip, weight transfer, and many other effects. 
Modelling this can improve accuracy, but can also make the problem intractable. This bicycle
model is a deliberate simplification.

This model collapses the four wheels into two, hence why it's called a bicycle model. It
ignores forces entirely, assuming the wheels can never slip.

This is a reasonable approximation at the speeds and lateral accelerations seen in 
FOrmula Student. However at the limit of grip, this solution could fall short.

- STATE AND CONTROLS -
The model tracks four state variables per timestep:
    x,y - the position in global frame
    yaw - heading angle (radians)
    v - forward speed (m/s)

The MPC has three control inputs:
    steer_cmd - steering in [-1, 1]
    throttle - [0,1]
    brake - [0,1]

- SPEED DYNAMICS -
No engine torque or drivetrain is modelled, speed is instead controlled by a simple
lag toward a throttle-determined target velocity, minus a brake deceleration term.
This captures the broad shape of longitudinal acceleration behaviour, such as 
smooth acceleration or sharp braking, with only a few tunable constants (K_SPEED,
K_BRAKE, V_MAX etc). Why? Because these are easier to observe and identify from
testing data.
"""


class KinematicBicycleModel:

    # Kinematic bicycle vehicle model used by the MPC.
    # Values taken from Unreal Engine Simulator and testing.

    L = 1.581872
    DELTA_MAX = math.radians(25.0)
    V_MAX = 25.346969892 # Car is faster than this but will never go beyond such a speed.
    K_SPEED = 0.158
    K_BRAKE = 7.5
    THROTTLE_EXP = 1.0

    def __init__(self, dt: float = 0.1):
        self.dt = dt

    def build_symbolic_dynamics(self):
        x_s = ca.SX.sym('x')
        y_s = ca.SX.sym('y')
        yaw_s = ca.SX.sym('yaw')
        v_s = ca.SX.sym('v')
        state = ca.vertcat(x_s, y_s, yaw_s, v_s)

        steer_s = ca.SX.sym('steer')
        throttle_s = ca.SX.sym('throttle')
        brake_s = ca.SX.sym('brake')
        ctrl = ca.vertcat(steer_s, throttle_s, brake_s)

        delta = steer_s * self.DELTA_MAX
        v_tgt = self.V_MAX * (throttle_s ** self.THROTTLE_EXP)
        accel_drive = self.K_SPEED * (v_tgt - v_s)
        accel_brake = self.K_BRAKE * brake_s
        accel = accel_drive - accel_brake

        x_next = x_s + v_s * ca.cos(yaw_s) * self.dt
        y_next = y_s + v_s * ca.sin(yaw_s) * self.dt
        yaw_next = yaw_s + (v_s / self.L) * ca.tan(delta) * self.dt
        v_next = ca.fmax(0.0, v_s + accel * self.dt)

        f = ca.Function(
            'f',
            [state, ctrl],
            [ca.vertcat(x_next, y_next, yaw_next, v_next)],
        )
        return f