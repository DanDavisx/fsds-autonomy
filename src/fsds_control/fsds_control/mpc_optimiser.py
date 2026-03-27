#!/usr/bin/env python3
from __future__ import annotations

import math
import casadi as ca

from .car_model import KinematicBicycleModel

"""
- MPC OPTIMISER - 
This file builds and solves the MPC optimisation problem that generates steering, throttle,
and brake commands for the car.

- MPC -
The Model Predictive Control works by repeatedly solving an optimisation problem at each control
timestep. Given the cars current state, it finds the sequence of control inputs over the next
N time steps that minimises a cost function, provided the car obeys the vehicle dynamics at 
each step. On the next tick the whole process repeats with the updated state.

- COST FUNCTION -
The optimiser minimises a weighted sum of penalties across the horizon:

    Tracking terms - cross-track error, heading error, speed error.

    Actuator terms - steering, throttle, brake magnitude.

    Smoothness terms - rate of change of each actuator between steps.

    Cornering terms - predicted lateral acceleration excess, throttle use
    in tight corner sections.

    Overlap penalty - simultaneous throttle and brake application.

- IMPLEMENTATION -
The problem is formulated with CasADi (the mathematical problem), and solved by IPOPT (the optimiser).
The problem is compiled once at startup in _build() into a reusable solver object. Each call to solve() 
supplies fresh state and reference data, warm-starts from a straight line prediction, and then returns
only the first control action (the rest is discarded).

If IPOPT fails to converge, solve() returns a gentle coast command rahter than crashing, and
should recover normally on the next tick.
"""

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class KinematicBicycleMPC:

    # State: [x, y, yaw, v]
    # Controls: [steer_cmd in [-1,1], throttle in [0,1], brake in [0,1]]

    def __init__(self, model: KinematicBicycleModel, N: int = 10, dt: float = 0.1, target_speed: float = 3.0):
        self.model = model
        self.N = N # prediction horizon length
        self.dt = dt # timestep (s)
        self.target_speed = target_speed
        self._solver = None
        self._build()

    def _build(self):
        # Construct the NLP (nonlinear program) using CasAdi.
        # Everything here is symbolic, so CasAdi builds a computation graph.
        # The actual numbers are supplied later in solve().

        N = self.N

        # - GET SYMBOLIC VEHICLE DYNAMICS -

        f = self.model.build_symbolic_dynamics()

        # - DECISION VARIABLES -

        X = ca.SX.sym('X', 4, N + 1)
        U = ca.SX.sym('U', 3, N)

        # - PARAMETER VECTOR - 
        # Parameters:[x0, y0, yaw0, v0, a_lat_limit,ref_x[0:N], ref_y[0:N], ref_yaw[0:N], ref_v[0:N], ref_kappa[0:N]]
        n_p = 5 + 5 * N
        p = ca.SX.sym('p', n_p)

        # Unpack the parameter vector into named variables
        a_lat_limit = p[4]
        ref_x = p[5 : 5 + N]
        ref_y = p[5 + N : 5 + 2*N]
        ref_yaw = p[5 + 2*N : 5 + 3*N]
        ref_v = p[5 + 3*N : 5 + 4*N]
        ref_kappa = p[5 + 4*N : 5 + 5*N]

        # - COST FUNCTION WEIGHTS -
        # These determine how much the solver cares about each objective.

        W_CTE = 3.0 # cross track error 
        W_HEADING = 2.5 # heading alignment with the path
        W_SPEED = 1.0 # tracking the speed reference
        W_STEER = 0.5 # steering magnitude
        W_THROTTLE = 0.1 # throttle magnitude
        W_BRAKE = 0.6 # brake magnitude
        W_DSTEER = 10.0 # steer rate smoothness
        W_DTHROTTLE = 0.5 # throttle rate
        W_DBRAKE = 6.0 # brake rate
        W_OVERLAP = 8.0 # simultaneous throttle and brake
        W_ALAT_EXCESS = 12.0 # lateral acceleration over the limit
        W_CURVE_THROTTLE = 0.75 # throttle in high curvature sections

        cost = 0.0
        constraints = []

        constraints.append(X[:, 0] - p[:4])

        # Prevent any near zero values by flooring lat acceleration limit.
        a_lat_limit_safe = ca.fmax(0.1, a_lat_limit)

        # - HORIZON LOOP -
        # Builds the cost and dynamics constraints for each step K in [0, N)
        for k in range(N):
            st = X[:, k] # predicted state at step k
            con = U[:, k] # control applied at step k

            px = st[0] # predicted x
            py = st[1] # predicted y
            pyaw = st[2] # predicted headinmg
            pv = st[3] # predicted speed

            rx = ref_x[k] # ref x at timestep k 
            ry = ref_y[k] # ref y at timestep k
            ryaw = ref_yaw[k] # ref heading at timestep k

            # - CROSS TRACK ERROR -
            # Signed lateral distance from the reference path.
            ex = px - rx
            ey = py - ry
            cte = ex * ca.sin(ryaw) - ey * ca.cos(ryaw)

            # - HEADING ERROR -
            # Angular difference between the predicted and reference yaw
            dh = pyaw - ryaw
            heading_err = ca.atan2(ca.sin(dh), ca.cos(dh))

            # - LATERAL ACCELERATION TERMS -
            # a_lat_pred estimatesthe lateral acceleration the car will experience at this step given its speed and path curvature.
            # a_lat_excess is how much it exceeds that limit.
            kappa_mag = ca.fabs(ref_kappa[k])
            a_lat_pred = pv * pv * kappa_mag
            a_lat_excess = ca.fmax(0.0, a_lat_pred - a_lat_limit_safe)
            curve_gate = ca.fmin(1.0, a_lat_pred / a_lat_limit_safe)

            # - TRACKING COSTS -
            cost += W_CTE * cte ** 2
            cost += W_HEADING * heading_err ** 2
            cost += W_SPEED * (pv - ref_v[k]) ** 2
            # - ACTUATOR COSTS -
            cost += W_STEER * con[0] ** 2
            cost += W_THROTTLE * con[1] ** 2
            cost += W_BRAKE * con[2] ** 2
            # - SIMULTANEOUS THROT + BRAKE PENMALTY -
            cost += W_OVERLAP * (con[1] * con[2]) ** 2

            # - CORNER COSTS -
            # THis reasoning exists here inside the optimiser so that it backs off proactively rather than
            # just following the externally computed speed reference.
            cost += W_ALAT_EXCESS * a_lat_excess ** 2
            cost += W_CURVE_THROTTLE * curve_gate * con[1] ** 2

            # - RATE COSTS -
            if k > 0:
                cost += W_DSTEER * (U[0, k] - U[0, k-1]) ** 2
                cost += W_DTHROTTLE * (U[1, k] - U[1, k-1]) ** 2
                cost += W_DBRAKE * (U[2, k] - U[2, k-1]) ** 2

            # - DYNAMICS CONSTRAINTS -
            # The state at k + 1 must equal what the model predicts from state k under control k.
            st_next = f(st, con)
            constraints.append(X[:, k+1] - st_next)

        # - TERMINAL COST -
        # Terminal costs are weighted more heavily than the per step costs.
        # Keeps the predicted trajectory in good shape at the end of the horizon.
        stN = X[:, N]
        pxN = stN[0]
        pyN = stN[1]
        pyawN = stN[2]
        pvN = stN[3]

        rxN = ref_x[N - 1]
        ryN = ref_y[N - 1]
        ryawN = ref_yaw[N - 1]
        rvN = ref_v[N - 1]
        kappaN = ca.fabs(ref_kappa[N - 1])

        exN = pxN - rxN
        eyN = pyN - ryN
        cteN = exN * ca.sin(ryawN) - eyN * ca.cos(ryawN)

        dhN = pyawN - ryawN
        heading_errN = ca.atan2(ca.sin(dhN), ca.cos(dhN))

        a_lat_predN = pvN * pvN * kappaN
        a_lat_excessN = ca.fmax(0.0, a_lat_predN - a_lat_limit_safe)

        cost += 5.0 * W_CTE * cteN ** 2
        cost += 8.0 * W_HEADING * heading_errN ** 2
        cost += 2.0 * W_SPEED * (pvN - rvN) ** 2
        cost += 15.0 * W_ALAT_EXCESS * a_lat_excessN ** 2

        # - NLP ASSEMBLY AND COMPILATION - 

        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        g = ca.vertcat(*constraints)

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

        n_states = 4 * (N + 1)
        self._u_offset = n_states
        self._lbx = [-ca.inf] * n_states + [-1.0, 0.0, 0.0] * N
        self._ubx = [ ca.inf] * n_states + [ 1.0, 1.0, 1.0] * N
        n_eq = 4 * (N + 1)
        self._lbg = [0.0] * n_eq
        self._ubg = [0.0] * n_eq

    # - SOLVE MPC -

    def solve(self, x0: float, y0: float, yaw0: float, v0: float, 
    ref_x: list[float], ref_y: list[float], ref_yaw: list[float], ref_v: list[float], ref_kappa: list[float],
    a_lat_limit: float, u_prev: list[float] | None = None, brake_ub: float = 1.0, throttle_ub: float = 1.0,
    ) -> tuple[float, float, float]:

        # Solve the MPC. Returns (steer_cmd, throttle, brake) for the first step.
        N = self._N

        p_val = [x0, y0, yaw0, v0, a_lat_limit] + list(ref_x) + list(ref_y) + list(ref_yaw) + list(ref_v) + list(ref_kappa)
        if u_prev is None:
            u_prev = [0.0, 0.3, 0.0]

        x_init = []
        sx, sy, syaw, sv = x0, y0, yaw0, v0
        for i in range(N + 1):
            x_init += [sx, sy, syaw, sv]
            if i < N:
                sx += sv * math.cos(syaw) * self.dt
                sy += sv * math.sin(syaw) * self.dt
        u_init = list(u_prev) * N
        x0_nlp = x_init + u_init

        lbx = list(self._lbx)
        ubx = list(self._ubx)
        brake_ub = clamp(brake_ub, 0.0, 1.0)
        throttle_ub = clamp(throttle_ub, 0.0, 1.0)
        for k in range(N):
            base = self._u_offset + 3 * k
            ubx[base + 1] = throttle_ub
            ubx[base + 2] = brake_ub

        try:
            sol = self._solver(
                x0=x0_nlp,
                lbx=lbx,
                ubx=ubx,
                lbg=self._lbg,
                ubg=self._ubg,
                p=p_val,
            )
            opt = sol['x'].full().flatten()
            base = self._u_offset
            steer_cmd = float(opt[base])
            throttle = float(opt[base + 1])
            brake = float(opt[base + 2])
            return (
                clamp(steer_cmd, -1.0, 1.0),
                clamp(throttle, 0.0, throttle_ub),
                clamp(brake, 0.0, brake_ub),
            )
        except Exception:
            return 0.0, 0.05, 0.0
