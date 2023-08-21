from collections import deque
import math
import os
from threading import Thread, Lock
import time
from typing import List, Sequence
import struct
import sys

import klampt
try:
    from motionlib import so3, se3, vectorops as vo
except ImportError:
    from klampt.math import so3, se3, vectorops as vo
from klampt.model import ik
import numpy as np
import cvxpy as cp

try:
    import trina
except ImportError:
    sys.path.append(os.path.expanduser("~/TRINA"))
    import trina
from trina.utils import profiling as prof
from Motion.abstract_controller import track_methods, include_method
from Motion.kinematic_controller import KlamptModelController, ControlMode

import franka_motion

IK_FAIL_BUFFER_SIZE = 20
IK_FAIL_THRESHOLD = 15

@track_methods
class KinematicFrankaController(KlamptModelController):
    """Class for running franka robot control logic.

    Works in kinematic and physical mode. Responsible for stuff like IK heuristics
    """

    def __init__(self, motion_inst, name, robot_model, EE_link, collision_checker, params):
        """
        params arguments:

        - elbow_lookahead: Scan step when doing elbow optimization
        - elbow_speed: Scan step multiplier when doing elbow optimization (should be 1 tbh)
        - qmin: software joint limits (min)
        - qmax: software joint limits (max)
        - kinematic_home: Home config in kinematic mode (since all 0s is in collision)
        """
        super().__init__(motion_inst, name, robot_model, EE_link, collision_checker, params)

        # Janky: Reading link 0, assumed to be UR base link
        self.base_transform = robot_model.link(name+":base_link").getTransform()
        self.shoulder_pos = robot_model.link(name+":panda_link1").getTransform()[1]
        print(name, self.shoulder_pos, self.base_transform)

        self.elbow_link = robot_model.link(name+":elbow_link")
        self.measured_elbow_transform = None

        self.min_drivers = np.array(params.get('qmin', self.qmin))
        self.max_drivers = np.array(params.get('qmax', self.qmax))

        # Kinematic simulation
        self.step_config = params.get('kinematic_home', [0.0]*7)
        robot_model.setConfig(robot_model.configFromDrivers(self.step_config))

        self.IK_fail_flag = False           # True if previous IK solve attempt failed
        self.IK_fail_count = 0
        self.IK_fail_buffer = deque([0]*IK_FAIL_BUFFER_SIZE)
        self.end_of_travel_flag = False     # Flag for being close to max extension (measured by elbow angle).
        self.self_collision_flag = False    # Not used in kinematic mode; used in physical to record self collision stops

        if self._feature_flag == 0:

            # Costs for different things.
            # Cartesian position error
            self._W_x = np.diag([1, 1, 1, 1, 1, 1])
            # Joint velocity
            self._W_v = np.diag([1, 1, 1, 1, 1, 1, 1]) * 0.02
            # Joint position bias
            self._W_b = np.diag([1, 1, 1, 1, 1, 1, 1]) * 0.0005

            self._J = cp.Parameter((6, 7))
            self._dq = cp.Variable(7)
            self._dx = cp.Parameter(6)
            self._q = cp.Parameter(7)
            self._q_bias = cp.Parameter(7)
            self._bias_config = self.step_config#self.min_drivers + self.max_drivers / 2;

            target_objective = cp.quad_form(self._dx - self._J @ self._dq, self._W_x)
            velocity_objective = cp.quad_form(self._dq, self._W_v)
            bias_objective = cp.quad_form(self._q - self._q_bias + self._dq, self._W_b)
            self._objective = cp.Minimize(target_objective + velocity_objective + bias_objective)

            joint_limits = [self.min_drivers <= self._q + self._dq,
                            self.max_drivers >= self._q + self._dq]

            # x, y, z, roll, pitch, yaw
            self._active_constraints = np.array([False, False, False, False, False, False], dtype=bool)
            self._reproject = False
            self._cvx_problem = cp.Problem(self._objective, joint_limits)
            self._qp_constraints_warn = False

    def _update_cvx_constraints(self, new_constraints):
        """Update convex problem constraints.

        Parameters:
        --------------------
            new_constraints:    Array[bool, 6]      constraint status to set (x, y, z, r, p, y)
        """
        change_happened = sum(np.logical_xor(self._active_constraints, new_constraints) > 0)
        if change_happened:
            self._reproject = sum(new_constraints) > 0
            self._active_constraints = new_constraints

            constraints = [self.min_drivers <= self._q + self._dq,
                           self.max_drivers >= self._q + self._dq]

            J_prod = (self._J @ self._dq)
            # Jacobian order is [rx, ry, rz, x, y, z].
            jac_order = [3, 4, 5, 0, 1, 2]
            for i, b in enumerate(new_constraints):
                if b:
                    constraints.append(J_prod[jac_order[i]] == 0)
            self._cvx_problem = cp.Problem(self._objective, constraints)

    def beginStep(self) -> None:
        super().beginStep()
        self.end_of_travel_flag = bool(self.measured_config[3] > -1.1)  # STUPID thing since json can't serialize _bool dumb
        self.measured_elbow_transform = self.elbow_link.getTransform()

    def _elbow_angle_heuristic(self, hand_transform, EE_transform):
        """Compute joint0 angle heuristic.

        Try to bias the elbow outwards on three conditions:
        1. Wrist is pitched downwards.
        2. Wrist is yawed inwards.
        3. Hand is raised upwards.

        k_angle controls the effect of pitch and yaw.
        k_height controls the effect of hand raising upwards (above h0).
        h0 controls the height at which hand raising starts having an effect (relative to the shoulder).

        r_min is the minimum "low" elbow angle (rad).
        r_max is the maximum "high" elbow angle (rad).
        """
        k_angle = 0.8
        k_height = 2
        h0 = -0.3
        r_min = 0.1
        r_max = np.pi/2

        rotvec = so3.rotation_vector(EE_transform[0])    # Extrinsic, XYZ
        hand_raised_dist = max(0, hand_transform[1][2] - (self.shoulder_pos[2] + h0))
        if self._feature_flag == 1:
            if self.get_name().startswith('left'):
                # positive is out
                r_min += min(max(0, hand_transform[1][1] - self.shoulder_pos[1]), 0.5)
                elbow_rotate_heuristic = r_min + max(0, min(max(k_angle*(abs(rotvec[1]) - rotvec[2]), 0) - k_height*hand_raised_dist, r_max-r_min))
            else:
                # negative is out
                r_min += min(max(0, self.shoulder_pos[1] - hand_transform[1][1]), 0.5)
                elbow_rotate_heuristic = -r_min + min(0, -min(max(k_angle*(abs(rotvec[1]) + rotvec[2]), 0) - k_height*hand_raised_dist, r_max-r_min))
        else:
            if self.get_name().startswith('left'):
                # positive is out
                elbow_rotate_heuristic = max(r_min, min(max(k_angle*(rotvec[1] - rotvec[2]), 0) + k_height*hand_raised_dist, r_max))
            else:
                # negative is out
                elbow_rotate_heuristic = min(-r_min, -min(max(k_angle*(rotvec[1] + rotvec[2]), 0) + k_height*hand_raised_dist, r_max))
        return elbow_rotate_heuristic

    def _elbow_heuristic(self, hand_transform, EE_transform) -> List[float]:
        #cur_elbow_pos = self.elbow_link.getTransform()[1]
        #cur_elbow_pos[2] -= 1
        #return cur_elbow_pos

        R, t = EE_transform
        elbow_target = list(t)
        cur_elbow_pos = self.elbow_link.getTransform()[1]
        elbow_target[0] = cur_elbow_pos[0]
        elbow_target[2] -= 0.15
        blend_ratio = max(min(self.shoulder_pos[2] - t[2] + 0.4, 0.5), 0) / 0.5
        elbow_target[1] = elbow_target[1] * blend_ratio + cur_elbow_pos[1] * (1 - blend_ratio)
        elbow_target[2] -= 0.05*(1-blend_ratio)

        if elbow_target[2] > self.shoulder_pos[2] + 0.1:
            elbow_target[2] = self.shoulder_pos[2] + 0.1

        if self.get_name().startswith('left'):
            if elbow_target[1] < self.shoulder_pos[1] + 0.2:
                fix_target = self.shoulder_pos[1] + 0.1
                blend_ratio = max(elbow_target[1] - fix_target, 0) / 0.1
                elbow_target[1] = (1-blend_ratio)*fix_target + blend_ratio*elbow_target[1]
        else:
            if elbow_target[1] > self.shoulder_pos[1] - 0.2:
                fix_target = self.shoulder_pos[1] - 0.1
                blend_ratio = max(fix_target - elbow_target[1], 0) / 0.1
                elbow_target[1] = (1-blend_ratio)*fix_target + blend_ratio*elbow_target[1]
        return elbow_target

    def drive_EE(self, target, params):
        """Compute the target joint configuration to send to the franka driver.
        based on a target end effector pose.

        Valid params:
            tool_center:    SE3     TCP transform relative to franka EE
            elbow:          Vec3    Target elbow location

        Parameters:
        --------------------
            target:         SE3     target end effector position from teleop.
            params:         dict    Other controller parameters, ex. tool center

        Return:
        --------------------
        (success, Union(config, None))
        """
        robot_model = self.klamptModel()
        hand_transform = target
        tool_offset = params.get('tool_center', se3.identity())
        target = se3.mul(target, se3.inv(tool_offset))

        #m_bar = (0.1, 2)    # TODO: move to settings/tune
        m_bar = 0.04    # TODO: move to settings/tune
        repel_step = 0.001  # TODO: vector
        R, t = self._singularity_avoidance(target, m_bar, repel_step, actives=list(range(1, 7)))

        if 'elbow' in params:
            goal = ik.objective(self.get_EE_link(), R=R, t=t)
            solver = ik.solver(goal, iters=100, tol=1e-3)
            solver.setActiveDofs(self.driven_dofs)
            elbow_target = params['elbow']
            secondary_objective = ik.objective(self.elbow_link, local=[0,0,0], world=elbow_target)
            solver.addSecondary(secondary_objective)
        elif self._feature_flag == 0:
            # cvxpy solve
            constraints = params.get('constraints', [False]*6)
            if len(constraints) != 6:
                if not self._qp_constraints_warn:
                    print(f"{self.get_name()}: Bad input length for constraints (expected Array(bool, 6))")
                    self._qp_constraints_warn = True
                constraints = [False]*6

            self._update_cvx_constraints(constraints)

            dx = se3.error(target, self.get_EE_link().getTransform())
            q = np.array(robot_model.configToDrivers(robot_model.getConfig()))
            self._J.value = self.get_EE_jacobian([0, 0, 0])
            self._dx.value = dx
            self._q.value = q
            self._q_bias.value = self._bias_config

            self._cvx_problem.solve()

            if self._reproject:
                # Gotta reproject onto constraint manifold.
                # For now this is all global constraints cause it makes my life easier :thumbsup:
                q_targ = q + self._dq.value
                robot_model.setConfig(robot_model.configFromDrivers(q_targ))
                R_c, t_c = target
                R_i, t_i = self.get_EE_link().getTransform()
                R_delta = so3.mul(R_i, so3.inv(R_c))
                R_delta_rpy = list(so3.rpy(R_delta))

                t_f = list(t_i)
                for i in range(3):
                    if constraints[i]:
                        t_f[i] = t_c[i]
                for i in range(3):
                    if constraints[i+3]:
                        R_delta_rpy[i] = 0

                R_f = so3.mul(so3.from_rpy(R_delta_rpy), R_c)

                # mfw we just throw all the nice guarantees of cvx IK out the window
                target_transform = (R_f, t_f)
                goal = ik.objective(self.get_EE_link(), R=R_f, t=t_f)
                solver = ik.solver(goal, iters=100, tol=1e-3)
            else:
                return (True, q + self._dq.value)
        elif self._feature_flag == 1:
            goal = ik.objective(self.get_EE_link(), R=R, t=t)
            solver = ik.solver(goal, iters=100, tol=1e-3)
            elbow_rotate_heuristic = self._elbow_angle_heuristic(hand_transform, target)
            edit_config = np.array(robot_model.getConfig())
            edit_config[self.driven_dofs[0]] = elbow_rotate_heuristic
            #qmin, qmax = solver.getJointLimits()
            #qmin[self.driven_dofs[0]] = max(qmin[self.driven_dofs[0]], elbow_rotate_heuristic - 0.5)
            #qmax[self.driven_dofs[0]] = min(qmax[self.driven_dofs[0]], elbow_rotate_heuristic + 0.5)
            robot_model.setConfig(edit_config)
            #solver.setJointLimits(qmin, qmax)
            solver.setActiveDofs(self.driven_dofs)
        elif self._feature_flag == 2:
            goal = ik.objective(self.get_EE_link(), R=R, t=t)
            solver = ik.solver(goal, iters=100, tol=1e-3)
            elbow_target = self._elbow_heuristic(hand_transform, target)
            secondary_objective = ik.objective(self.elbow_link, local=[0,0,0], world=elbow_target)
            solver.addSecondary(secondary_objective)

        if solver.minimize():
            cfg = robot_model.configToDrivers(robot_model.getConfig())
            return (True, cfg)

        return (False, None)

    def update_IK_failure(self, status: bool):
        self.IK_fail_buffer.append(int(status))
        self.IK_fail_count += status - self.IK_fail_buffer.popleft()
        self.IK_fail_flag = self.IK_fail_count > IK_FAIL_THRESHOLD

    @prof.profiled
    def endStep(self) -> None:
        """Control the robot.

        In EE mode, attempts to pull the elbow towards
            a provided (or guessed) position in space.
        """
        robot_model = self.klamptModel()
        save_config = robot_model.getConfig()

        with self.control_lock:
            control_mode = self.control_mode
            target = self.target
            params = self.controller_params

        if control_mode == ControlMode.POSITION:
            self.step_config = target
            self.update_IK_failure(False)

        elif control_mode == ControlMode.POSITION_EE:
            success, cfg = self.drive_EE(target, params)
            self.update_IK_failure(not success)
            if success:
                self.step_config = cfg
            else:
                pass
                #print("ik solve fail", solver.getResidual(), vo.norm(solver.getResidual()), solver.getSecondaryResidual())

        else:
            self.update_IK_failure(False)

        robot_model.setConfig(save_config)

    @include_method
    def to_dict(self):
        with self.control_lock:
            ret = super().to_dict()
            ret['flags'] = {
                'end_of_travel': self.end_of_travel_flag,
                'IK_fail': self.IK_fail_flag,
                'self_collision': self.self_collision_flag
            }
            #print([f"{x}: {type(ret['flags'][x])}" for x in ret['flags']])
        return ret
