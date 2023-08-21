import os
from threading import Thread, Lock
import time
from typing import List, Sequence
import struct
import sys

import klampt
from klampt.math import so3, se3, vectorops as vo
from klampt.model import ik
import numpy as np

try:
    import trina
except ImportError:
    sys.path.append(os.path.expanduser("~/TRINA"))
    import trina
from Motion.abstract_controller import track_methods, include_method
from Motion.kinematic_controller import ControlMode
from Motion.Limb.panda.kinematic_controller import KinematicFrankaController

import franka_motion

@track_methods
class FrankaController(KinematicFrankaController):
    """Class for interfacing with the Franka robot.

    This code talks to our custom driver (implemented as a python extension module in C++) which runs in a separate thread.

    This is basically a driver wrapper. For control logic @see Motion/limb/panda/kinematic_controller.py
    """

    def __init__(self, motion_inst, name, robot_model, EE_link, collision_checker, params):
        """
        params arguments:

        - gravity: WORLD gravity vector (TRINA frame)
        - address: IP of the panda arm
        - payload: estimated payload in kg (default: 0)
        - impedance: 7D vector of joint impedances to set
        - elbow_lookahead: Scan step when doing elbow optimization
        - elbow_speed: Scan step multiplier when doing elbow optimization (should be 1 tbh)
        - qmin: software joint limits (min)
        - qmax: software joint limits (max)
        """
        super().__init__(motion_inst, name, robot_model, EE_link, collision_checker, params)

        # Computing gravity vector for panda
        world_gravity = params.get('gravity', [0, 0, -9.81])
        host = params['address']
        gravity  = so3.apply(so3.inv(self.base_transform[0]), world_gravity)
        impedance = params.get('impedance', [3000, 3000, 3000, 2500, 2500, 2000, 2000])

        gravity_center = params.get('com', [0, 0, 0])
        payload = params.get('payload', 0)

        self.driver = franka_motion.FrankaDriver(host, gravity, impedance, tool_center=(so3.identity(), gravity_center), payload=payload)

        self.started = False

        joint_lim_pad = params.get('joint_limit_padding', 0.1)
        self.min_drivers = np.array(params.get('qmin', self.qmin)) + joint_lim_pad
        self.max_drivers = np.array(params.get('qmax', self.qmax)) - joint_lim_pad
        robot_model.setJointLimits(robot_model.configFromDrivers(self.min_drivers),
                                   robot_model.configFromDrivers(self.max_drivers))

        self.measured_joint_torque = [0.0]*7
        self.measured_EE_wrench = [0.0]*6
        self.kp_soft = params.get('kp_soft', None)
        self.kd_soft = params.get('kd_soft', None)
        self.kp_hard = params.get('kp_hard', None)
        self.kd_hard = params.get('kd_hard', None)
        self.measured_accel = [0.0]*7

        # Collision checker state variables and configuration
        self.col_check_counter = params.get('col_check_offset', 0)
        self.col_check_modulus = params.get('col_check_modulus', 1)
        self.col_stop_timer = 0     # Counts down to zero.
        self.prev_safe_config = None
        # TODO: hack to zero ft sensor
        self.wrench_offset = [0.0]*6
        self.wrench_calibrate_samples = 100
        self.wrench_calibrate_state = self.wrench_calibrate_samples
        # Pstop time
        self.pstop_duration = 0.5 # in seconds

    @include_method
    def status(self) -> str:
        # TODO: pstop
        if self.started:
            time_since_pstop = self.driver.get_time_since_pstop()
            if time_since_pstop > self.pstop_duration: # Normal
                return super().status()
            else: # PSTOP
                return "protective_stop"
        return "idle"

    def initialize(self) -> bool:
        """Starts the franka driver.

        Franka driver is implemented as a python extension module in C++
        and uses libfranka to control the robot.
        """
        self.driver.start()
        for i in range(10):
            if self.driver.state_valid:
                self.started = True
                self.beginStep()
                return True
            time.sleep(1)
        return False

    def close(self) -> bool:
        del self.driver # for good measure
        self.started = False
        return True

    def beginStep(self) -> None:
        """Read state from franka driver, update robot model and measured params."""
        state = self.driver.get_state()
        self.measured_config = state['q']
        self.end_of_travel_flag = self.measured_config[3] > -1.1
        self.measured_vel = state['dq']
        self.measured_accel = state['ddq']
        self.measured_joint_torque = state['tau_J']

        wrench_robot_meas = state['EE_wrench']
        R_base_global = self.base_transform[0]
        self.measured_EE_wrench = (so3.apply(R_base_global, wrench_robot_meas[0:3])
                        + so3.apply(R_base_global, wrench_robot_meas[3:6]))
        if self.wrench_calibrate_state > 0:
            self.wrench_calibrate_state -= 1
            self.wrench_offset = vo.add(self.wrench_offset, self.measured_EE_wrench)
            if self.wrench_calibrate_state == 0:
                self.wrench_offset = vo.div(self.wrench_offset, self.wrench_calibrate_samples)
        else:
            self.measured_EE_wrench = vo.sub(self.measured_EE_wrench, self.wrench_offset)

        robot_model = self.klamptModel()
        old_config = robot_model.configToDrivers(robot_model.getConfig())
        robot_model.setConfig(robot_model.configFromDrivers(self.measured_config))
        self.measured_elbow_transform = self.elbow_link.getTransform()

        jac = self.get_EE_jacobian()
        self.measured_EE_vel = jac @ self.measured_vel
        self.measured_EE_transform = self.get_EE_link().getTransform()

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

        kp = params.get('kp', None)
        kd = params.get('kd', None)
        alpha = params.get('alpha', None)

        #if kp and kd:
        if kp == 'soft':
            kp = self.kp_soft
        elif kp == 'hard':
            kp = self.kp_hard
        if kd == 'soft':
            kd = self.kd_soft
        elif kd == 'hard':
            kd = self.kd_hard

        gain_vals = {'kp': kp, 'kd': kd, 'alpha': alpha}
        self.driver.set_gains(**{k: v for k, v in gain_vals.items() if v})

        target_drivers = None
        if self.control_mode == ControlMode.FREEDRIVE:
            target_drivers = self.measured_config
            self.update_IK_failure(False)
        elif control_mode == ControlMode.POSITION:
            target_drivers = target
            self.update_IK_failure(False)
        elif control_mode == ControlMode.POSITION_EE:
            success, cfg = self.drive_EE(target, params)
            self.update_IK_failure(not success)
            if success:
                target_drivers = cfg
        else:
            self.update_IK_failure(False)

        self.col_check_counter += 1
        do_col_check = self.col_check_counter == self.col_check_modulus
        if do_col_check:
            self.col_check_counter = 0
        if self.col_stop_timer > 0:
            self.col_stop_timer -= 1
        elif target_drivers is not None:
            initial_drivers = robot_model.configToDrivers(save_config)
            col_res = self.check_collision(initial_drivers, target_drivers)
            self.self_collision_flag = col_res
            if col_res:
                # Collision detected!
                self.col_stop_timer = self.col_check_modulus - 1

                if self.prev_safe_config is None:
                    print(f'{self.get_name()}: collision detected, exited..')
                    target_arm_config = initial_drivers
                else:
                    print(f'{self.get_name()}: collision detected, reverted..')
                    target_arm_config = self.prev_safe_config

                self.driver.set_target(target_arm_config)
            else:
                self.prev_safe_config = target_drivers
                self.driver.set_target(target_drivers)

        robot_model.setConfig(save_config)

    @include_method
    def set_freedrive(self, freedrive_mode: bool):
        # This behaves a bit differently... hm.
        with self.control_lock:
            if freedrive_mode:
                self.control_mode = ControlMode.FREEDRIVE
            else:
                self.control_mode = ControlMode.NONE
            self.target = None
            self.controller_params = {}

    @include_method
    def get_EE_wrench(self) -> List[float]:
        """Get the EE wrench, in the "world aligned EE" frame.

        (Force is measured as if it was at the end effector, but expressed in the XYZ coordinates of the robot base frame.)
        """
        return self.measured_EE_wrench

    @include_method
    def to_dict(self):
        with self.control_lock:
            ret = super().to_dict()
            ret['ddq'] = self.measured_accel
            ret['EE_wrench'] = self.measured_EE_wrench
        return ret
