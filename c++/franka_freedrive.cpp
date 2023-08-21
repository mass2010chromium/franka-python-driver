// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <array>
#include <atomic>
#include <cmath>
#include <functional>
#include <iostream>
#include <iterator>
#include <mutex>
#include <thread>
#include <vector>

#include <math.h>   // acos, sin, cos
#include <stdlib.h> // rand
#include <unistd.h> // sleep

#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/model.h>
#include <franka/rate_limiting.h>
#include <franka/robot.h>

// NOTE: requires motion-c
#include <motionlib/so3.h>
#include <motionlib/vectorops.h>

namespace {
template <class T, size_t N>
std::ostream& operator<<(std::ostream& ostream, const std::array<T, N>& array) {
  ostream << "[";
  std::copy(array.cbegin(), array.cend() - 1, std::ostream_iterator<T>(ostream, ","));
  std::copy(array.cend() - 1, array.cend(), std::ostream_iterator<T>(ostream));
  ostream << "]";
  return ostream;
}
}  // anonymous namespace

std::array<double, 3> random_direction() {
    // https://math.stackexchange.com/questions/442418/random-generation-of-rotation-matrices
    double u1 = rand() / (double)RAND_MAX;
    double u2 = rand() / (double)RAND_MAX;
    double theta = acos(2*u1 - 1);
    double phi = M_PI * u2 * 2;
    return {
        sin(theta) * cos(phi),
        sin(theta) * sin(phi),
        cos(theta)
    };
}

#define RADIANS(x) ((x)*M_PI/180.0)

std::array<double, 7> joint_min = {{RADIANS(-166), RADIANS(-101), RADIANS(-166), RADIANS(-176), RADIANS(-166), RADIANS(-1), RADIANS(-166)}};
std::array<double, 7> joint_max = {{RADIANS(166), RADIANS(101), RADIANS(166), RADIANS(-4), RADIANS(166), RADIANS(215), RADIANS(166)}};

std::array<double, 7> get_barrier_torque(std::array<double, 7> joint_positions) {
    static const double BARRIER_MAX_TORQUE = 30;    // N-m
    static const double BARRIER_START = 0.1;        // radians
    std::array<double, 7> ret;
    for (size_t i = 0; i < 7; ++i) {
        const double min_violation = joint_positions[i] - (joint_min[i] + BARRIER_START);
        if (min_violation < 0) {
            double normalized_violation = min_violation / BARRIER_START;
            if (normalized_violation < -1) {
                normalized_violation = -1;
            }
            ret[i] = normalized_violation*normalized_violation * BARRIER_MAX_TORQUE;
            continue;
        }
        const double max_violation = joint_positions[i] - (joint_max[i] - BARRIER_START);
        if (max_violation > 0) {
            double normalized_violation = max_violation / BARRIER_START;
            if (normalized_violation > 1) {
                normalized_violation = 1;
            }
            ret[i] = normalized_violation*normalized_violation * -BARRIER_MAX_TORQUE;
            continue;
        }
        ret[i] = 0;
    }
    return ret;
}

/**
 * @example joint_impedance_control.cpp
 * An example showing a joint impedance type control that executes a Cartesian motion in the shape
 * of a circle. The example illustrates how to use the internal inverse kinematics to map a
 * Cartesian trajectory to joint space. The joint space target is tracked by an impedance control
 * that additionally compensates coriolis terms using the libfranka model library. This example
 * also serves to compare commanded vs. measured torques. The results are printed from a separate
 * thread to avoid blocking print functions in the real-time loop.
 */
int main(int argc, char** argv) {
  // Check whether the required arguments were passed.
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <robot-hostname>" << std::endl;
    return -1;
  }

  bool given_gravity;
  if (argc == 5) {
    given_gravity = true;
  }


  // Set print rate for comparing commanded vs. measured torques.
  const double print_rate = 2.0;

  // Initialize data fields for the print thread.
  struct {
    std::mutex mutex;
    bool has_data;
    std::array<double, 7> tau_d_last;
    franka::RobotState robot_state;
    std::array<double, 7> gravity;
    std::array<double, 7> gravity_orig;
    std::array<double, 7> coriolis;
  } print_data{};
  std::atomic_bool running{true};

  // Start print thread.
  std::thread print_thread([print_rate, &print_data, &running]() {
    while (running) {
      // Sleep to achieve the desired print rate.
      std::this_thread::sleep_for(
          std::chrono::milliseconds(static_cast<int>((1.0 / print_rate * 1000.0))));

      // Try to lock data to avoid read write collisions.
      if (print_data.mutex.try_lock()) {
        if (print_data.has_data) {
          std::array<double, 7> tau_error{};
          double error_rms(0.0);
          std::array<double, 7> tau_d_actual{};
          for (size_t i = 0; i < 7; ++i) {
            tau_d_actual[i] = print_data.tau_d_last[i] + print_data.gravity[i];
            tau_error[i] = tau_d_actual[i] - print_data.robot_state.tau_J[i];
            error_rms += std::pow(tau_error[i], 2.0) / tau_error.size();
          }
          error_rms = std::sqrt(error_rms);
          //std::array<double, 16>& measured_pose = robot_state.O_T_EE;
          //std::array<double, 16>& desired_pose = robot_state.O_T_EE_d;

          // Print data to console
          std::cout << "gravity:  " << print_data.gravity << std::endl
//                    << "coriolis: " << print_data.coriolis << std::endl
                    << "grav_orij: " << print_data.gravity_orig << std::endl
                    << "config:   " << print_data.robot_state.q << std::endl
//                    << "torque:   " << print_data.robot_state.tau_J << std::endl
//                    << "torque_prev:   " << print_data.robot_state.tau_J_d << std::endl
                    << "-----------------------" << std::endl;
//          std::cout << "tau_error [Nm]: " << tau_error << std::endl
//                    << "tau_commanded [Nm]: " << tau_d_actual << std::endl
//                    << "tau_measured [Nm]: " << print_data.robot_state.tau_J << std::endl
//                    << "root mean square of tau_error [Nm]: " << error_rms << std::endl
//                    << "-----------------------" << std::endl;
          print_data.has_data = false;
        }
        print_data.mutex.unlock();
      }
    }
  });

  const std::array<double, 7> impedances = {{150, 150, 150, 125, 125, 100, 100}};
  // Connect to robot.
  //franka::Robot robot(argv[1], franka::RealtimeConfig::kIgnore);
  franka::Robot robot(argv[1], franka::RealtimeConfig::kEnforce);
  try {

    franka::RobotState calib_state = robot.readOnce();
    std::cout << calib_state << std::endl;
    auto calib_torque = calib_state.tau_J;

    // Set additional parameters always before the control loop, NEVER in the control loop!
    // Set collision behavior.
    robot.setCollisionBehavior({{1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0}},
                               {{1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0}},
                               {{1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0}},
                               {{1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0}});
    robot.automaticErrorRecovery();
    //robot.setJointImpedance(impedances);
    //robot.setCartesianImpedance({{3000, 3000, 3000, 300, 300, 300}});
    //setDefaultBehavior(robot);

    // Load the kinematics and dynamics model.
    franka::Model model = robot.loadModel();

    // calibrate gravity vector.
    std::cout << "Calibrating gravity vector..." << std::endl;
    double gravity_amount = 9.81;
    std::array<double, 3> gravity_dir = {0.0, 0.0, -1.0};
    if (given_gravity) {
        gravity_dir[0] = std::stod(argv[2]);
        gravity_dir[1] = std::stod(argv[3]);
        gravity_dir[2] = std::stod(argv[4]);
    }
    else {
        double step_size = 0.5;    // radians
        int failed_attempts = 0;
        int fail_threshold = 500;
        double step_size_threshold = 0.001;     // stop optimizing when step size goes below this
        double step_size_multiplier = 0.5;      // geometric decrease
        int max_steps = 10;     // why not
        double best_error = 0;

        std::array<double, 7> gravity_torques = model.gravity(calib_state);
        double tmp[7];
        __vo_subv(tmp, gravity_torques.data(), calib_torque.data(), 7);
        best_error += __vo_normSquared(tmp, 7);

        std::cout << "initial error: " << best_error << std::endl;
        std::cout << "Measured: " << calib_torque << std::endl
                  << "Predicted: " << model.gravity(calib_state, gravity_dir) << std::endl;
        while (true) {
            std::array<double, 3> rotation_axis = random_direction();
            double rotation_matrix[9];
            std::array<double, 3> test_gravity;
            double tmp[7];
            double error = best_error;
            std::array<double, 7> best_torques;
            std::array<double, 3> best_gravity;
            bool found_better = false;
            for (int i = 1; i <= max_steps; ++i) {
                __so3_rotation(rotation_matrix, rotation_axis.data(), i*step_size);
                __so3_apply(test_gravity.data(), rotation_matrix, gravity_dir.data());
                __vo_mul(test_gravity.data(), test_gravity.data(), gravity_amount, 3);

                double tmp_error = 0;
                std::array<double, 7> gravity_torques;
                gravity_torques = model.gravity(calib_state, test_gravity);
                __vo_subv(tmp, gravity_torques.data(), calib_torque.data(), 7);
                tmp_error += __vo_normSquared(tmp, 7);
                if (tmp_error < best_error) {
                    found_better = true;
                    error = tmp_error;
                    best_torques = gravity_torques;
                    best_gravity = test_gravity;
                }
                else {
                    break;
                }
            }
            if (found_better) {
                failed_attempts = 0;
                best_error = error;
                __vo_unit(gravity_dir.data(), best_gravity.data(), 1e-5, 3);
                std::cout << "found better: " << best_error << std::endl;
                std::cout << best_torques << std::endl;
                continue;
            }
            if (step_size <= step_size_threshold) {
                break;
            }
            ++failed_attempts;
            if (failed_attempts == fail_threshold) {
                std::cout << "Decreasing step size..." << std::endl;
                step_size *= step_size_multiplier;
                failed_attempts = 0;
            }
        }
        std::cout << "Calibrated gravity vector." << std::endl;
        std::cout << "Error: " << best_error << std::endl;
    }

    __vo_mul(gravity_dir.data(), gravity_dir.data(), gravity_amount, 3);
    std::cout << "Check gravity vector: " << gravity_dir << std::endl;
    std::cout << "Measured: " << calib_torque << std::endl
              << "Predicted: " << model.gravity(calib_state, gravity_dir) << std::endl
              << "Press Enter to continue..." << std::endl;
    std::cin.ignore();

    // Set gains for the joint impedance control.
    // Stiffness
    //const std::array<double, 7> k_gains = {{600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0}};
    const std::array<double, 7> k_gains = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    // Damping
    //const std::array<double, 7> d_gains = {{50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0}};
    const std::array<double, 7> d_gains = {{5.0, 5.0, 5.0, 5.0, 3.0, 2.50, 1.50}};

    auto q_target = calib_state.q;

    double run_time = 120;
    double time = 0.0;

    // Define callback for the joint torque control loop.
    std::function<franka::Torques(const franka::RobotState&, franka::Duration)>
        impedance_control_callback =
            [&print_data, &model, &gravity_dir, &q_target, k_gains, d_gains](
                const franka::RobotState& state, franka::Duration /*period*/) -> franka::Torques {
      //std::cout << state.q_d << std::endl;
      // Read current coriolis terms from model.
      std::array<double, 7> coriolis = model.coriolis(state);
      std::array<double, 7> gravity_original = model.gravity(state);
      std::array<double, 7> gravity_mod = model.gravity(state, gravity_dir);

      // Compute torque command from joint impedance control law.
      // Note: The answer to our Cartesian pose inverse kinematics is always in state.q_d with one
      // time step delay.
      std::array<double, 7> tau_d_calculated;
      std::array<double, 7> barrier_torque = get_barrier_torque(state.q);

      for (size_t i = 0; i < 7; i++) {
        tau_d_calculated[i] =
            k_gains[i] * (q_target[i] - state.q[i]) - d_gains[i] * state.dq[i]
            /*+ coriolis[i]*/ - gravity_original[i] + gravity_mod[i] + barrier_torque[i];
      }
      q_target = state.q;

      // The following line is only necessary for printing the rate limited torque. As we activated
      // rate limiting for the control loop (activated by default), the torque would anyway be
      // adjusted!
      std::array<double, 7> tau_d_rate_limited = tau_d_calculated;
//      std::array<double, 7> tau_d_rate_limited =
//          franka::limitRate(franka::kMaxTorqueRate, tau_d_calculated, state.tau_J_d);

      // Update data to print.
      if (print_data.mutex.try_lock()) {
        print_data.has_data = true;
        print_data.robot_state = state;
        print_data.tau_d_last = tau_d_rate_limited;
        print_data.gravity = gravity_mod;
        print_data.gravity_orig = gravity_original;
        print_data.coriolis = coriolis;
        print_data.mutex.unlock();
      }

      // Send torque command.
      return tau_d_rate_limited;
    };

    // Start real-time control loop.
    robot.control(impedance_control_callback);
  } catch (const franka::Exception& ex) {
    running = false;
    std::cerr << ex.what() << std::endl;
  }

  if (print_thread.joinable()) {
    print_thread.join();
  }
  return 0;
}
