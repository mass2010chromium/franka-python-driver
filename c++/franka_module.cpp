#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <cmath>
#include <mutex>
#include <thread>
#include <chrono>
#include <atomic>
#include <algorithm>

#include <stdio.h>
#include <fcntl.h>

#include <franka/exception.h>
#include <franka/robot.h>

#include <franka/model.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#define motion_dtype double
#include <motionlib/utils.h>
#include <motionlib/se3.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/QR>

// lol
#include "franka_gripper.cpp"

// analytical IK mmmm
#include "franka_analytical_ik/franka_ik_He.hpp"

// Code for franka arm.
// TODO: merge with franka's enums -- if they support the 3 states we care about
enum FrankaExitStatus { NORMAL, PROTECTIVE_STOP, EMERGENCY_STOP };


// Stiffness
//const std::array<double, 7> K_DEFAULT_GAINS = {{300.0, 300.0, 300.0, 300.0, 125.0, 75.0, 25.0}};
//const std::array<double, 7> K_DEFAULT_GAINS = {{600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0}};
//const std::array<double, 7> K_DEFAULT_GAINS = {{150.0, 150.0, 150.0, 150.0, 62.5, 37.5, 12.5}};
//const std::array<double, 7> K_DEFAULT_GAINS = {{75.0, 75.0, 75.0, 75.0, 31.25, 18.75, 6.25}};
const std::array<double, 7> K_DEFAULT_GAINS = {{75.0, 75.0, 75.0, 75.0, 75.0, 50.0, 40.0}}; // N-m/rad
//const std::array<double, 7> K_DEFAULT_GAINS = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
// Damping
//const std::array<double, 7> D_DEFAULT_GAINS = {{50.0, 50.0, 50.0, 50.0, 25.0, 25.0, 15.0}};
//const std::array<double, 7> D_DEFAULT_GAINS = {{25.0, 25.0, 25.0, 25.0, 18, 18, 7.5}};
const std::array<double, 7> D_DEFAULT_GAINS = {{25.0, 25.0, 25.0, 25.0, 25, 12.5, 12.5}};   // N-m/(rad/s)
//const std::array<double, 7> D_DEFAULT_GAINS = {{5.0, 5.0, 5.0, 5.0, 3.0, 2.50, 1.50}};

const std::array<double, 7> DEFAULT_LIGHTEN_FACTOR = {{1, 1, 1, 1, 1, 1, 1}};   // UL

const std::array<double, 7> MAX_POWER_LIMIT = {{10, 10, 10, 8, 5, 3, 2}};  // watts
//const std::array<double, 7> MAX_TORQUE_LIMIT = {{70, 70, 70, 60, 50, 30, 20}};  // N-m
const std::array<double, 7> MAX_TORQUE_LIMIT = {{100, 110, 100, 55, 30, 25, 3}};  // N-m

// TODO: bubble up to teleop (depends on input dt)
const double DEFAULT_LOWPASS_ALPHA = 0.99;

class FrankaController {
    /**
    * The joint state command subscriber class for position control.
    */
    public:
        FrankaController(std::string robot_ip,
                         const std::array<double, 3>& gravity,
                         const std::array<double, 7>& impedance,
                         const std::array<double, 16>& tcp,
                         double payload);

        /**
         * Set up communcation
         */
        void initialize();

        /**
        * So this is error recovery but also does things like
        * set the impedance parameters
        */
        void setup_controller();

        FrankaExitStatus loop(); // TODO: error code?

        /**
         * Start a thread to control the robot. (bootloop the robot to reset pstop)
         */
        void start_control_thread();

        /**
         * Shuts down this controller (and its associated control thread).
         */
        void shutdown();

        const std::string robot_ip;

        /**
         * Get a copy of the robot state.
         */
        franka::RobotState get_state(std::array<double, 7>& ddq);

        /**
         * Get the robot's status.
         */
        double get_time_since_pstop();

        /**
         * Set the commanded config.
         */
        void set_target(const std::array<double, 7>& target_q);

        /**
         * Set the gains kp, kd.
         */
        void set_gains(const std::array<double, 7>& kp, const std::array<double, 7>& kd, const std::array<double, 7>& alpha);

        /**
         * Invoke blackbox model to compute gravity torques.
         */
        std::array<double, 7> get_gravity(const std::array<double, 7>& q, const std::array<double, 3>& tcp, double payload, const std::array<double, 3>& gravity_dir);

        /**
         * True if read state is valid; false otherwise.
         */
        bool state_valid;

        std::array<double, 7> get_kp() { return k_gains; };
        std::array<double, 7> get_kd() { return d_gains; };
        std::array<double, 7> get_ka() { return lighten_factor; };

    private:

        std::mutex cmd_lock;
        std::array<double, 7> commanded_q;
        std::array<double, 3> gravity_dir;
        std::array<double, 7> joint_impedance;
        std::array<double, 7> k_gains;          // kp for impedance control. diagonal matrix
        std::array<double, 7> d_gains;          // kd for impedance control. diagonal matrix
        std::array<double, 7> lighten_factor;   // amount to reduce the system's simulated parameters by
                                                //   (affects kp and kd; also affects mass matrix)
        std::array<double, 7> ddq;
        std::array<double, 16> tcp;
        std::array<double, 7> tau_d_calculated;
        double payload;
        double alpha;   // target q low-pass filter parameter
        std::atomic_bool running;
        std::atomic_bool startup;

        franka::Robot robot;
        franka::Model model;
        std::mutex state_lock;
        franka::RobotState state;
        std::chrono::time_point<std::chrono::steady_clock> last_pstop_time;
};

FrankaController::FrankaController(std::string robot_ip,
                                   const std::array<double, 3>& gravity,
                                   const std::array<double, 7>& impedance,
                                   const std::array<double, 16>& tcp,
                                   double payload) :
            robot_ip(robot_ip), robot(robot_ip), model(robot.loadModel()),
            gravity_dir(gravity), joint_impedance(impedance), tcp(tcp), payload(payload),
            running(false), startup(false), state_valid(false),
            k_gains(K_DEFAULT_GAINS), d_gains(D_DEFAULT_GAINS), lighten_factor(DEFAULT_LIGHTEN_FACTOR),
            alpha(DEFAULT_LOWPASS_ALPHA), last_pstop_time(std::chrono::steady_clock::now()) {
}

void FrankaController::initialize() {
    //initialize robot
    // TODO: read from params
    franka::RobotState initial_state = robot.readOnce();
    for (int i = 0; i < 7; ++i) {
        commanded_q[i] = initial_state.q[i];    // Set commanded position to be current position initially.
    }
    state = initial_state;
    state_valid = true;
}

void FrankaController::start_control_thread() {
    startup = true;
    std::thread control_thread([&]() {

        initialize();
        setup_controller();

        // init and loop is wrapped to recover from pstop
        while (startup) {
            FrankaExitStatus status = loop();    // doesn't return until exit is commanded, or error condition
            if (status == FrankaExitStatus::PROTECTIVE_STOP) {
                last_pstop_time = std::chrono::steady_clock::now();
                std::cout << "tau_d: [ ";
                for (int i = 0; i < 7; ++i) {
                    std::cout << tau_d_calculated[i] << ", ";
                }
                std::cout << "]" << std::endl;
                initialize();
                setup_controller();  // clears current errors, including pstop
                // TODO: Teleop integration required -- don't move EE target when pstopped
                continue;
            }
            break;
        }
        // NOTE: hahahahahahahahaha
        delete this;
        std::clog << "control thread exited" << std::endl;
    });
    control_thread.detach();
}

void FrankaController::shutdown() {
    if (startup) {
        startup = false;
        running = false;
    }
    else {
        delete this;
    }
}

/**
 * So this is error recovery but also does things like
 * set the impedance parameters
 */
void FrankaController::setup_controller() {
    //if (!get_vector(this->node_handle, name + "_joint_impedance", joint_impedance, 7)) {throw std::runtime_error("Franka joint impedance parameters not set!"); }

    try {
        //robot.setCollisionBehavior(collision_lower_torque,collision_upper_torque,collision_lower_force,collision_upper_force);
        robot.setCollisionBehavior({{1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0}},
                                   {{1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0}},
                                   {{1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0}},
                                   {{1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0}});
        robot.automaticErrorRecovery();
        running = true;
        robot.setJointImpedance(joint_impedance);

    } catch (const franka::Exception& e) {
        std::cerr << e.what() << std::endl;
        running = false;

        // Read debug messages
        size_t count = 0;
        robot.read([&count](const franka::RobotState& robot_state) {
            // Printing to std::cout adds a delay. This is acceptable for a read loop such as this, but
            // should not be done in a control loop.
            std::cerr << robot_state << std::endl;
            return count++ < 2;
        });
    }
}

franka::RobotState FrankaController::get_state(std::array<double, 7>& o_ddq) {
    franka::RobotState ret;
    state_lock.lock();
    ret = state;
    o_ddq = ddq;
    state_lock.unlock();
    return ret;
}

double FrankaController::get_time_since_pstop() {
    std::chrono::duration<double> diff = std::chrono::steady_clock::now() - last_pstop_time;
    return diff.count();
}

void FrankaController::set_target(const std::array<double, 7>& target_q) {
    cmd_lock.lock();
    commanded_q = target_q;
    cmd_lock.unlock();
}

void FrankaController::set_gains(const std::array<double, 7>& kp, const std::array<double, 7>& kd, const std::array<double, 7>& alpha) {
    cmd_lock.lock();
    k_gains = kp;
    d_gains = kd;
    lighten_factor = alpha;
    cmd_lock.unlock();
}

#define RADIANS(x) ((x)*M_PI/180.0)

const std::array<double, 7> joint_min = {{RADIANS(-166), RADIANS(-101), RADIANS(-166), RADIANS(-176), RADIANS(-166), RADIANS(-1), RADIANS(-166)}};  // rad
const std::array<double, 7> joint_max = {{RADIANS(166), RADIANS(101), RADIANS(166), RADIANS(-4), RADIANS(166), RADIANS(215), RADIANS(166)}};    // rad
const std::array<double, 7> MAX_V_LIMIT = {{3, 3, 3, 3, 3, 3, 3}};  // rad/s

// https://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

std::array<double, 7> get_barrier_torque(const std::array<double, 7>& joint_positions, const std::array<double, 7>& joint_velocities) {
    static const double BARRIER_MAX_TORQUE = 30;    // N-m
    static const double BARRIER_START = 0.1;        // radians
    static const double V_BARRIER_MAX_TORQUE = 30;  // N-m
    static const double V_BARRIER_START = 1.5;      // rad/s TODO: move to config
    std::array<double, 7> ret;
    for (size_t i = 0; i < 7; ++i) {
        ret[i] = 0;
        double violation = abs(joint_velocities[i]) - (MAX_V_LIMIT[i] - V_BARRIER_START);
        if (violation > 0) {
            double normalized_violation = violation / V_BARRIER_START;
            if (normalized_violation > 1) normalized_violation = 1;
            int dir = -sgn(joint_velocities[i]);
            ret[i] = dir * normalized_violation * normalized_violation * V_BARRIER_MAX_TORQUE;
        }
    }
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
    }
    return ret;
}

std::array<double, 7> FrankaController::get_gravity(const std::array<double, 7>& q, const std::array<double, 3>& tcp, double payload, const std::array<double, 3>& gravity_dir) {
    return model.gravity(q, payload, tcp, gravity_dir);
}

FrankaExitStatus FrankaController::loop() {
    //TODO: More control modes than position control? idk

    // TODO read from config / ????
    const double joint_speed_limit = 3;

    double time = 0;
    std::array<double, 7> cmd_q = commanded_q;      // low-pass filtered commanded q
    std::array<double, 7> cmd_dq = {{0, 0, 0, 0, 0, 0, 0}}; // estimated commanded dq
    Eigen::Map<const Eigen::Matrix4d> tcp_matrix(tcp.data());
    try {
        robot.control([&, &time, joint_speed_limit, &tcp_matrix, &cmd_q, &cmd_dq]
                (const franka::RobotState& robot_state, franka::Duration period) mutable
                -> franka::Torques {
            auto dt = period.toSec();
            time += dt;

            auto cmd_q_prev = cmd_q;
            cmd_lock.lock();
            for (int i = 0; i < 7; ++i) {
                cmd_q[i] = alpha*cmd_q[i] + (1-alpha)*commanded_q[i];
            }
            cmd_lock.unlock();

            auto cmd_dq_prev = cmd_dq;
            if (false and dt != 0) {
                for (int i = 0; i < 7; ++i) {
                    cmd_dq[i] = (cmd_q[i] - cmd_q_prev[i]) / dt;
                }
            }

            // Read current coriolis terms from model.
            std::array<double, 7> coriolis = model.coriolis(robot_state);
            std::array<double, 49> mass_data = model.mass(robot_state);
            Eigen::Map<const Eigen::Matrix<double, 7, 7>> mass(mass_data.data());
            std::array<double, 7> gravity_original = model.gravity(robot_state);
            //std::array<double, 7> gravity_mod = model.gravity(robot_state, gravity_dir);
            std::array<double, 7> gravity_mod = model.gravity(robot_state.q, payload, {tcp[12], tcp[13], tcp[14]}, gravity_dir);
            //std::cout << gravity_dir[0] << "," << gravity_dir[1] << "," << gravity_dir[2] << std::endl;

            auto prev_dq = state.dq;
            franka::RobotState _state = robot_state;
            for (size_t i = 0; i < 7; i++) {
                _state.tau_J[i] = robot_state.tau_ext_hat_filtered[i] + gravity_original[i] - gravity_mod[i];
                //_state.tau_J[i] = gravity_mod[i];
            }
            // in kEndEffector mode, second argument (stiffness frame) is ignored. First argument is desired position in EE frame.
            auto jac_data = model.zeroJacobian(franka::Frame::kEndEffector, robot_state.q, tcp, tcp);
            Eigen::Map<const Eigen::Matrix<double, 6, 7>> jac(jac_data.data());
            const double alpha = 0.0001;

            auto pinv = (jac * jac.transpose() + alpha * Eigen::Matrix<double, 6, 6>::Identity()).inverse() * jac;
            //auto pinv = jac.transpose().completeOrthogonalDecomposition().pseudoInverse();    no worko
            Eigen::Map<Eigen::Matrix<double, 6, 1>>(_state.O_F_ext_hat_K.data()) = -(pinv * Eigen::Map<Eigen::Matrix<double, 7, 1>>(_state.tau_J.data()));

            //Eigen::Map<Eigen::Matrix<double, 6, 1>>(_state.O_F_ext_hat_K.data()) = jac.transpose().bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(-Eigen::Map<Eigen::Matrix<double, 7, 1>>(_state.tau_J.data()));

            state_lock.lock();
            state = _state;
            for (size_t i = 0; i < 7; ++i) {
                ddq[i] = (robot_state.dq[i] - prev_dq[i]) / dt;
            }
            state_lock.unlock();

            // Compute torque command from joint impedance control law.
            // time step delay.
            std::array<double, 7> barrier_torque = get_barrier_torque(state.q, state.dq);

            for (size_t i = 0; i < 7; i++) {
                double beta = 1 - (1 / lighten_factor[i]);
                double delta = cmd_q[i] - robot_state.q[i];
                double tau_cmd = k_gains[i] * delta + d_gains[i] * (cmd_dq[i] - robot_state.dq[i]) + beta * _state.tau_J[i];
                if (tau_cmd * robot_state.dq[i] > MAX_POWER_LIMIT[i]) { tau_cmd = MAX_POWER_LIMIT[i] / robot_state.dq[i]; }
                double tau_d = tau_cmd + coriolis[i] - gravity_original[i] + gravity_mod[i] + barrier_torque[i];

                if (tau_d > MAX_TORQUE_LIMIT[i]) { tau_d = MAX_TORQUE_LIMIT[i]; }
                if (tau_d < -MAX_TORQUE_LIMIT[i]) { tau_d = -MAX_TORQUE_LIMIT[i]; }

                tau_d_calculated[i] = tau_d;
            }

            if (!running){
                std::clog << std::endl << "Finished motion, shutting down" << std::endl;
                return franka::MotionFinished(franka::Torques(tau_d_calculated));
            }
            // Send torque command.
            return tau_d_calculated;
        });
    } catch (const franka::ControlException& e) {
        running = false;
        std::cerr << "franka control exc " << e.what() << std::endl;
        return FrankaExitStatus::PROTECTIVE_STOP;
    }
    return FrankaExitStatus::NORMAL;
}

typedef struct {
    PyObject_HEAD
    /* Type-specific fields go here. */
    FrankaController* controller;   // broken out in various methods
    std::mutex startup_lock;
} FrankaDriverObject;

static void
FrankaDriver_dealloc(FrankaDriverObject* self) {
    self->startup_lock.lock();
    self->controller->shutdown();
    self->startup_lock.unlock();
    Py_TYPE(self)->tp_free((PyObject*) self);
}

static PyObject*
FrankaDriver_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    static char* kwlist[] = {
        "host", "gravity", "impedance", "tool_center", "payload", NULL
    };
    PyObject* _host;
    PyObject* _gravity;
    PyObject* _impedance;
    PyObject* _tcp = NULL;
    double payload = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "UOO|Od", kwlist,
                                     &_host, &_gravity, &_impedance, &_tcp, &payload)){
        return NULL;
    }
    Py_ssize_t size;
    const char* host = PyUnicode_AsUTF8AndSize(_host, &size);
    if (host == NULL) {
        return NULL;
    }

    FrankaDriverObject* self = (FrankaDriverObject*) type->tp_alloc(type, 0);

    std::array<double, 3> gravity;
    std::array<double, 7> impedance;
    std::array<double, 16> tcp = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

    if (_tcp != NULL) {
        double tcp_se3[12];
        if (parse_se3(tcp_se3, _tcp)) { return NULL; }
        for (size_t col = 0; col < 4; ++col) {
            for (size_t row = 0; row < 3; ++row) {
                tcp[row + col*4] = tcp_se3[row + col*3];
            }
        }
    }

    // iterable check.
#ifdef MOTION_DEBUG
    Py_ssize_t g_len = PyObject_Length(_gravity);
    Py_ssize_t i_len = PyObject_Length(_impedance);
    if (g_len < 0) { PyErr_SetString(PyExc_TypeError, "gravity has no length"); return NULL; }
    if (i_len < 0) { PyErr_SetString(PyExc_TypeError, "impedance has no length"); return NULL; }
    if (g_len != 3) { PyErr_SetString(PyExc_ValueError, "gravity vector len mismatch (expect: 3)"); return NULL; }
    if (i_len != 7) { PyErr_SetString(PyExc_ValueError, "impedance len mismatch (expect: 7)"); return NULL; }
#endif

    if (list_to_vector(_gravity, gravity.data())) {
        // shouldn't get here.. list conversion failure
        return NULL;
    }
    if (list_to_vector(_impedance, impedance.data())) {
        // shouldn't get here.. list conversion failure
        return NULL;
    }

    FrankaController* controller = new FrankaController(std::string(host), gravity, impedance, tcp, payload);

    self->controller = controller;
    return (PyObject*) self;
}

static int
FrankaDriver_init(FrankaDriverObject* self, PyObject* args, PyObject* kwds) {
    return 0;
}

static PyObject*
FrankaDriver_start(FrankaDriverObject* self, PyObject* /*args*/) {
    self->startup_lock.lock();
    self->controller->start_control_thread();
    self->startup_lock.unlock();
    Py_RETURN_NONE;
}

static PyObject*
FrankaDriver_get_state(FrankaDriverObject* self, PyObject* /*args*/) {
    // NOTE: uses lock internally
    std::array<double, 7> _ddq;
    franka::RobotState state = self->controller->get_state(_ddq);
    PyObject* ret = PyDict_New();
    if (!ret) { return NULL; }

    PyObject* q = vector_to_list(state.q.data(), 7);
    if (!q) { Py_DECREF(ret); return NULL; }
    PyObject* dq = vector_to_list(state.dq.data(), 7);
    if (!dq) { Py_DECREF(q); Py_DECREF(ret); return NULL; }
    PyObject* torques = vector_to_list(state.tau_J.data(), 7);
    if (!torques) { Py_DECREF(dq); Py_DECREF(q); Py_DECREF(ret); return NULL; }
    PyObject* EE_wrench = vector_to_list(state.O_F_ext_hat_K.data(), 6);
    if (!torques) { Py_DECREF(torques); Py_DECREF(dq); Py_DECREF(q); Py_DECREF(ret); return NULL; }
    PyObject* ddq = vector_to_list(_ddq.data(), 7);
    if (!ddq) { Py_DECREF(EE_wrench); Py_DECREF(torques); Py_DECREF(dq); Py_DECREF(q); Py_DECREF(ret); return NULL; }

    if (PyDict_SetItemString(ret, "q", q) == -1
            || PyDict_SetItemString(ret, "dq", dq) == -1
            || PyDict_SetItemString(ret, "tau_J", torques) == -1
            || PyDict_SetItemString(ret, "EE_wrench", EE_wrench) == -1
            || PyDict_SetItemString(ret, "ddq", ddq) == -1) {
        Py_DECREF(ret); ret = NULL;
    }
    Py_DECREF(ddq);
    Py_DECREF(EE_wrench);
    Py_DECREF(torques);
    Py_DECREF(dq);
    Py_DECREF(q);
    return ret;
}

static PyObject*
FrankaDriver_get_time_since_pstop(FrankaDriverObject* self, PyObject*) {
    double time_since_pstop = self->controller->get_time_since_pstop();
    return PyFloat_FromDouble(time_since_pstop);
}

static PyObject*
FrankaDriver_set_target(FrankaDriverObject* self, PyObject* _target_q) {
    std::array<double, 7> target_q;
    Py_ssize_t q_len = PyObject_Length(_target_q);
    if (q_len < 0) { PyErr_SetString(PyExc_TypeError, "config has no length"); return NULL; }
    if (q_len != 7) { PyErr_SetString(PyExc_ValueError, "config len mismatch (expect: 7)"); return NULL; }

    if (list_to_vector(_target_q, target_q.data())) {
        return NULL;
    }

    self->controller->set_target(target_q);
    Py_RETURN_NONE;
}

static PyObject*
FrankaDriver_set_gains(FrankaDriverObject* self, PyObject* const* args, Py_ssize_t nargsf, PyObject* kwnames) {
    Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);

    std::array<double, 7> kp, kd, alpha;
    kp = self->controller->get_kp();
    kd = self->controller->get_kd();
    alpha = self->controller->get_ka();
    std::array<double, 7>* targets[3] = {&kp, &kd, &alpha};

    if (nargs > 3) {
        PyErr_SetString(PyExc_TypeError, "Wrong Number of arguments (expected up to 3: kp, kd, alpha)");
        return NULL;
    }
    size_t i;
    for (i = 0; i < nargs; ++i) {
        Py_ssize_t n = PyObject_Length(args[i]);
        if (n < 0) {
            PyErr_SetString(PyExc_TypeError, "object has no length");
            return NULL;
        }
        if (n != 7) {
            PyErr_SetString(PyExc_ValueError, "Invalid size for vector (expected 7)");
            return NULL;
        }
        if (list_to_vector(args[i], targets[i]->data())) {
            return NULL;
        }
    }
    if (kwnames) {  // NULL if there are no keywords.
        for (size_t j = 0; j < PyTuple_Size(kwnames); ++j) {
            PyObject* keyword = PyTuple_GET_ITEM(kwnames, j);
            const char* kw_str = PyUnicode_AsUTF8(keyword);
            std::array<double, 7>* target;
            if (strcmp(kw_str, "kp") == 0) { target = &kp; }
            else if (strcmp(kw_str, "kd") == 0) { target = &kd; }
            else if (strcmp(kw_str, "alpha") == 0) { target = &alpha; }
            else {
                PyErr_SetString(PyExc_KeyError, "Invalid keyword argument (expected kp, kd, or alpha)");
                return NULL;
            }

            if (list_to_vector(args[i+j], target->data())) {
                return NULL;
            }
        }
    }

    //std::cout << kp[0] << ", " << kd[0] << ", " << alpha[0] << std::endl;
    self->controller->set_gains(kp, kd, alpha);
    Py_RETURN_NONE;
}

static PyObject*
FrankaDriver_get_gravity(FrankaDriverObject* self, PyObject* const* args, Py_ssize_t nargs) {

    if (nargs != 4) {
        PyErr_SetString(PyExc_TypeError, "Wrong Number of arguments (expected 4: q, tcp, payload, gravity_dir)");
        return NULL;
    }
    Py_ssize_t n;
    n = PyObject_Length(args[0]);
    if (n < 0) { PyErr_SetString(PyExc_TypeError, "config has no length"); return NULL; }
    if (n != 7) { PyErr_SetString(PyExc_ValueError, "Invalid size for config vector (expected 7)"); return NULL; }

    n = PyObject_Length(args[1]);
    if (n < 0) { PyErr_SetString(PyExc_TypeError, "tcp has no length"); return NULL; }
    if (n != 3) { PyErr_SetString(PyExc_ValueError, "Invalid size for tcp vector (expected 3)"); return NULL; }

    if (!(PyFloat_Check(args[2]) || PyLong_Check(args[2]))) {
        PyErr_SetString(PyExc_TypeError, "Expected number for args[2]");
        return NULL;
    }

    n = PyObject_Length(args[3]);
    if (n < 0) { PyErr_SetString(PyExc_TypeError, "gravity_dir has no length"); return NULL; }
    if (n != 3) { PyErr_SetString(PyExc_ValueError, "Invalid size for gravity_dir vector (expected 3)"); return NULL; }

    std::array<double, 7> q;
    std::array<double, 3> tcp, gravity_dir;
    if (list_to_vector(args[0], q.data()) ||
            list_to_vector(args[1], tcp.data()) ||
            list_to_vector(args[3], gravity_dir.data())) {
        return NULL;
    }
    double payload = PyFloat_AsDouble(args[2]);

    std::array<double, 7> res = self->controller->get_gravity(q, tcp, payload, gravity_dir);
    return vector_to_list(res.data(), 7);
}

static PyMethodDef FrankaDriver_methods[] = {
    {"start", (PyCFunction) FrankaDriver_start, METH_NOARGS,
            PyDoc_STR("Start the franka driver. Begins communication with the robot")},
    {"get_state", (PyCFunction) FrankaDriver_get_state, METH_NOARGS,
            PyDoc_STR("Get the state (config, vel, torque) of the franka driver.")},
    {"get_time_since_pstop", (PyCFunction) FrankaDriver_get_time_since_pstop, METH_NOARGS,
            PyDoc_STR("Get the time since the last pstop.")},
    {"set_target", (PyCFunction) FrankaDriver_set_target, METH_O,
            PyDoc_STR("Set the target joint config. The driver will try to move towards it.")},
    {"set_gains", (PyCFunction) FrankaDriver_set_gains, METH_FASTCALL | METH_KEYWORDS,
            PyDoc_STR("Set the gains (K, D, A), each a 7-vector.")},
    {"get_gravity", (PyCFunction) FrankaDriver_get_gravity, METH_FASTCALL,
            PyDoc_STR("Get gravity vector.")},
    {NULL}  /* Sentinel */
};

static PyObject*
FrankaDriver_get_host(FrankaDriverObject* self, void* /*closure*/) {
    return PyUnicode_FromString(self->controller->robot_ip.c_str());
}

static PyObject*
FrankaDriver_get_state_valid(FrankaDriverObject* self, void* /*closure*/) {
    if (self->controller->state_valid) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyMemberDef FrankaDriver_members[] = {
    {NULL}  /* Sentinel */
};

static PyGetSetDef FrankaDriver_getsetters[] = {
    {"host", (getter) FrankaDriver_get_host, (setter) NULL,
     "ip address of the franka arm", NULL},
    {"state_valid", (getter) FrankaDriver_get_state_valid, (setter) NULL,
     "Whether the driver's state readings are valid or not", NULL},
    {NULL}  /* Sentinel */
};

static PyTypeObject FrankaDriverType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "franka_motion.FrankaDriver",
    .tp_basicsize = sizeof(FrankaDriverObject),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor) FrankaDriver_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = PyDoc_STR("Franka Driver, implemented in C++"),
    .tp_methods = FrankaDriver_methods,
    .tp_members = FrankaDriver_members,
    .tp_getset = FrankaDriver_getsetters,
    .tp_init = (initproc) FrankaDriver_init,
    .tp_new = (newfunc) FrankaDriver_new,
};

static PyObject*
Franka_analytical_ik_all(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    /***
     * Perform analytical IK.
     * Returns all four IK solutions.
     *
     * Parameters:
     *      target:     SE3 transform
     *      q7:         Position of joint 7. must be specified to get 1 unique soln
     *      q_actual:   Current joint position. q_actual[0] is used for stabilizing singularity
     *
     * optional parameters (specify both, or none):
     *      q_min:      min joint angles.
     *      q_max:      max joint angles.
     *
     * Return 4-tuple of 4 length 7 lists (possibly containing NaN)
     */

    if (nargs != 3 && nargs != 5) {
        PyErr_SetString(PyExc_TypeError, "Wrong Number of arguments (expected 3 or 5: target, q7, q_actual, [q_min, q_max] )");
        return NULL;
    }
    std::array<double, 16> target_arr = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    double target_se3[12];
    if (parse_se3(target_se3, args[0])) { return NULL; }
    for (size_t col = 0; col < 4; ++col) {
        for (size_t row = 0; row < 3; ++row) {
            target_arr[row + col*4] = target_se3[row + col*3];
        }
    }

    if (!(PyFloat_Check(args[1]) || PyLong_Check(args[1]))) {
        PyErr_SetString(PyExc_TypeError, "Expected number for args[1] (q7)");
        return NULL;
    }
    double q7 = PyFloat_AsDouble(args[1]);

    std::array<double, 7> q_actual;
    std::array<double, 7> q_min;
    std::array<double, 7> q_max;

    Py_ssize_t n;
    n = PyObject_Length(args[2]);
    if (n < 0) { PyErr_SetString(PyExc_TypeError, "q_actual has no length"); return NULL; }
    if (n != 7) { PyErr_SetString(PyExc_ValueError, "Invalid size for q_actual vector (expected 7)"); return NULL; }
    if (list_to_vector(args[2], q_actual.data())) {
        // shouldn't get here.. list conversion failure
        return NULL;
    }
    if (nargs == 5) {
        n = PyObject_Length(args[3]);
        if (n < 0) { PyErr_SetString(PyExc_TypeError, "q_min has no length"); return NULL; }
        if (n != 7) { PyErr_SetString(PyExc_ValueError, "Invalid size for q_min vector (expected 7)"); return NULL; }
        if (list_to_vector(args[3], q_min.data())) {
            // shouldn't get here.. list conversion failure
            return NULL;
        }

        n = PyObject_Length(args[4]);
        if (n < 0) { PyErr_SetString(PyExc_TypeError, "q_max has no length"); return NULL; }
        if (n != 7) { PyErr_SetString(PyExc_ValueError, "Invalid size for q_max vector (expected 7)"); return NULL; }
        if (list_to_vector(args[4], q_max.data())) {
            // shouldn't get here.. list conversion failure
            return NULL;
        }
    }
    else {
        q_min = FRANKA_QMIN;
        q_max = FRANKA_QMAX;
    }

    std::array< std::array<double, 7>, 4 > res = franka_IK_EE(target_arr, q7, q_actual, q_min, q_max);
    
    PyObject* ret = PyList_New(4);
    if (ret == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* vec = vector_to_list(res[i].data(), 7);
        if (vec == NULL) {
            Py_DECREF(ret);
            return NULL;
        }
        PyList_SET_ITEM(ret, i, vec);
    }
    return ret;
}

static PyMethodDef frankaMethods[] = {
    {"analytical_ik_all", (PyCFunction) Franka_analytical_ik_all, METH_FASTCALL,
            PyDoc_STR("Compute analytical IK solution based on final joint angle.")},
    {NULL, NULL, 0, NULL}
};

static PyModuleDef franka_module = {
    PyModuleDef_HEAD_INIT,
    "franka_motion",
    NULL,   // Documentation
    -1,     /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
    frankaMethods
};

PyMODINIT_FUNC
PyInit_franka_motion(void) {
    PyObject *m;
    if (PyType_Ready(&FrankaDriverType) < 0)
        return NULL;
    if (PyType_Ready(&FrankaGripperType) < 0)
        return NULL;

    m = PyModule_Create(&franka_module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&FrankaDriverType);
    if (PyModule_AddObject(m, "FrankaDriver", (PyObject *) &FrankaDriverType) < 0) {
        Py_DECREF(&FrankaDriverType);
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&FrankaGripperType);
    if (PyModule_AddObject(m, "FrankaGripper", (PyObject *) &FrankaGripperType) < 0) {
        Py_DECREF(&FrankaGripperType);
        Py_DECREF(&FrankaDriverType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
