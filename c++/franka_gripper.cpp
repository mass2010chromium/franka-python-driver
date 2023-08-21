#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <cmath>
#include <mutex>
#include <thread>
#include <chrono>
#include <atomic>

#include <stdio.h>
#include <fcntl.h>

#include <franka/exception.h>
#include <franka/gripper.h>
#include <franka/gripper_state.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

class FrankaGripper {
    public:
        FrankaGripper(std::string robot_ip);

        void start_control_thread();   // note: will move gripper -- home
        void shutdown_control();

        void move(double width, double speed, double force);

        franka::GripperState get_state();

    private:
        std::mutex cmd_lock;
        std::mutex read_lock;
        franka::Gripper gripper;
        volatile bool startup;
        volatile bool shutdown;
        volatile bool thread2_stop;
        franka::GripperState state;
        double target;
        double close_force;
        double max_speed;
};

FrankaGripper::FrankaGripper(std::string robot_ip) : gripper(robot_ip), startup(false) {}

void FrankaGripper::move(double width, double speed, double force) {
    cmd_lock.lock();
    target = width;
    close_force = force;
    max_speed = speed;
    cmd_lock.unlock();
}

franka::GripperState FrankaGripper::get_state() {
    read_lock.lock();
    auto ret = state;
    read_lock.unlock();
    return ret;
}

// startup isn't thread safe.
void FrankaGripper::start_control_thread() {
    if (startup) { return; }
    startup = true;
    shutdown = false;
    thread2_stop = false;
    gripper.homing();
    state = gripper.readOnce();
    target = state.width;
    std::thread read_thread([&]() {
        while (startup) {
            // lol
            auto _state = gripper.readOnce();
            read_lock.lock();
            state = _state;
            read_lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        // NOTE: hahahahahahahahaha
        while (!thread2_stop) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        delete this;
    });
    read_thread.detach();
    std::thread write_thread([&]() {
        double old_target = target;
        while (startup) {
            cmd_lock.lock();
            double _target = target;
            double speed = max_speed;
            double force = close_force;
            cmd_lock.unlock();
            if (old_target != _target) {
                if (state.width > _target) {
                    gripper.grasp(_target, speed, force, 0.4, 0.4);
                }
                else {
                    gripper.move(_target, speed);
                }
                old_target = _target;
            }
            else { std::this_thread::sleep_for(std::chrono::milliseconds(20)); }
        }
        thread2_stop = true;
    });
    write_thread.detach();
}

// not thread safe.
void FrankaGripper::shutdown_control() {
    if (shutdown) { return; }
    shutdown = true;
    if (startup) {
        startup = false;
    }
    else {
        delete this;
    }
}

typedef struct {
    PyObject_HEAD
    /* Type-specific fields go here. */
    FrankaGripper* controller;   // broken out in various methods
    std::mutex startup_lock;
} FrankaGripperObject;

static void
FrankaGripper_dealloc(FrankaGripperObject* self) {
    self->startup_lock.lock();
    self->controller->shutdown_control();
    self->startup_lock.unlock();
    Py_TYPE(self)->tp_free((PyObject*) self);
}

static PyObject*
FrankaGripper_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    static char* kwlist[] = {
        "host", NULL
    };
    PyObject* _host;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "U", kwlist, &_host)){
        return NULL;
    }
    Py_ssize_t size;
    const char* host = PyUnicode_AsUTF8AndSize(_host, &size);
    if (host == NULL) {
        return NULL;
    }

    FrankaGripperObject* self = (FrankaGripperObject*) type->tp_alloc(type, 0);

    FrankaGripper* controller = new FrankaGripper(std::string(host));

    self->controller = controller;
    return (PyObject*) self;
}

static int
FrankaGripper_init(FrankaGripperObject* self, PyObject* args, PyObject* kwds) {
    return 0;
}

static PyObject*
FrankaGripper_start(FrankaGripperObject* self, PyObject* /*args*/) {
    self->startup_lock.lock();
    self->controller->start_control_thread();
    self->startup_lock.unlock();
    Py_RETURN_NONE;
}

static PyObject*
FrankaGripper_get_state(FrankaGripperObject* self, PyObject* /*args*/) {
    // NOTE: uses lock internally
    franka::GripperState state = self->controller->get_state();
    PyObject* ret = PyDict_New();
    if (!ret) { return NULL; }

    PyObject* q = PyFloat_FromDouble(state.width);
    if (!q) { Py_DECREF(ret); return NULL; }
    PyObject* qmax = PyFloat_FromDouble(state.max_width);
    if (!qmax) { Py_DECREF(q); Py_DECREF(ret); return NULL; }

    if (PyDict_SetItemString(ret, "q", q) == -1
            || PyDict_SetItemString(ret, "qmax", qmax) == -1) {
        Py_DECREF(ret); ret = NULL;
    }
    Py_DECREF(qmax);
    Py_DECREF(q);
    return ret;
}

static PyObject*
FrankaGripper_move(FrankaGripperObject* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 3) {
        PyErr_SetString(PyExc_TypeError, "Wrong Number of arguments (expected [target, speed, force])");
        return NULL;
    }
    if (!(PyFloat_Check(args[0]) || PyLong_Check(args[0]))) { PyErr_SetString(PyExc_TypeError, "target expected number"); return NULL; }
    if (!(PyFloat_Check(args[1]) || PyLong_Check(args[1]))) { PyErr_SetString(PyExc_TypeError, "speed expected number"); return NULL; }
    if (!(PyFloat_Check(args[2]) || PyLong_Check(args[2]))) { PyErr_SetString(PyExc_TypeError, "force expected number"); return NULL; }

    self->controller->move(PyFloat_AsDouble(args[0]), PyFloat_AsDouble(args[1]), PyFloat_AsDouble(args[2]));
    Py_RETURN_NONE;
}

static PyMethodDef FrankaGripper_methods[] = {
    {"start", (PyCFunction) FrankaGripper_start, METH_NOARGS,
            PyDoc_STR("Start the franka gripper driver. Begins communication with the gripper ")},
    {"get_state", (PyCFunction) FrankaGripper_get_state, METH_NOARGS,
            PyDoc_STR("Get the state (q, qmax) of the franka gripper.")},
    {"move", (PyCFunction) FrankaGripper_move, METH_FASTCALL,
            PyDoc_STR("Set the target gripper width. The driver will try to move towards it.")},
    {NULL}  /* Sentinel */
};

static PyMemberDef FrankaGripper_members[] = {
    {NULL}  /* Sentinel */
};

static PyGetSetDef FrankaGripper_getsetters[] = {
    {NULL}  /* Sentinel */
};

static PyTypeObject FrankaGripperType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "franka_motion.FrankaGripper",
    .tp_basicsize = sizeof(FrankaGripperObject),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor) FrankaGripper_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = PyDoc_STR("Franka Gripper, implemented in C++"),
    .tp_methods = FrankaGripper_methods,
    .tp_members = FrankaGripper_members,
    .tp_getset = FrankaGripper_getsetters,
    .tp_init = (initproc) FrankaGripper_init,
    .tp_new = (newfunc) FrankaGripper_new,
};

