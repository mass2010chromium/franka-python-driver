#Overview

Files in this folder (and subfolders) are used for controlling the Franka Emika Panda 7DOF robot arms.

# WARNINGS

Franka is kinda silly and does not let us set the gravity vector on the robot side.
The workaround for this we are using is to do gravity compensation in "user space" (i.e. in our torque controller code).
This requires setting the torque limits high and makes the robot somewhat unsafe in operation, as well as producing "artifacts":
For example, the robot arm will "drop" downwards and outwards (away from the +Z direction of the robot) each time the internal controller is engaged.
This happens when the manual control button / manual stop button is pressed and released, as well as whenever a controller running on the PC is stopped.
*KEEP ARMS FREE OF OBSTRUCTIONS (especially humans / other fragile objects) when releasing control of the robot!*

## Python stuff

The main file is `franka_controller.py`, which is a Motion component that interfaces with the arms. It runs the controller in physical mode. (Requires python extension module)

The kinematic mode controller is in `kinematic_controller.py`. This is needed because the franka arm has one more DOF, thus is not fully specified by `set_EE_transform` calls; the kinematic controller handles the elbow position.

A utility file `ik_move_test.py` is provided with some sample code for control and visualization of the robot using Klampt. (Requires python extension module)

## C++ stuff

Since the primary API for talking to the arms is `libfranka`, a C++ library, our robot drivers are also written in C++. There are four C++ files in the `./c++` folder:

1) `franka_module.cpp`: Python extension module file.
2) `franka_freedrive.cpp`: Freedrive code + gravity auto-calibration.
3) `franka_driver.cpp`: Produces an executable that communicates using `stdin` and `stdout` piping and a custom protocol (defunct)
4) `franka_ros_controller.cpp`: Produces an executable that communicates using ROS (defunct)

## Requirements:

- `libfranka` must be installed: https://github.com/frankaemika/libfranka
    - Real-time kernel required: https://frankaemika.github.io/docs/installation\_linux.html
- `motionlib` headers are required: https://github.com/mass2010chromium/motion-c
- For ROS interface: tested on ROS noetic, more steps required
    - You need to configure a ros project with the right filenames under `~/catkin_ws` that is piggybacked off for building the ROS driver.
    - look at Makefile to see the janky details
- Make and a c++ compiler are required
    - Makefile was tested with g++ 9.4.0 on ubuntu 20.04

## Building

`make python` to build and install the python extension module.
`make freedrive` to build the standalone binary for running freedrive mode with gravity calibration.
`make clean` to remove built binaries/python libs (does not affect installed python module)

## Integrating code into Motion

`franka_controller.FrankaController` and `kinematic_controller.KinematicFrankaController` can be used as controller classes for the Motion module.

## Freedrive

`./bin/freedrive` can be used to freedrive the robot. It takes one command line argument (the robot's IP address), and auto calibrates the gravity vector, showing the calibration results before starting.

*NOTE: if the calibrated gravity vector does not match the expected gravity vector (can be indicated with a quick check by the calculated error, which should be ~1), DO NOT ATTEMPT TO DRIVE THE ROBOT! ALWAYS CHECK THE AUTOCALIBRATION RESULTS BEFORE PROCEEDING! It is possible that gravity vector calibration failed.* The script can be exited with ctrl-C before or after freedrive mode has been engaged.

## Keyboard drive

`ik_move_test.py` can be used to control the left or right arm of the Diphtheria robot directly using the keyboard. Keymappings are as follows:

```
Translation:
w/s: +/- X
a/d: +/- Y
q/e: +/- Z

Rotation:
j/l: left/right (+Z/-Z)
i/k: up/down (-Y/+Y)
o/u: twist cw / twist ccw (+X/-X)
```

The program takes up to one argument, either `left_limb` or `right_limb`, and activates the corresponding arm.

If no argument is supplied a kinematic simulation is run instead.
