#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""

import sys
import argparse
import pyglet
from pyglet.window import key
import numpy as np
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper
from gym_duckietown.simulator import ROBOT_LENGTH, ROBOT_WIDTH
from PIL import Image
from shutil import copyfile
import glob
import os

# from experiments.utils import save_img

"""

Mobile robot motion planning sample with Dynamic Window Approach

author: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı

"""

import math
from enum import Enum

# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt, transforms
import numpy as np


def dwa_control(x, config, goal, ob):
    """
    Dynamic Window Approach control
    """
    dw = calc_dynamic_window(x, config)

    u, trajectory = calc_control_and_trajectory(x, dw, config, goal, ob)

    return u, trajectory


class RobotType(Enum):
    circle = 0
    rectangle = 1


class Config:
    """
    simulation parameter class
    """

    def __init__(self, env):
        # robot parameter
        self.max_speed = 0.44  # [m/s]
        self.min_speed = 0  # [m/s]
        self.max_yaw_rate = 40.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 0.2  # [m/ss]
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/ss]
        self.v_resolution = 0.01  # [m/s]
        self.yaw_rate_resolution = 0.1 * math.pi / 180.0  # [rad/s]
        self.dt = 0.1  # [s] Time tick for motion prediction
        self.predict_time = 0.1  # [s]
        self.to_goal_cost_gain = 10
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 0.2
        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked
        self.robot_type = RobotType.circle

        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.robot_radius = 0.01 # [m] for collision check # hardcoded for duckietown
        # self.robot_radius = (ROBOT_WIDTH + ROBOT_LENGTH) / 4

        # if robot_type == RobotType.rectangle
        self.robot_width = ROBOT_WIDTH  # [m] for collision check
        self.robot_length = ROBOT_LENGTH  # [m] for collision check
        self.ob = self.set_ob(env)
    
    def set_ob(self, env):
        # obstacles [ [(x, y), (x, y) ], ...] assumes obstacles are lines
        ob_intervals = []

        # TODO: toggle based on map name
        if "udem1" in env.map_name:
            ob_intervals = [
                [[1,-1], [7,-1]], # outside border
                [[7,-1], [7,-5]],
                [[7,-5], [6,-5]],
                [[6,-5], [6,-6]],
                [[6,-6], [1,-6]],
                [[1,-6], [1,-1]],
                [[2,-2], [3,-2]], # top-left tile
                [[3,-2], [3,-3]],
                [[3,-3], [2,-3]],
                [[2,-3], [2,-2]],
                [[2,-4], [3,-4]], # bot-left tile
                [[3,-4], [3,-5]],
                [[3,-5], [2,-5]],
                [[2,-5], [2,-4]],
                [[4,-2], [6,-2]], # Right weird shape
                [[6,-2], [6,-4]],
                [[6,-4], [5,-4]],
                [[5,-4], [5,-5]],
                [[5,-5], [4,-5]],
                [[4,-5], [4,-2]]
            ]
        elif "straight_road" in env.map_name:
            ob_intervals = [
                [[0,0], [35, 0]],
                [[0,-1], [35, -1]]
            ]
        else: 
            for tile in env.obstacle_tiles: 
                # reflect about x-axis to fit duckietown's map
                # e.g. duckietown's (2,3) coordinate is actually (2, -3)
                [x, y] = tile['coords']
                # (x, -y) refer to starting coordinate of tile. tile is 1x1 grid from (x, -y).
                top_left = [x, -y]
                top_right = [x + env.road_tile_size, -y]
                bottom_left = [x, -y-env.road_tile_size]
                bottom_right = [x + env.road_tile_size, -y-env.road_tile_size]
                intervals = [
                    [top_left, top_right],
                    [top_left, bottom_left],
                    [bottom_left, bottom_right],
                    [top_right, bottom_right],
                ]
                ob_intervals.extend(intervals)
        return np.array(ob_intervals)

    @property
    def robot_type(self):
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value):
        if not isinstance(value, RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value

def motion(x, u, dt):
    """
    motion model
    """

    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]

    return x

def calc_dynamic_window(x, config):
    """
    calculation dynamic window based on current state x
    """

    # Dynamic window from robot specification
    Vs = [config.min_speed, config.max_speed,
          -config.max_yaw_rate, config.max_yaw_rate]

    # Dynamic window from motion model
    Vd = [x[3] - config.max_accel * config.dt, 
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_delta_yaw_rate * config.dt,
          x[4] + config.max_delta_yaw_rate * config.dt]

    #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
    # print(f"vs[0]: {Vs[0]}")
    # print(f"vd[0]: {Vd[0]}")

    # print(f'Vd: {Vd}')
    # print(f'Vs: {Vs}')
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]
    # print(f"dw: {dw}")

    return dw

def predict_trajectory(x_init, v, y, config):
    """
    predict trajectory with an input
    """

    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt)
        trajectory = np.vstack((trajectory, x))
        time += config.dt

    return trajectory

def calc_control_and_trajectory(x, dw, config, goal, ob):
    """
    calculation final input with dynamic window
    """

    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])

    best_goal_cost = None
    best_speed_cost = None
    best_ob_cost = None

    # evaluate all trajectory with sampled input in dynamic window
    
    # Hardcoded for duckytown
    combinations = [
        [0, 0],
        [0.44, 0.0],
        [0.35, 1],
        [0.35, -1]
    ]

    # for v in [0.44, 0.66]:
    #     for y in [-1.5, -1, 0, 1, 1.5]:
    # for v in np.arange(dw[0], dw[1], config.v_resolution):
    #     for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):
    for combination in combinations:
        [v, y] = combination
        trajectory = predict_trajectory(x_init, v, y, config)
        # calc cost
        to_goal_cost = config.to_goal_cost_gain * euclidean_dist_cost(trajectory, goal)
        speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
        ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(trajectory, ob, config)

        # print(f'to_goal_cost: {to_goal_cost}, speed_cost: {speed_cost}, ob_cost: {ob_cost}')
        # assert False
        final_cost = to_goal_cost + speed_cost + ob_cost

        # search minimum trajectory
        if min_cost >= final_cost:
            min_cost = final_cost
            # print(f'to_goal_cost: {to_goal_cost}, speed_cost: {speed_cost}, ob_cost: {ob_cost}')    
            # print(f'min_cost: {min_cost}')
            best_goal_cost = to_goal_cost
            best_speed_cost = speed_cost
            best_ob_cost = ob_cost

            best_u = [v, y]
            best_trajectory = trajectory
            # if abs(best_u[0]) < config.robot_stuck_flag_cons \
            #         and abs(x[3]) < config.robot_stuck_flag_cons:
                # to ensure the robot do not get stuck in
                # best v=0 m/s (in front of an obstacle) and
                # best omega=0 rad/s (heading to the goal with
                # angle difference of 0)
                # Commented out for duckytown as I want it to not move when stuck.
                # best_u[1] = -config.max_delta_yaw_rate
    # print(f'best_u: {best_u}')
    # print(f'best_goal_cost: {best_goal_cost}, speed_cost: {best_speed_cost}, ob_cost: {best_ob_cost}')   
    # assert False
    return best_u, best_trajectory

def projection(p1, p2, p3):
    """ p3 is the point. p1 and p2 forms the line.
    """

    #distance between p1 and p2
    l2 = get_sld(p1, p2)
    if l2 == 0:
        assert False, 'p1 and p2 are the same points'

    #The line extending the segment is parameterized as p1 + t (p2 - p1).
    #The projection falls where t = [(p3-p1) . (p2-p1)] / |p2-p1|^2

    #if you need the point to project on line extention connecting p1 and p2
    # t = np.sum((p3 - p1) * (p2 - p1)) / l2

    #if you need to ignore if p3 does not project onto line segment
    # if t > 1 or t < 0:
    #     print('p3 does not project onto p1-p2 line segment')

    #if you need the point to project on line segment between p1 and p2 or closest point of the line segment
    t = max(0, min(1, np.sum((p3 - p1) * (p2 - p1)) / l2))

    projection = p1 + t * (p2 - p1)
    return projection

def get_sld(p1, p2):
    return np.sum((p1-p2)**2)

def calc_obstacle_cost(trajectory, ob, config):
    """
    calc obstacle cost inf: collision
    """
    r = []
    for o in ob:
        p1 = o[0]
        p2 = o[1]
        for t in trajectory:
            # print(f't: {t}')
            p3 = (t[0], t[1])
            if (np.array_equal(p1, p2)):
                r.append(get_sld(p1, p3))
            else:
                # print(f"p1: {p1}, p2: {p2}, p3: {p3}")
                p = projection(p1, p2, p3)
                # print(f"p:{p}")
                r.append(get_sld(p, p3))
                # print(f"p1: {p1}, p2: {p2}, p3: {p3}, r: {r}")
    r = np.array(r) # distance from obstacle
    # print(f'r: {r}')
    # assert False
    
    # ox = ob[:, 0]
    # oy = ob[:, 1]
    # print(f'trajectory: {trajectory}')
    # dx = trajectory[:, 0] - ox[:, None]
    # dy = trajectory[:, 1] - oy[:, None]
    # r = np.hypot(dx, dy)

    if config.robot_type == RobotType.rectangle:
        assert False, "Rectangle type not supported"
        yaw = trajectory[:, 2]
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        rot = np.transpose(rot, [2, 0, 1])
        local_ob = ob[:, None] - trajectory[:, 0:2]
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        local_ob = np.array([local_ob @ x for x in rot])
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        upper_check = local_ob[:, 0] <= config.robot_length / 2
        right_check = local_ob[:, 1] <= config.robot_width / 2
        bottom_check = local_ob[:, 0] >= -config.robot_length / 2
        left_check = local_ob[:, 1] >= -config.robot_width / 2
        if (np.logical_and(np.logical_and(upper_check, right_check),
                           np.logical_and(bottom_check, left_check))).any():
            return float("Inf")
    elif config.robot_type == RobotType.circle:
        if np.array(r <= config.robot_radius).any():
            return float("Inf")

    min_r = np.min(r)
    return 1.0 / min_r  # OK

def euclidean_dist_cost(trajectory, goal):
    dx = np.abs(goal[0] - trajectory[-1, 0])
    dy = np.abs(goal[1] - trajectory[-1, 1])
    cost = np.sqrt(np.power(dx,2) + np.power(dy,2))

    return cost

def calc_to_goal_cost(trajectory, goal):
    """
        calc to goal cost with angle difference
    """
    

    # relative_error_angle = math.atan2(dy, dx)
    # if goal[0] > trajectory[-1, 0] and goal[1] > trajectory[-1, 1]: # normal:
    #     error_angle = relative_error_angle
    # elif goal[0] < trajectory[-1, 0] and goal[1] > trajectory[-1, 1]: # 2nd:
    #     error_angle = math.pi - relative_error_angle
    # elif goal[0] < trajectory[-1, 0] and goal[1] < trajectory[-1, 1]: # 3rd:
    #     error_angle = math.pi + relative_error_angle
    # elif goal[0] > trajectory[-1, 0] and goal[1] < trajectory[-1, 1]:
    #     error_angle = 2 * math.pi - relative_error_angle
    # else:
    #     assert False, "Out of all quadrants"
    # cost = error_angle
    # cost = abs(error_angle - trajectory[-1, 2])
    
    # print(f'trajectory: {trajectory}')
    # print(f'goal: {goal}')
    # print(f'error_angle: {error_angle}')
    # print(f'diff: {error_angle - trajectory[-1, 2]}')
    dx = goal[0] - trajectory[-1, 0]
    # dy = trajectory[-1, 1] - goal[1]
    dy = goal[1] - trajectory[-1, 1]
    error_angle = math.atan2(dy, dx)
    # print(f'error_angle: {error_angle}')
    # assert False
    cost_angle = error_angle - trajectory[-1, 2]
    cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

    return cost

def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)

def plot_robot(x, y, yaw, config):  # pragma: no cover
    if config.robot_type == RobotType.rectangle:
        outline = np.array([[-config.robot_length / 2, config.robot_length / 2,
                             (config.robot_length / 2), -config.robot_length / 2,
                             -config.robot_length / 2],
                            [config.robot_width / 2, config.robot_width / 2,
                             - config.robot_width / 2, -config.robot_width / 2,
                             config.robot_width / 2]])
        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                         [-math.sin(yaw), math.cos(yaw)]])
        outline = (outline.T.dot(Rot1)).T
        outline[0, :] += x
        outline[1, :] += y
        plt.plot(np.array(outline[0, :]).flatten(),
                 np.array(outline[1, :]).flatten(), "-k")
    elif config.robot_type == RobotType.circle:
        circle = plt.Circle((x, y), config.robot_radius, color="b")
        plt.gcf().gca().add_artist(circle)
        out_x, out_y = (np.array([x, y]) +
                        np.array([np.cos(yaw), np.sin(yaw)]) * config.robot_radius)
        plt.plot([x, out_x], [y, out_y], "-k")

def plot_relative_to_agent(dist, xi):
    plt.xlim([xi[0] - dist, xi[0] + dist]) 
    plt.ylim([xi[1] - dist, xi[1] + dist])

def plot_initial_positions(xi, goal, ob, config):
    plt.plot(xi[0], xi[1], "xr")
    plt.plot(goal[0], goal[1], "xb")
    plt.plot(ob[:, 0], ob[:, 1], "ok")
    plot_robot(xi[0], xi[1], xi[2], config)
    plot_arrow(xi[0], xi[1], xi[2])
    plt.axis("equal")
    plt.grid(True)
    plt.pause(0.0001)

def dwa(env, config, robot_type=RobotType.circle, plan_threshold = 10, show_animation = True):
    """
    Dynamic Window Approach. Plots the intention image. If successful, trajectory will be shown. Else, only map will be shown.
    
    Credits: https://github.com/larrylawl/PythonRobotics/tree/master/PathPlanning/DynamicWindowApproach
    """

    print(__file__ + " start!!")
    info = env.get_agent_info()['Simulator']
    # print(f'Agent Info: {info}')
    px, _, pz = info['cur_pos'] 
    pz *= -1 # flip y-axis to fit duckytown's coordinates 
    yaw = info['cur_angle']
    v = info['robot_speed']
    # Formula for angular velocity: http://www.cs.columbia.edu/~allen/F17/NOTES/icckinematics.pdf
    Vl, Vr = info['wheel_velocities']
    l = env.wheel_dist
    w = (Vr - Vl) / l
    # initial state [x(m), y(m), yaw(rad), v(m/s), omega/angular velocity(rad/s)]
    x = np.array([px, pz, yaw, v, w])
    xi = np.array([px, pz, yaw, v, w]) # stores initial
    # print(f"[px, pz, yaw, v, w]: {x}")
    # goal position [x(m), y(m)]
    gx, _, gz = env.goal_pos
    goal = np.array([gx, -gz]) # GOAL

    # input [forward speed, yaw_rate]

    config.robot_type = robot_type
    trajectory = np.array(x)
    ob = config.ob

    # best_dist_to_goal = 99999
    # initial_dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
    for i in range(plan_threshold):
        u, predicted_trajectory = dwa_control(x, config, goal, ob)
        x = motion(x, u, config.dt)  # simulate robot
        trajectory = np.vstack((trajectory, x))  # store state history

        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")
            plt.plot(x[0], x[1], "xr") 
            plt.plot(goal[0], goal[1], "xb")
            for o in ob:
                p1 = o[0]
                p2 = o[1]
                if np.array_equal(p1, p2): 
                    print(p1)
                    plt.plot(p1[0], p1[1], "ok")
                else: plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "k")
            # plt.plot([1, 4], "k")
            # iterate through obstacles
            # if only one elt in it, it's a point
            # if pair, plot a line from the coordinates
            # plt.plot(ob[:, 0], ob[:, 1], "ok")
            plot_robot(x[0], x[1], x[2], config)
            plot_arrow(x[0], x[1], x[2])
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)

        # check reaching goal
        dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
        if dist_to_goal <= config.robot_radius + 0.5:
            print("Goal!!")
            break

    # if show_animation:
    #     plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")
    #     plt.pause(0.0001)
    # if args.save:
        # plt.figure(figsize=(1.12, 1.12)) # fixed size before plotting

    # if best_dist_to_goal < initial_dist_to_goal: # good plan saves trajectory
    plt.cla()
    plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")
    plot_initial_positions(xi, goal, ob, config)
    # else: # failed plan shows initial location. 
    #     plt.cla()
    #     plot_initial_positions(xi, goal, ob, config)
    
    plot_relative_to_agent(0.75, xi)
    
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(1.12, 1.12, forward = True) # inet size with dpi 200
    temp_path = 'temp.png'
    plt.savefig(temp_path, dpi = 200)

    # TODO: rotate internally with matplotlib instead
    im = Image.open(temp_path).convert('RGB')
    im = im.rotate(90 - (yaw * 180 / math.pi))
    os.remove(temp_path)
    return im
    # im.show()
    # im.save(intent_dir + f'I_{counter}.png')

# pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# # Enter main event loop
# pyglet.app.run()

# env.close()
