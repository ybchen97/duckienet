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
from dwa import dwa
from dwa import Config
from shutil import copyfile
import glob
import os
import cv2

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

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--map-name', default='udem1')
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
parser.add_argument('--seed', default=1, type=int, help='seed')
parser.add_argument('--save', action='store_true', help='Saves datasets.')
parser.add_argument('--image_dir', default="./data/images/", type=str, help='Directory to save images.')
parser.add_argument('--action_dir', default="./data/actions/", type=str, help='Directory to save actions.')
parser.add_argument('--intent_dir', default="./data/intentions/", type=str, help='Directory to save intentions.')
parser.add_argument('--counter-start', default=0, type=int, help='Saved image filenames start from this number')
args = parser.parse_args()

env = DuckietownEnv(
    seed = args.seed,
    map_name = args.map_name,
    draw_curve = args.draw_curve,
    draw_bbox = args.draw_bbox,
    domain_rand = args.domain_rand,
    frame_skip = args.frame_skip,
    distortion = args.distortion,
)

# SET THIS
steps_before_plan = 20
plan_threshold = 25
plan_counter = steps_before_plan
intention = None
config = Config(env)

print(args)
if args.save:
    # sets variables
    save = args.save
    image_dir = args.image_dir
    action_dir = args.action_dir
    intent_dir = args.intent_dir
    counter = args.counter_start
    # Ensures image and action directory are specified
    assert image_dir is not None
    assert action_dir is not None
    assert intent_dir is not None

    print(f"Save is turned on. Images will be saved in {image_dir}. Actions will be saved in {action_dir}. Intentions will be saved in {intent_dir}.")
    print(f"Counter starts from {counter}.")

env.reset()
env.render()
top_down = False

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """
    global top_down

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render()
        top_down = False
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        cv2.destroyAllWindows()
        sys.exit(0)
    elif symbol == key.SPACE:
        top_down = not top_down
        if top_down:
            env.render(mode='top_down')
        else:
            env.render()

    # Take a screenshot
    # UNCOMMENT IF NEEDED - Skimage dependency
    # elif symbol == key.RETURN:
    #     print('saving screenshot')
    #     img = env.render('rgb_array')
    #     save_img('screenshot.png', img)

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

def did_move():
    action = env.get_agent_info()['Simulator']['action']
    return action[0] != 0.0 or action[1] != 0.0

def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    
    global plan_counter, intention
    planned = plan_counter == steps_before_plan
    moved = did_move()
    if planned:
        plan_counter = 0
        intention = dwa(env, config, plan_threshold=plan_threshold)
        # intention.show()

        img = cv2.cvtColor(np.array(intention), cv2.COLOR_RGB2BGR)
        cv2.imshow("intention", img)
        cv2.waitKey(1)
    elif moved: # only increase plan counter if you move
        plan_counter += 1

    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action = np.array([0.44, 0.0])
    if key_handler[key.DOWN]:
        action = np.array([-0.44, 0])
    if key_handler[key.LEFT]:
        action = np.array([0.15, +1])
    if key_handler[key.RIGHT]:
        action = np.array([0.15, -1])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    obs, reward, done, info, loss, done_code = env.step(action)
    # print('step_count = %s, reward=%.3f, loss=%i' % (env.unwrapped.step_count, reward, loss))

    if key_handler[key.RETURN]:
        # TODELETE: print('key.RETURN pressed!')
        im = Image.fromarray(obs)

        im.save('screen.png')

    if done:
        print(f'Done:{done_code}')
        env.reset()
    
    if top_down:
        env.render(mode = 'top_down')
    else:
        env.render()

    if args.save and moved:
        global counter 
        print(f'counter:{counter}')

        X = Image.fromarray(obs)

        # from random import randint
        image_filename = f'X_{counter}.png'
        action_filename = f'Y_{counter}.npy'
        intention_filename=f'I_{counter}.png'
        # im = im.resize(size = (224, 224))
        X.save(image_dir + image_filename)
        np.save(action_dir + action_filename, action)
        intention.save(intent_dir + intention_filename)
        counter += 1

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
