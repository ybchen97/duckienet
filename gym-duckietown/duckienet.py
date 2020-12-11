#!/usr/bin/env python
# manual

"""
Script to run IntentionNet together with BiSeNet.
Manual control is allowed too.
"""

import sys
import os
import argparse
import pyglet
from pyglet.window import key
import numpy as np
import gym
import gym_duckietown
import time
import cv2


from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper
from dwa import dwa
from dwa import Config
from PIL import Image


from duckienet_config import inet_cfg, bisenet_cfg

sys.path.append(inet_cfg['directory'])
from control.policy import Policy
from keras.preprocessing.image import load_img, img_to_array

# Not sure why but this import must be done after keras import...
import torch
import torch.nn as nn

sys.path.append(bisenet_cfg['directory'])
from lib.models import BiSeNetV2
import lib.transform_cv2 as T


parser = argparse.ArgumentParser()
parser.add_argument('--map-name', default='udem1')
parser.add_argument('--seg', action='store_true', help='Use bisenet if this flag is specified')
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
parser.add_argument('--seed', default=1, type=int, help='seed')
args = parser.parse_args()

# SET THIS
steps_before_plan = 20
plan_counter = steps_before_plan

env = DuckietownEnv(
    seed = args.seed,
    map_name = args.map_name,
    draw_curve = args.draw_curve,
    draw_bbox = args.draw_bbox,
    domain_rand = args.domain_rand,
    frame_skip = args.frame_skip,
    distortion = args.distortion,
)

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
        top_down = False
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.SPACE:
        top_down = not top_down
        if top_down:
            env.render(mode='top_down')
        else:
            env.render()
    elif symbol == key.ESCAPE:
        env.close()
        cv2.destroyAllWindows()
        event_loop.exit()

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

# Load inet model
num_intentions = 4
num_control = 2
inet = Policy(inet_cfg['intention_mode'], inet_cfg['input_frame'], num_control, inet_cfg['model_dir'], num_intentions)


if args.seg:
    torch.set_grad_enabled(False)

    # Load bisenet model
    bisenet = BiSeNetV2(bisenet_cfg['n_classes'])
    bisenet.load_state_dict(torch.load(bisenet_cfg['weights_path'], map_location='cpu'))
    bisenet.eval()
    bisenet.cuda()

    # prepare data (for bisenet)
    to_tensor = T.ToTensor(
        mean=(0.3257, 0.3690, 0.3223), # city, rgb
        std=(0.2112, 0.2148, 0.2115),
    )

next_action = np.array([0.0, 0.0])

def did_move():
    action = env.get_agent_info()['Simulator']['action']
    return action[0] != 0.0 or action[1] != 0.0

intention = None
config = Config(env)
list_cycles = []
image_frames = []

def update(dt):
    """    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    
    global next_action, intention, cycle_start

    # Get time taken for one inference cycle
    cycle_end = time.time()
    cycle = cycle_end - cycle_start
    cycle_start = cycle_end
    list_cycles.append(str(cycle * 1000))  # milliseconds

    action = next_action
     
    global plan_counter
    planned = plan_counter == steps_before_plan
    moved = did_move()
    if planned:
        list_cycles.append("Planning...")
        plan_counter = 0
        intention = dwa(env, config, plan_threshold=30)
        # intention.show()

        img = cv2.cvtColor(np.array(intention), cv2.COLOR_RGB2BGR)
        cv2.imshow("intention", img)
        cv2.waitKey(1)
    elif moved: # only increase plan counter if you move
        plan_counter += 1

    if key_handler[key.UP]:
        action = np.array([0.44, 0.0])
    if key_handler[key.DOWN]:
        action = np.array([-0.44, 0])
    if key_handler[key.LEFT]:
        action = np.array([0.35, +1])
    if key_handler[key.RIGHT]:
        action = np.array([0.35, -1])
    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    obs, reward, done, info, loss, done_code = env.step(action)
    # print('step_count = %s, reward=%.3f, loss=%i' % (env.unwrapped.step_count, reward, loss))
    image_frames.append(obs)

    ###########
    # BISENET #
    ###########
    if args.seg:
        # Generate segmentation labels from bisenet
        im = to_tensor(dict(im=obs, lb=None))['im'].unsqueeze(0).cuda()
        labels = bisenet(im)[0].argmax(dim=1).squeeze().detach().cpu().numpy()

        labels = labels.astype(np.uint8)
        labels = Image.fromarray(labels).resize(size=(224, 224))
        # labels.show()
        labels = img_to_array(labels)

        # INet only accepts RGB images i.e. input needs 3 channels
        # However, predicted labels only has a single channel
        # Hacky fix: duplicate along the single channel to get 3 input channels
        labels = np.repeat(labels, 3, axis=2)
    else:
        # Use image directly
        labels = img_to_array(Image.fromarray(obs).resize(size=(224, 224)))

    #################
    # INTENTION-NET #
    #################
    intention_array = img_to_array(intention)
    speed = np.array([0])

    # From labels, intention and speed, derive next action using IntentionNet
    next_action = inet.predict_control(labels, intention_array, speed, segmented=args.seg).squeeze()
    next_action[0] = min(0.25, next_action[0])
    print("Predicted control: ", next_action)

    if key_handler[key.RETURN]:
        im = Image.fromarray(obs)
        im.save('screen.png')

    if done:
        list_cycles.append("Done. Planning...")
        print(f'Done:{done_code}')
        success = False
        if done_code == 'finished':
            success = True
        end = time.time()
        time_taken = end - start
        objects_avoided = env.get_agent_info()['Simulator']['objects_avoided']
        log(success, reward, loss, time_taken, objects_avoided)
        env.reset()

        # Restart plan
        intention = dwa(env, config, plan_threshold=30)
        # intention.show()

        img = cv2.cvtColor(np.array(intention), cv2.COLOR_RGB2BGR)
        cv2.imshow("intention", img)
        cv2.waitKey(1)
        # event_loop.exit()

    if top_down:
        env.render(mode='top_down')
    else:
        env.render()

def log(success, reward, loss, time_taken, obstacles_avoided):
    import datetime, os
    # if os.path.exists("log.txt"):
    #     append_write = 'a'
    # else:
    #     append_write = 'w'
    text_file = open("log.txt", 'w')
    text_file.write(f'{datetime.datetime.now()}: success: {success}, reward: {reward}, loss: {loss}, time_taken: {time_taken}, obstacles_avoided: {obstacles_avoided}.\n')
    text_file.close()

pyglet.clock.schedule_interval(func=update, interval=1.0 / env.unwrapped.frame_rate)

# Enter main event loop
start = time.time()
cycle_start = time.time()
event_loop = pyglet.app.EventLoop()
event_loop.run()

env.close()

# Save list of cycles into txt file
# with open('list_cycles.txt','w') as f:
#     for item in list_cycles:
#         f.write(item + "\n")

# video_name = "recorded_instance"
# video = cv2.VideoWriter(f"{video_name}.avi", 0, 60, (640, 480))
# for img in image_frames:
#     video.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# cv2.destroyAllWindows()
# video.release()
# print(f"Video saved as {video_name}.")
