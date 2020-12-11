#! /usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

# This ROS Node converts Joystick inputs from the joy node
# into commands for turtlesim or any other robot

MAX_X = 5
MAX_Z = 5

# joy stick mappings
MODEL='wired'
MAPPING={}
# wireless model
MAPPING['wireless'] = {
        # axes
        'axes':{
        'left_stick_lr' : 0,
        'left_stick_ud' : 1,
        'lt' : 2,
        'right_stick_lr' : 3,
        'right_stick_ud' : 4,
        'rt' : 5,
        'left_button_lr' : 6,
        'left_button_ud' : 7,
        },
        # button
        'buttons':{
        'A' : 0,
        'B' : 1,
        'X' : 2,
        'Y' : 3,
        'lb' : 4,
        'rb' : 5,
        'back' : 6,
        'start' : 7,
        'unkown' : 8,
        'left_stick_btn' : 9,
        'right_stick_btn' : 10
        }
}

# wired_model model
MAPPING['wired'] = {
        # axes
        'axes':{
        'left_stick_lr' : 0,
        'left_stick_ud' : 1,
        'right_stick_lr' : 2,
        'right_stick_ud' : 3,
        'left_button_lr' : 4,
        'left_button_ud' : 5,
        },
        # button
        'buttons':{
        'A' : 1,
        'B' : 2,
        'X' : 0,
        'Y' : 3,
        'lb' : 4,
        'rb' : 5,
        'lt' : 6,
        'rt' : 7,
        'back' : 8,
        'start' : 9,
        'left_stick_btn' : 10,
        'right_stick_btn' : 11
        }
}

JOY_MAPPING=MAPPING[MODEL]

# Receives joystick messages (subscribed to Joy topic)
# then converts the joysick inputs into Twist commands
# axis 1 aka left stick vertical controls linear speed
# axis 0 aka left stick horizonal controls angular speed
def callback(data):
    global MAX_X, MAX_Z
    twist = Twist()
    twist.linear.x = MAX_X*data.axes[JOY_MAPPING['axes']['left_stick_ud']]
    twist.angular.z = MAX_Z*data.axes[JOY_MAPPING['axes']['left_stick_lr']]
    # reset max_x, max_y
    MAX_X = MAX_X*(1-data.buttons[JOY_MAPPING['buttons']['back']])
    MAX_Z = MAX_Z*(1-data.buttons[JOY_MAPPING['buttons']['back']])
    MAX_X += data.buttons[JOY_MAPPING['buttons']['lb']]
    MAX_Z += data.buttons[JOY_MAPPING['buttons']['rb']]
    pub.publish(twist)

# Intializes everything
def start():
    # publishing to "turtle1/cmd_vel" to control turtle1
    global pub
    pub = rospy.Publisher('turtle1/cmd_vel', Twist, queue_size=1)
    # subscribed to joystick inputs on topic "joy"
    rospy.Subscriber("joy", Joy, callback)
    # starts the node
    rospy.init_node('Joy2Turtle')
    rospy.spin()

if __name__ == '__main__':
    start()
