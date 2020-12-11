#INTETION
FORWARD='forward'
BACKWARD='backward'
STOP='stop'
LEFT='left'
RIGHT='right'

############################################################################################
# for intention planner
############################################################################################
NUM_INTENTION=100
AHEAD_DIST = 3
LOCAL_SHIFT = 2
TURNING_THRESHOLD = 0.7
# the size of the map
MAX_X, MAX_Y = 9, 9
WITH_PATH_TOPIC=False
#PATH_TOPIC='/move_base/DWAPlannerROS/global_plan'
TOLERANCE=0.0
#PATH_TOPIC='/move_base/NavfnROS/plan'
PATH_TOPIC='/global_planner/navfn_planner/plan'
POSE_TOPIC='/amcl_pose'
NAV_GOAL_TOPIC = '/move_base_simple/goal'
