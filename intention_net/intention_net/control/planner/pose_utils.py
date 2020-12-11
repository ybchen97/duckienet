import math
import tf_conversions
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, PoseWithCovariance, PoseWithCovarianceStamped

#def pose(m):
    #print isinstance(m, Odometry)
    #if isinstance(m, (Odometry, PoseWithCovarianceStamped)):
        #return m.pose.pose
    #elif isinstance(m, (PoseStamped, PoseWithCovariance)):
        #return m.pose
    #else:
        #raise ValueError(type(m))

def pose(m):
    if hasattr(m, 'position') and hasattr(m, 'orientation'):
        return m
    elif hasattr(m, 'pose'):
        return pose(m.pose)
    else:
        raise ValueError(m)

def toPoseStamped(m):
    p = pose(m)
    ps = PoseStamped()
    ps.pose = p
    return ps

def dist(m0, m1):
    p0 = pose(m0).position
    p1 = pose(m1).position
    d = math.sqrt((p1.x-p0.x)**2 + (p1.y-p0.y)**2)
    return d

def angle(m):
    k = tf_conversions.fromMsg(pose(m))
    r, p, y = k.M.GetRPY()
    #assert r == 0 and p == 0
    print ('r p y', r, p, y)
    return y

def angle_pose_pair(m0, m1):
    p0 = pose(m0).position
    p1 = pose(m1).position
    dx = p1.x - p0.x
    dy = p1.y - p0.y
    ans =  math.atan2(dy, dx)
    return ans

def norm_angle(a):
    while a > math.pi:
        a -= 2*math.pi
    while a < -math.pi:
        a += 2*math.pi
    return a

def angle_diff(m1, m0):
    "m1 - m0"
    r0 = angle(m0)
    r1 = angle(m1)
    return norm_angle(r1 - r0)

def add_bias_to_pose(src, bias):
    m0 = pose(src)
    bias = pose(bias)
    m0.position.x += bias.position.x
    m0.position.y += bias.position.y
    m0.position.z += bias.position.z
    return m0
