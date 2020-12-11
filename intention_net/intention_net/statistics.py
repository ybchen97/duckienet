import numpy as np
import logging
import rospy
from datetime import datetime
import time
from geometry_msgs.msg import Twist

def get_logging_fn(input_frame='NORMAL', mode='DLM'):
    return input_frame + '_' + mode + '_' + datetime.now().strftime('_%H_%M_%d_%m_%Y.log')

# statistics in stream
class OnlineStatistics(object):
    """
    Welford's algorithm computes the sample variance incrementally.
    """
    def __init__(self, iterable=None, ddof=1):
        self.reset(ddof)
        if iterable is not None:
            for datum in iterable:
                self.include(datum)

    def include(self, datum, is_acc=False):
        # acc stat
        if self.last_datum is not None:
            self.acc_n, self.tmp_acc_mean, self.acc_M2, self.acc_variance = self.update(self.acc_n, self.tmp_acc_mean, self.acc_M2, np.fabs(datum-self.last_datum))
        self.last_datum = datum
        self.n, self.tmp_mean, self.M2, self.variance = self.update(self.n, self.tmp_mean, self.M2, datum)
        # standard
    def update(self, n, mean, M2, datum):
        n += 1
        delta = datum - mean
        mean += delta/n
        M2 += delta * (datum-mean)
        if n == self.ddof:
            variance = M2 / n
        else:
            variance = M2 / (n - self.ddof)
        return (n, mean, M2, variance)

    def reset(self, ddof=1):
        self.ddof = ddof
        self.n = 0
        self.tmp_mean = 0.0
        self.M2 = 0.0
        self.variance = 0.0
        self.last_datum = None
        self.acc_n = 0
        self.tmp_acc_mean = 0.0
        self.acc_M2 = 0.0
        self.acc_variance = 0.0

    @property
    def std(self):
        return np.sqrt(self.variance)

    @property
    def mean(self):
        return self.tmp_mean

    @property
    def acc_std(self):
        return np.sqrt(self.acc_variance)

    @property
    def acc_mean(self):
        return self.tmp_acc_mean

class SmoothStatistics(object):
    def __init__(self, input_frame, mode):
        self.name = input_frame+'_'+mode
        self.stat_vel = OnlineStatistics()
        self.stat_ang = OnlineStatistics()
        self.initialize_logger(input_frame, mode)
        self.start_time = time.time()

    def initialize_logger(self, input_frame, mode):
        import os
        self.logger = logging.getLogger(self.name)
        hdlr = logging.FileHandler(os.path.join('log', get_logging_fn(input_frame, mode)))
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        self.logger.addHandler(hdlr)
        self.logger.setLevel(logging.INFO)

    def include(self, datum):
        if self.stat_vel.n == 1:
            self.start_time = time.time()
        self.stat_vel.include(datum[0])
        self.stat_ang.include(datum[1])

    def reset(self):
        self.stat_vel.reset()
        self.stat_ang.reset()

    def log(self):
        self.logger.info(self.str())

    def str(self):
        msg = '\n######################################################'
        msg += '\nStatistics Name: %s' % self.name
        msg += '\nVelocity: %f +- %f' % (self.stat_vel.mean, self.stat_vel.std)
        msg += '\nVelocity Acceleration: %f +- %f' % (self.stat_vel.acc_mean, self.stat_vel.acc_std)
        msg += '\nAngular: %f +- %f' % (self.stat_ang.mean, self.stat_ang.std)
        msg += '\nAngular Acceleration: %f +- %f' % (self.stat_ang.acc_mean, self.stat_ang.acc_std)
        msg += '\nSmoothness mean: %s' % repr(np.mean([self.stat_vel.acc_mean, self.stat_ang.acc_mean]))
        msg += '\nTime: %s' % repr(time.time() - self.start_time)
        msg += '\n######################################################'
        return msg

def parse_topic():

    stat = SmoothStatistics('result')
    START = time.time()
    def callback(msg):
        global stat
        stat.include([msg.linear.x, msg.angular.z])
        global START
        print (stat.str())
        print (time.time() - START)

    rospy.init_node('parse_topic')
    #rospy.Subscriber('/navigation_velocity_smoother/raw_cmd_vel', Twist, callback)
    rospy.Subscriber('/mobile_base/commands/velocity', Twist, callback)
    rospy.spin()

#if __name__ == '__main__':
    #main()
    #parse_topic()

