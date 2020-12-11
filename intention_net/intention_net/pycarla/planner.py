"""
Intention planner for carla simulator
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from carla.planner.planner import Planner
import cv2
from collections import deque
import numpy as np

class IntentionPlanner(Planner):
    def __init__(self, city_name, mode, radius=20.0):
        super().__init__(city_name)
        self.mode = mode
        self.offset = self._city_track._map.convert_to_pixel([radius, radius, 0.22])[0] - self._city_track._map.convert_to_pixel([0, 0, 0.22])[0]
        self.map = self._city_track._map.map_image.astype(np.uint8)
        self.map = cv2.copyMakeBorder(self.map, self.offset, self.offset, self.offset, self.offset, cv2.BORDER_CONSTANT, value=[255, 255, 255, 255])
        # remember history for better planning
        self.history = deque(maxlen=120)
        self.is_replan = False

    def get_next_command(self, source, source_ori, target, target_ori):
        print (source, source_ori, target, target_ori)
        print (self.mode)

        if self.mode == 'DLM':
            intention = super().get_next_command(source, source_ori, target, target_ori)
        else:
            track_source = self._city_track.project_node(source)
            track_target = self._city_track.project_node(target)
            if self._city_track.is_away_from_intersection(track_source) or self.current_route is None:
                route = self._city_track.compute_route(track_source, source_ori,
                                                       track_target, target_ori)
                self.current_route = route
                self.is_replan = True
            else:
                route = self.current_route
                self.is_replan = False
            map_route = [(int(pixel[0]+self.offset), int(pixel[1]+self.offset)) for pixel in [self._city_track._map.convert_to_pixel(node) for node in route]]
            s = self._city_track._map.convert_to_pixel(source)
            s = (s[0]+self.offset, s[1]+self.offset)
            self.history.append(s)
            intention = np.copy(self.map)
            if self.is_replan:
                cv2.line(intention, s, map_route[0], (255, 0, 0, 255), 10)
                for i in range(len(map_route)-1):
                    cv2.line(intention, map_route[i], map_route[i+1], (255, 0, 0, 255), 10)
            else:
                cv2.line(intention, s, map_route[3], (255, 0, 0, 255), 10)
                for i in range(3, len(map_route)-1):
                    cv2.line(intention, map_route[i], map_route[i+1], (255, 0, 0, 255), 10)

            for h in self.history:
                cv2.circle(intention, h, 2, (0,0,255,255), 5)
            source_pixel = self._city_track._map.convert_to_pixel(source)
            source_pixel = (int(source_pixel[0]), int(source_pixel[1]))
            intention = intention[source_pixel[1]:source_pixel[1]+2*self.offset, source_pixel[0]:source_pixel[0]+2*self.offset]
            theta = np.arctan2(source_ori[1], source_ori[0]) * 180 / np.pi
            col, row, channel = intention.shape
            M = cv2.getRotationMatrix2D((col/2, row/2), 90+theta, 1)
            intention = cv2.warpAffine(intention, M, (col, row), cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 1)
            intention = cv2.resize(intention, (224, 224))[:, :, :3]

        return intention

def test():
    p = IntentionPlanner('Town01', 'LPE_SIAMESE')
    #s, so, t, to = (191.0800018310547, 55.84000015258789, 0.22), (-1.0, 4.0531158447265625e-06, 0.0), (92.11000061035156, 30.820009231567383, 0.22), (-5.245208740234375e-06, -0.9999999403953552, 0.0)
    #s, so, t, to = (92.1099853515625, 227.22000122070312, 0.22), (-5.245208740234375e-06, -0.9999999403953552, 4.235164736271502e-22), (158.0800018310547, 27.18000030517578, 0.22), (-5.245208740234375e-06, -0.9999999403953552, 0.0)
    s, so, t, to = (92.08429718017578, 206.23475646972656, 0.22), (0.011081933975219727, -0.9999384880065918, -0.0001826928200898692), (158.0800018310547, 27.18000030517578, 0.22), (-5.245208740234375e-06, -0.9999999403953552, 0.0)
    intention = p.get_next_command(s, so, t, to)
    if p.mode.startswith('LPE'):
        import matplotlib.pyplot as plt
        plt.imshow(intention)
        plt.show()

#test()
