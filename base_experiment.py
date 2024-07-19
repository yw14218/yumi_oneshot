#!/usr/bin/env python3
from abc import ABC, abstractmethod

class YuMiExperiment(ABC):
    def __init__(self, dir, object, demo_waypoints, demo_head_rgb, demo_head_mask, demo_wrist_rgb, demo_wrist_mask):
        """
        Initialize an ExperimentData instance with the provided data.

        :param dir: Directory path containing the experiment data.
        :param object: Description or label for the experiment object.
        :param demo_waypoints: Array of waypoints for the experiment.
        :param demo_head_rgb: RGB image from the head camera.
        :param demo_head_mask: Mask image from the head camera.
        :param demo_wrist_rgb: RGB image from the wrist camera.
        :param demo_wrist_mask: Mask image from the wrist camera.
        """
        self.dir = dir
        self.object = object
        self.demo_waypoints = demo_waypoints
        self.demo_head_rgb = demo_head_rgb
        self.demo_head_mask = demo_head_mask
        self.demo_wrist_rgb = demo_wrist_rgb
        self.demo_wrist_mask = demo_wrist_mask

    @abstractmethod
    def replay(self, live_waypoints):
        raise NotImplementedError()



