#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from octomap_msgs.msg import Octomap
from ament_index_python.packages import get_package_share_path
import open3d as o3d
import numpy as np
import pyoctomap

class Octomap2Octree(Node):

    def __init__(self):
        super().__init__('octomap_saver')


        self.create_subscription(
            Octomap,
            'octomap_full',
            self.listener_callback,
            10)
        
        self.octree = None

    def listener_callback(self, msg):
        if self.octree is None:
            self.octree = o3d.geometry.Octree()
        else:
            self.octree.clear()

        data = np.array(msg.data, dtype=np.uint8)

        # Decode the octomap message to an octree
        self.octree.from_binary(data.tobytes())
        print(self.octree, "----------------\n\n")

    def convert_octomap_to_octree(octree):
        octree_array = np.frombuffer(octree.data, dtype=np.uint8)
        voxel_size = octree.resolution
        octree_depth = octree.tree_depth
        octree_leafs = octree.num_leafs
        octree_bitset_bytes = int(octree_size / 8)
        octree_bitset = np.unpackbits(octree_array[octree_bitset_bytes:])
        octree_bitset = np.reshape(octree_bitset, (octree_leafs, 8))
        octree_leafs = np.argwhere(octree_bitset[:, 0] == 1)
        octree_leafs = np.squeeze(octree_leafs)
        octree_centers = np.array([octree.keyToCoord(octree.key(i)) for i in octree_leafs], dtype=np.float32)
        octree_centers = octree_centers * voxel_size + voxel_size / 2
        octree_voxels = np.ones(len(octree_centers), dtype=np.int8)
        octree = o3d.geometry.Octree(voxel_size)
        octree.initialize(octree_centers, octree_voxels)
        return octree


def main():
    rclpy.init()

    octomap_saver = Octomap2Octree()

    rclpy.spin(octomap_saver)

    octomap_saver.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()