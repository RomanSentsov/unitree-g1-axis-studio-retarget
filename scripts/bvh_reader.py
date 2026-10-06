import pybvh
from pybvh import bvhplot
from pin_opti import G1_29_ArmIK
import pinocchio as pin
import numpy as np
import time

from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

# ------------------------------ BVH OPEN ------------------------------

G1_FULL_JOINT_NAMES = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
        "R_thumb_MCP_joint1",
        "R_thumb_MCP_joint2",
        "R_thumb_PIP_joint",
        "R_thumb_DIP_joint",
        "R_index_MCP_joint",
        "R_index_DIP_joint",
        "R_middle_MCP_joint",
        "R_middle_DIP_joint",
        "R_ring_MCP_joint",
        "R_ring_DIP_joint",
        "R_pinky_MCP_joint",
        "R_pinky_DIP_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "L_thumb_MCP_joint1",
        "L_thumb_MCP_joint2",
        "L_thumb_PIP_joint",
        "L_thumb_DIP_joint",
        "L_index_MCP_joint",
        "L_index_DIP_joint",
        "L_middle_MCP_joint",
        "L_middle_DIP_joint",
        "L_ring_MCP_joint",
        "L_ring_DIP_joint",
        "L_pinky_MCP_joint",
        "L_pinky_DIP_joint"
    ]

FRAMERATE = 30
FILEPATH = "recs/bvh_recs/bvh_misha_no_xyz_chr01_MAYA.bvh"
# FILEPATH = "recs/bvh_recs/Aleksandr_recs/rec1_chr01_MAYA.bvh"
FILEPATH_EX = "recs/bvh_recs/example1.bvh"

# reorient to match pinocchio 
bvh = pybvh.read_bvh_file(FILEPATH)
bvh = bvh.reorient_world_up('+z')
bvh = bvh.rotate_vertical(np.pi / 2)
# bvh.play()

# extract node poses
poses = bvh.node_positions(centered="skeleton")
poses = np.array(poses[::3, :, :])
idxes_dict = {name: bvh.node_index[name] for name in bvh.joint_names}
print(poses.shape)
frame_num = poses.shape[0]

# ------------------------------ Play BVH via retarget  ------------------------------

PUBLISH_ROS2 = True

arm_ik = G1_29_ArmIK(Unit_Test = False, Visualization = False)

ros_node = None
pub_joints = None

if PUBLISH_ROS2:
    rclpy.init()
    ros_node = Node("mocap_retargeter")
    pub_joints = ros_node.create_publisher(JointState, "/joint_states", 10)

until = None
now = None

while True:
    for i in range(frame_num):

        # current bvh frame
        bvh_dict = {name: poses[i, idxes_dict[name], :] for name in bvh.joint_names} 

        until = time.time()

        q_bend = 0
        q, tau, q_bend = arm_ik.solve_ik_bvh_frame(bvh_dict, BendToBalance=True)

        # print(com[0])

        now = time.time()
        elapsed = now - until

        if PUBLISH_ROS2:
            stamp = ros_node.get_clock().now().to_msg()
            # print(q)

            # publish arm and waist joints
            js = JointState()
            js.header.stamp = stamp
            js.name = G1_FULL_JOINT_NAMES
            joint_positions = [0.0] * len(G1_FULL_JOINT_NAMES)
            joint_positions[G1_FULL_JOINT_NAMES.index("waist_yaw_joint")] = float(q[0])
            joint_positions[G1_FULL_JOINT_NAMES.index("waist_pitch_joint")] = float(q_bend)

            right_arm = [
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ]

            left_arm = [
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
            ]

            for name, value in zip(left_arm, q[1:8]):
                joint_positions[G1_FULL_JOINT_NAMES.index(name)] = float(value)

            for name, value in zip(right_arm, q[8:15]):
                joint_positions[G1_FULL_JOINT_NAMES.index(name)] = float(value)

            js.position = joint_positions

            pub_joints.publish(js)

        print(f"time: {now - until}")
        # print(q)
        time.sleep(np.max([1/FRAMERATE - elapsed, 0.00001]))

    