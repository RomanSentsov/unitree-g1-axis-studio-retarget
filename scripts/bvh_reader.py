import pybvh
from pybvh import bvhplot
from pin_opti import G1_29_ArmIK
import pinocchio as pin
import numpy as np
import time

import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from hands_retargeting.retargeting_wrapper import HandRetargeterWrapper

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ------------------------------ BVH OPEN ------------------------------
ASSETS_DIR = PROJECT_ROOT / "hands"
BVH_FILEPATH = PROJECT_ROOT / "recs" / "bvh_recs" / "bvh_misha_no_xyz_chr01_MAYA.bvh"

CONFIG_PATH_RIGHT = PROJECT_ROOT / "scripts" / "hands_retargeting" / "configs" / "teleop" / "rh56dftp_right.yml"
CONFIG_PATH_LEFT = PROJECT_ROOT / "scripts" / "hands_retargeting" / "configs" / "teleop" / "rh56dftp_left.yml"

URDF_PATH_RIGHT = ASSETS_DIR / "rh56dftp" / "rh56dftp_modified_right.urdf"
URDF_PATH_LEFT = ASSETS_DIR / "rh56dftp" / "rh56dftp_modified_left.urdf"

RIGHT_HAND_NODES = [
    "RightHand",
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "EndSiteRightHandThumb3",
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "EndSiteRightHandIndex3",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "EndSiteRightHandMiddle3",
    "RightHandRing1", "RightHandRing2", "RightHandRing3", "EndSiteRightHandRing3",
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3", "EndSiteRightHandPinky3",
]

LEFT_HAND_NODES = [
    "LeftHand",
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3", "EndSiteLeftHandThumb3",
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "EndSiteLeftHandIndex3",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "EndSiteLeftHandMiddle3",
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "EndSiteLeftHandRing3",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3", "EndSiteLeftHandPinky3",
]



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

FRAMERATE = 50
FILEPATH = "recs/bvh_recs/bvh_misha_no_xyz_chr01_MAYA.bvh"
# FILEPATH = "recs/bvh_recs/Aleksandr_recs/rec1_chr01_MAYA.bvh"
FILEPATH_EX = "recs/bvh_recs/example1.bvh"


# hands

retargeter_right = HandRetargeterWrapper(
            config_path=CONFIG_PATH_RIGHT,
            joint_names=RIGHT_HAND_NODES,
            assets_dir=ASSETS_DIR,
)
        # .retargeter_right.optimizer.opt.set_maxtime(0.015) # in seconds
retargeter_right.optimizer.opt.set_maxeval(5) 
        
retargeter_left = HandRetargeterWrapper(
            config_path=CONFIG_PATH_LEFT,
            joint_names=LEFT_HAND_NODES,
            assets_dir=ASSETS_DIR,
)
        # .retargeter_left.optimizer.opt.set_maxtime(0.015) # in seconds
retargeter_left.optimizer.opt.set_maxeval(5)

all_hand_nodes = RIGHT_HAND_NODES + LEFT_HAND_NODES

# reorient to match pinocchio 
bvh = pybvh.read_bvh_file(FILEPATH)
bvh = bvh.reorient_world_up('+z')
bvh = bvh.rotate_vertical(np.pi / 2)
# bvh.play()

# extract node poses
poses = bvh.node_positions(centered="skeleton")
# poses = np.array(poses[::3, :, :])
idxes_dict = {name: bvh.node_index[name] for name in bvh.joint_names}
print(poses.shape)
frame_num = poses.shape[0]

# ------------------------------ Play BVH via retarget  ------------------------------

# DEBUG
time_arms_arr = []
time_hand_l_arr = []
time_hand_r_arr = []

PUBLISH_ROS2 = True

arm_ik = G1_29_ArmIK(Unit_Test = False, Visualization = False)

ros_node = None
pub_joints = None
pub_hand_l = None
pub_hand_r = None

if PUBLISH_ROS2:
    rclpy.init()
    ros_node = Node("mocap_retargeter")
    pub_joints = ros_node.create_publisher(JointState, "/joint_states", 10)
    pub_hand_l = ros_node.create_publisher(Float64MultiArray, "/hand_state/l", 10)
    pub_hand_r = ros_node.create_publisher(Float64MultiArray, "/hand_state/r", 10)

until = None
now = None

cond = True
while cond:
    cond = False
    for i in range(int(frame_num/8)):

        until = time.perf_counter()
        # current bvh frame
        bvh_dict = {name: poses[i, idxes_dict[name], :] for name in bvh.joint_names} 


        q_bend = 0
        q, tau, q_bend = arm_ik.solve_ik_bvh_frame(bvh_dict, BendToBalance=True)

        q_l_hand = retargeter_left.retarget(bvh_dict)[[2, 6, 4, 0, 9, 8]]
        q_r_hand = retargeter_right.retarget(bvh_dict)[[2, 6, 4, 0, 9, 8]]

        start_time = time.perf_counter()
        # Solve ik in this block
        q, _, q_bend = arm_ik.solve_ik_bvh_frame(bvh_dict, BendToBalance=True)

        time_arms_arr.append(time.perf_counter() - start_time)

        start_time = time.perf_counter()

        q_r_hand = retargeter_right.retarget(bvh_dict)[[2, 6, 4, 0, 9, 8]]

        time_hand_r_arr.append(time.perf_counter() - start_time)

        start_time = time.perf_counter()
        
        q_l_hand = retargeter_left.retarget(bvh_dict)[[2, 6, 4, 0, 9, 8]]

        time_hand_l_arr.append(time.perf_counter() - start_time)

        # print(com[0])

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

            #publish r and l hands
            js = Float64MultiArray()
            js.data = [float(x) for x in q_r_hand]
            pub_hand_r.publish(js)

            js = Float64MultiArray()
            js.data = [float(x) for x in q_l_hand]
            pub_hand_l.publish(js)


        elapsed = time.perf_counter() - until
        print(f"time: {elapsed}")
        # print(q)
        time.sleep(np.max([1/FRAMERATE - elapsed, 0.00001]))


        x = list(range(len(time_arms_arr)))   # [0, 1, 2, 3, 4, 5]

# Строим линии
plt.plot(x, time_arms_arr, marker='o', label='Руки')
plt.plot(x, time_hand_l_arr, marker='s', label='Левая кисть')
plt.plot(x, time_hand_r_arr, marker='^', label='Правая кисть')

plt.xlabel('Номер точки')
plt.ylabel('Время ретаргетинга')
plt.title('Время ретаргетинга')
plt.legend()
plt.grid(True)
plt.show()

    