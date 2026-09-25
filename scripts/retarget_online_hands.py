from pin_opti import G1_29_ArmIK
from utils.axis_studio_bvh import AxisStudioFK
import time
import array

import numpy as np
# for publishing hand and joint states
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hands_retargeting.retargeting_wrapper import HandRetargeterWrapper
from hands_retargeting.viser_wrapper import ViserHandsVisualizer

from mocap_api import *
import logging

logging.getLogger("yourdfpy").setLevel(logging.ERROR)

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

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

# Unitree g1 joint names:

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

# Application class
class MocapAxisDemo:

    def __init__(self):

        self.app = None
        self.running = False

        self.prev_posture_time_ms = None

        # init arm ik (retargeter)
        self.arm_ik = G1_29_ArmIK(False)

        # hand retarget
        self.retargeter_right = HandRetargeterWrapper(
            config_path=CONFIG_PATH_RIGHT,
            joint_names=RIGHT_HAND_NODES,
            assets_dir=ASSETS_DIR,
        )
        self.retargeter_left = HandRetargeterWrapper(
            config_path=CONFIG_PATH_LEFT,
            joint_names=LEFT_HAND_NODES,
            assets_dir=ASSETS_DIR,
        )

        self.all_hand_nodes = RIGHT_HAND_NODES + LEFT_HAND_NODES

        # publisher init
        rclpy.init()
        self.ros_node = Node("mocap_retargeter")
        self.pub_joints = self.ros_node.create_publisher(JointState, "/joint_states", 10)
        self.pub_hand_l = self.ros_node.create_publisher(Float64MultiArray, "/hand_state/l", 10)
        self.pub_hand_r = self.ros_node.create_publisher(Float64MultiArray, "/hand_state/r", 10)
        self.timer = self.ros_node.create_timer(0.02, lambda: self.publish_joint_state())
        self.debug_printed = False


    def start(self, udp_port=7012):

        self.app = MCPApplication()
        settings = MCPSettings()
        settings.set_udp(udp_port)
        # Should be coherent with Axis Studio BVH stream settings
        settings.set_bvh_rotation(
            MCPBvhRotation.XYZ
        )
        self.app.set_settings(settings)
        self.app.open()
        print(
            f"Mocap application initialized, "
            f"UDP port: {udp_port}"
        )
        self.running = True

        try:
            rclpy.spin(self.ros_node)
        except KeyboardInterrupt:
            print("Program interrupted by user")
        finally:
            self.stop()

    def publish_joint_state(self):
        evts = self.app.poll_next_event()
        for evt in evts:
            if evt.event_type == MCPEventType.AvatarUpdated:
                self._handle_avatar_data(evt)

    def stop(self):
        self.running = False

        rclpy.shutdown()
        
        if self.app:
            self.app.close()
            print("Mocap application closed")

    def _handle_avatar_data(self, evt):

        # event data
        avatar = MCPAvatar(evt.event_data.avatar_handle)

        # joint data
        joints = avatar.get_joints()

        current_time_ms = time.time() * 1000.0 # TODO change to time from joint_data

        # joint names
        joints_dict = {
            j.get_name(): j
            for j in joints
        }

        _cache = {}

        # Retarget
        bvh_frame = {name: AxisStudioFK.get_global_transform(joints_dict, name, relative_to_hips=True, _cache=_cache)[0] for name in AxisStudioFK.NODE_NAMES}
        until = time.time()

        # Solve ik in this block
        q, _ = self.arm_ik.solve_ik_bvh_frame(bvh_frame)

        q_r_hand = self.retargeter_right.retarget(bvh_frame)[[2, 6, 4, 0, 9, 8]]
        q_l_hand = self.retargeter_left.retarget(bvh_frame)[[2, 6, 4, 0, 9, 8]]

        stamp = self.ros_node.get_clock().now().to_msg()
        print(q)

        # publish arm and waist joints
        js = JointState()
        js.header.stamp = stamp
        js.name = G1_FULL_JOINT_NAMES
        joint_positions = [0.0] * len(G1_FULL_JOINT_NAMES)
        joint_positions[G1_FULL_JOINT_NAMES.index("waist_yaw_joint")] = float(q[0])

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

        self.pub_joints.publish(js)

        #publish r and l hands
        js = Float64MultiArray()
        js.data = [float(x) for x in q_r_hand]
        self.pub_hand_r.publish(js)

        js = Float64MultiArray()
        js.data = [float(x) for x in q_l_hand]
        self.pub_hand_l.publish(js)
        
        elapsed = time.time() - until
        print(f"Solved and published in: {elapsed * 1000.0:.2f} ms")


if __name__ == "__main__":

    demo = MocapAxisDemo()
    print("Starting Mocap Axis Studio demo...")
    demo.start()