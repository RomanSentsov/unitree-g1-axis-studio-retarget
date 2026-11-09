#!/usr/bin/env python3
"""ROS 2 /joint_states -> Unitree G1 (rt/arm_sdk)."""

import time
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from typing import Literal
import tyro

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unitree_g1.Unitree_g1 import UnitreeG1


IFNAME = "enxc84d4427fee8"
CONTROL_DT = 0.02              # 50 Гц
ENABLE_DURATION = 5.0

TOPIC_ARMS = "/joint_states"
TOPIC_R_HAND = "/hand_state/r"
TOPIC_L_HAND = "/hand_state/l"

LEFT_ARM_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]

RIGHT_ARM_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

WAIST_NAMES = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]

LEFT_HAND_NAMES = [
    "L_pinky_MCP_joint",
    "L_ring_MCP_joint",
    "L_middle_MCP_joint",
    "L_index_MCP_joint",
    "L_thumb_MCP_joint1",
    "L_thumb_MCP_joint2",
]

RIGHT_HAND_NAMES = [
    "R_pinky_MCP_joint",
    "R_ring_MCP_joint",
    "R_middle_MCP_joint",
    "R_index_MCP_joint",
    "R_thumb_MCP_joint1",
    "R_thumb_MCP_joint2",
]

LEFT_HAND_MAX  = [1.410, 1.410, 1.410, 1.410, 1.096, 0.625]
RIGHT_HAND_MAX = [1.410, 1.410, 1.410, 1.410, 1.096, 0.625]


class Bridge(Node):
    def __init__(self, g1):
        super().__init__("joint_states_to_g1")
        self.g1 = g1
        self.last_send = 0.0

        # Hand callback takes forever blocking arm cb's
        # So we put them in different ExecutorCallbackGroups
        arm_cb_group = MutuallyExclusiveCallbackGroup()
        hand_r_cb_group = MutuallyExclusiveCallbackGroup()
        hand_l_cb_group = MutuallyExclusiveCallbackGroup()

        self.create_subscription(JointState, TOPIC_ARMS, self.on_msg_arms, 10, callback_group=arm_cb_group)
        self.create_subscription(Float64MultiArray, TOPIC_R_HAND, self.on_msg_r_hand, 1, callback_group=hand_r_cb_group)
        self.create_subscription(Float64MultiArray, TOPIC_L_HAND, self.on_msg_l_hand, 1, callback_group=hand_l_cb_group)
        self.get_logger().info(f"слушаю топики (hand_interface={self.g1.hand_interface})")

    def on_msg_arms(self, msg):
  
        now = time.time()

        self.last_send = now

        jp = dict(zip(msg.name, msg.position))

        def vec(names):
            return [float(jp.get(n, 0.0)) for n in names]

        self.g1.set_arm_l(vec(LEFT_ARM_NAMES))
        self.g1.set_arm_r(vec(RIGHT_ARM_NAMES))
        self.g1.set_waist(vec(WAIST_NAMES))

    def on_msg_l_hand(self, msg):
        q = msg.data
        self.g1.set_hand_l(q)

    def on_msg_r_hand(self, msg):
        q = msg.data
        self.g1.set_hand_r(q)


def main(
    sim: bool = False,
    hand_interface: Literal["dds", "tcp"] = "dds",
    ifname: str = IFNAME,
    control_dt: float = CONTROL_DT,
    enable_duration: float = ENABLE_DURATION,
):
    """Bridge ROS 2 /joint_states and hand topics to Unitree G1 & Inspire Hands.

    Args:
        sim: Whether to run in simulation mode (connects to Isaac Lab via DDS channel 1).
        hand_interface: Hand control protocol ('dds' for Unitree DDS rt/inspire/cmd or 'tcp' for Modbus TCP).
        ifname: Network interface for wired connection to the real robot.
        control_dt: Control loop timestep in seconds (default: 0.02 -> 50 Hz).
        enable_duration: Duration in seconds to ramp into arm_sdk on the real robot.
    """
    rclpy.init()
    print("ros2 inited")

    g1 = UnitreeG1(
        ifname=ifname,
        control_dt=control_dt,
        hand_interface=hand_interface,
        sim=sim,
    )
    print(f"g1 inited (sim={sim}, hand_interface={g1.hand_interface})")

    node = Bridge(g1)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    print("node inited")

    if not sim:
        print("will start enabling hands in 5 seconds")
        time.sleep(5.0)
        g1.enable_arm_sdk(duration=enable_duration)
        print(f"включаю arm_sdk ({enable_duration} c)...")
    else:
        print("стримлю команды в симулятор (rt/lowcmd и rt/inspire/cmd), Ctrl+C для выхода")

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        print("выключение...")
        if not sim:
            print("выключаю arm_sdk...")
            g1.disable_arm_sdk(duration=enable_duration)
        g1.hand_r.open()
        g1.hand_l.open()
        time.sleep(enable_duration + 0.2 if not sim else 0.5)
        g1.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    tyro.cli(main)