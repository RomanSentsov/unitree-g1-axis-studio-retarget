#!/usr/bin/env python3
"""ROS 2 /joint_states -> Unitree G1 в симуляторе (rt/lowcmd, DDS channel 1)."""

import time
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unitree_g1.Unitree_g1 import UnitreeG1


CONTROL_DT = 0.02              # 50 Гц

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


class Bridge(Node):
    def __init__(self, g1):
        super().__init__("joint_states_to_g1_sim")
        self.g1 = g1
        self.last_send = 0.0
        self.msg_count = 0

        arm_cb_group = MutuallyExclusiveCallbackGroup()
        hand_r_cb_group = MutuallyExclusiveCallbackGroup()
        hand_l_cb_group = MutuallyExclusiveCallbackGroup()

        self.create_subscription(JointState, TOPIC_ARMS, self.on_msg_arms, 10,
                                 callback_group=arm_cb_group)
        self.create_subscription(Float64MultiArray, TOPIC_R_HAND, self.on_msg_r_hand, 1,
                                 callback_group=hand_r_cb_group)
        self.create_subscription(Float64MultiArray, TOPIC_L_HAND, self.on_msg_l_hand, 1,
                                 callback_group=hand_l_cb_group)
        self.get_logger().info("слушаю топики (SIM)")

    def on_msg_arms(self, msg):
        self.last_send = time.time()
        self.msg_count += 1

        jp = dict(zip(msg.name, msg.position))

        def vec(names):
            return [float(jp.get(n, 0.0)) for n in names]

        self.g1.set_arm_l(vec(LEFT_ARM_NAMES))
        self.g1.set_arm_r(vec(RIGHT_ARM_NAMES))
        self.g1.set_waist(vec(WAIST_NAMES))

        if self.msg_count % 100 == 0:
            self.get_logger().info(f"принято {self.msg_count} сообщений /joint_states")

    def on_msg_l_hand(self, msg):
        # TODO: интерфейс кисти симулятора (заглушка)
        q = np.array(msg.data) * 1800 * 4 / 3.14
        self.g1.set_hand_l(q)

    def on_msg_r_hand(self, msg):
        # TODO: интерфейс кисти симулятора (заглушка)
        q = np.array(msg.data) * 1800 * 4 / 3.14
        self.g1.set_hand_r(q)


def main():
    rclpy.init()
    print("ros2 inited")

    g1 = UnitreeG1(sim=True, control_dt=CONTROL_DT)
    print("g1 inited (SIM)")

    node = Bridge(g1)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    print("node inited")

    # В симуляторе arm_sdk включать не нужно — команды идут сразу.
    print("стримлю команды в rt/lowcmd, Ctrl+C для выхода")

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        print("остановка...")
        g1.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()