#!/usr/bin/env python3
"""ROS 2 /joint_states -> Unitree G1 (rt/arm_sdk)."""

import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unitree_g1.Unitree_g1 import UnitreeG1


IFNAME = "enxc84d4427fee8"
CONTROL_DT = 0.05              # 20 Гц
ENABLE_DURATION = 5.0
TOPIC = "/joint_states"

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
        self.create_subscription(JointState, TOPIC, self.on_msg, 10)
        self.get_logger().info(f"слушаю {TOPIC}")

    def to_inspire(self, value_rad, max_rad):
        """0..max_rad -> 0..1000 для Inspire Hand."""
        if max_rad <= 0:
            return 0
        v = max(0.0, min(max_rad, value_rad))
        return int(1000.0 * v / max_rad)

    def on_msg(self, msg):
        # 20 Гц
        now = time.time()
        print(f"reseived msg with stamp time {msg}")

        self.last_send = now

        jp = dict(zip(msg.name, msg.position))

        def vec(names):
            return [float(jp.get(n, 0.0)) for n in names]

        self.g1.set_arm_l(vec(LEFT_ARM_NAMES))
        self.g1.set_arm_r(vec(RIGHT_ARM_NAMES))
        self.g1.set_waist(vec(WAIST_NAMES))

        # hand_l = [self.to_inspire(jp.get(n, 0.0), m)
        #         for n, m in zip(LEFT_HAND_NAMES, LEFT_HAND_MAX)]
        # hand_r = [self.to_inspire(jp.get(n, 0.0), m)
        #         for n, m in zip(RIGHT_HAND_NAMES, RIGHT_HAND_MAX)]
        # self.g1.set_hand_l(hand_l)
        # self.g1.set_hand_r(hand_r)


def main():
    rclpy.init()
    print("ros2 inited")

    g1 = UnitreeG1(ifname=IFNAME, control_dt=CONTROL_DT)
    print("g1 inited")
    node = Bridge(g1)
    print("node inited")

    time.sleep(5.0)

    g1.enable_arm_sdk(duration=ENABLE_DURATION)
    print(f"включаю arm_sdk ({ENABLE_DURATION} c)...")
    # time.sleep(ENABLE_DURATION + 0.2)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        print("выключаю arm_sdk...")
        g1.disable_arm_sdk(duration=ENABLE_DURATION)
        # g1.hand_r.open()
        # g1.hand_l.open()
        time.sleep(ENABLE_DURATION + 0.2)
        g1.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()