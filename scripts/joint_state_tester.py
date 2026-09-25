import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState


class TestJointPublisher(Node):

    def __init__(self):
        super().__init__("test_joint_publisher")

        # robot_state_publisher использует BEST_EFFORT для /joint_states,
        # поэтому задаём такой же QoS явно.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.pub = self.create_publisher(
            JointState,
            "/joint_states",
            qos,
        )

        self.timer = self.create_timer(
            0.02,  # 50 Hz
            self.publish_joint_state,
        )

        # ВАЖНО:
        # Здесь присутствуют ВСЕ суставы между pelvis и torso,
        # включая waist_roll и waist_pitch.
        self.joint_names = [
            # Waist
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",

            # Right arm
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",

            # Left arm
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
        ]

        self.get_logger().info(
            f"Publishing {len(self.joint_names)} joints at 50 Hz"
        )

    def publish_joint_state(self):

        msg = JointState()

        msg.header.stamp = self.get_clock().now().to_msg()

        msg.name = self.joint_names

        # ВСЕ СУСТАВЫ = 0
        msg.position = [0.0] * len(self.joint_names)

        self.pub.publish(msg)


def main(args=None):

    rclpy.init(args=args)

    node = TestJointPublisher()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()