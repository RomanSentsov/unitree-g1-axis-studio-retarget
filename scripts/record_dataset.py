import os
import sys
import time
import argparse
import threading
from typing import Optional, List
from enum import IntEnum

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from sshkeyboard import listen_keyboard, stop_listening

# --- ВСТРОЕННЫЙ ЧТЕНИЕ СОСТОЯНИЯ G1_29 ЧЕРЕЗ DDS (Только чтение) ---
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as hg_LowState

class G1_29_JointArmIndex(IntEnum):
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28

class G1_29_StateReader:
    def __init__(self, domain_id: int = 0, ifname: str = None):
        if ifname:
            ChannelFactoryInitialize(domain_id, ifname)
        else:
            ChannelFactoryInitialize(domain_id)
        
        self.latest_q = np.zeros(14)
        self.latest_dq = np.zeros(14)
        self.lock = threading.Lock()
        self.sub_ready = False

        self.sub = ChannelSubscriber("rt/lowstate", hg_LowState)
        self.sub.Init()

        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

        timeout = 5.0
        start = time.time()
        while not self.sub_ready and (time.time() - start) < timeout:
            time.sleep(0.01)
        if not self.sub_ready:
            raise RuntimeError("Не удалось получить данные DDS rt/lowstate в течение 5 секунд.")

    def _read_loop(self):
        while True:
            msg = self.sub.Read()
            if msg is not None:
                with self.lock:
                    for idx, joint_id in enumerate(G1_29_JointArmIndex):
                        self.latest_q[idx] = msg.motor_state[joint_id].q
                        self.latest_dq[idx] = msg.motor_state[joint_id].dq
                self.sub_ready = True
            time.sleep(0.002)

    def get_current_dual_arm_q(self) -> np.ndarray:
        with self.lock:
            return self.latest_q.copy()

    def get_current_dual_arm_dq(self) -> np.ndarray:
        with self.lock:
            return self.latest_dq.copy()
# -----------------------------------------------------------------------------------------

# ИМПОРТ НОВОГО API TELEIMAGER
from teleimager.client import TeleImageClient
from utils.episode_writer import EpisodeWriter

# --- Глобальные переменные ---
STOP = False
RECORD_RUNNING = False
RECORD_TOGGLE = False

def on_press(key):
    global STOP, RECORD_TOGGLE
    if key == 's':
        RECORD_TOGGLE = True
    elif key == 'q':
        STOP = True

ARM_JOINTS_LEFT = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
]
ARM_JOINTS_RIGHT = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

class DemoRecorderNode(Node):
    def __init__(self, args):
        super().__init__('demo_recorder_node')
        self.args = args
        
        # 1. Инициализация чтения состояния робота через DDS
        domain_id = 1 if args.sim else 0
        self.get_logger().info(f"Initializing DDS State Reader (domain_id={domain_id}, ifname={args.network_interface})...")
        self.arm_state_reader = G1_29_StateReader(domain_id=domain_id, ifname=args.network_interface)
        self.get_logger().info("✅ Unitree DDS State Reader initialized.")

        # 2. Подписка на КОМАНДЫ (actions) из ROS 2
        self.latest_action_js: Optional[JointState] = None
        self.action_sub = self.create_subscription(
            JointState, '/joint_states', self._action_cb, 10)
        
        # 3. Подписки на кисти
        self.latest_hand_l: Optional[Float64MultiArray] = None
        self.latest_hand_r: Optional[Float64MultiArray] = None
        
        self.hand_l_sub = self.create_subscription(
            Float64MultiArray, '/hand_l', self._hand_l_cb, 10)
        self.hand_r_sub = self.create_subscription(
            Float64MultiArray, '/hand_r', self._hand_r_cb, 10)

        # 4. Клиент изображений (НОВОЕ API)
        self.get_logger().info(f"Scanning Image Server at {args.img_server_ip}...")
        self.camera_config, from_server = TeleImageClient.scan(server_host=args.img_server_ip)
        
        if not from_server:
            self.get_logger().error("Не удалось получить конфигурацию камер с сервера. Проверьте IP и запущен ли teleimager-server.")
            raise RuntimeError("Camera config fetch failed")

        self.camera_clients = {}
        for topic, cfg in self.camera_config.items():
            if cfg.get('enable_zmq'):
                self.get_logger().info(f"Initializing ZMQ client for camera: {topic} (port {cfg['zmq_port']})")
                # ВАЖНО: topic должен быть первым позиционным аргументом!
                self.camera_clients[topic] = TeleImageClient(
                    topic,
                    server_host=args.img_server_ip,
                    zmq_port=cfg['zmq_port'],
                    request_bgr=True
                )

        # 5. Инициализация рекордера
        task_path = os.path.join(args.task_dir, args.task_name)
        self.recorder = EpisodeWriter(
            task_dir=task_path,
            task_goal=args.task_goal,
            task_desc=args.task_desc,
            task_steps=args.task_steps,
            frequency=args.frequency,
            rerun_log=False
        )
        self.get_logger().info("✅ Recorder initialized.")

        # 6. Клавиатура
        self.kb_thread = threading.Thread(
            target=listen_keyboard,
            kwargs={"on_press": on_press, "until": None, "sequential": False},
            daemon=True
        )
        self.kb_thread.start()
        self.get_logger().info("🟢 Press [s] to START/SAVE recording. Press [q] to exit.")

    def _action_cb(self, msg: JointState):
        self.latest_action_js = msg

    def _hand_l_cb(self, msg: Float64MultiArray):
        self.latest_hand_l = msg

    def _hand_r_cb(self, msg: Float64MultiArray):
        self.latest_hand_r = msg

    def _extract_arm_data_from_ros(self, msg: JointState, side: str) -> tuple[List[float], List[float], List[float]]:
        if msg is None or not msg.name:
            return [], [], []
        
        target_names = ARM_JOINTS_LEFT if side == "left" else ARM_JOINTS_RIGHT
        qpos, qvel, eff = [], [], []
        
        for name in target_names:
            if name in msg.name:
                idx = msg.name.index(name)
                qpos.append(float(msg.position[idx]) if idx < len(msg.position) else 0.0)
                qvel.append(float(msg.velocity[idx]) if msg.velocity and idx < len(msg.velocity) else 0.0)
                eff.append(float(msg.effort[idx]) if msg.effort and idx < len(msg.effort) else 0.0)
            else:
                qpos.append(0.0)
                qvel.append(0.0)
                eff.append(0.0)
                
        return qpos, qvel, eff

    def _build_data_dict(self):
        try:
            current_lr_arm_q = self.arm_state_reader.get_current_dual_arm_q()
            current_lr_arm_dq = self.arm_state_reader.get_current_dual_arm_dq()
            
            half = len(current_lr_arm_q) // 2
            left_arm_qpos = current_lr_arm_q[:half].tolist()
            right_arm_qpos = current_lr_arm_q[half:].tolist()
            left_arm_qvel = current_lr_arm_dq[:half].tolist()
            right_arm_qvel = current_lr_arm_dq[half:].tolist()
        except Exception as e:
            self.get_logger().warning(f"Failed to read DDS state: {e}")
            return None, None

        hand_l_qpos = list(self.latest_hand_l.data) if self.latest_hand_l else []
        hand_r_qpos = list(self.latest_hand_r.data) if self.latest_hand_r else []

        states = {
            "left_arm": {"qpos": left_arm_qpos, "qvel": left_arm_qvel, "torque": []},
            "right_arm": {"qpos": right_arm_qpos, "qvel": right_arm_qvel, "torque": []},
            "left_ee": {"qpos": hand_l_qpos, "qvel": [], "torque": []},
            "right_ee": {"qpos": hand_r_qpos, "qvel": [], "torque": []},
            "body": {"qpos": []} 
        }

        if self.latest_action_js is None:
            return None, None

        l_a_qpos, l_a_qvel, l_a_eff = self._extract_arm_data_from_ros(self.latest_action_js, "left")
        r_a_qpos, r_a_qvel, r_a_eff = self._extract_arm_data_from_ros(self.latest_action_js, "right")
        
        hand_l_action = list(self.latest_hand_l.data) if self.latest_hand_l else []
        hand_r_action = list(self.latest_hand_r.data) if self.latest_hand_r else []

        actions = {
            "left_arm": {"qpos": l_a_qpos, "qvel": l_a_qvel, "torque": l_a_eff},
            "right_arm": {"qpos": r_a_qpos, "qvel": r_a_qvel, "torque": r_a_eff},
            "left_ee": {"qpos": hand_l_action, "qvel": [], "torque": []},
            "right_ee": {"qpos": hand_r_action, "qvel": [], "torque": []},
            "body": {"qpos": []}
        }

        return states, actions

    def run(self):
        global RECORD_RUNNING, RECORD_TOGGLE, STOP
        try:
            while rclpy.ok() and not STOP:
                rclpy.spin_once(self, timeout_sec=0.01)
                start_time = time.time()

                if RECORD_TOGGLE:
                    RECORD_TOGGLE = False
                    if not RECORD_RUNNING:
                        if self.recorder.create_episode():
                            RECORD_RUNNING = True
                            self.get_logger().info("🔴 RECORDING STARTED")
                        else:
                            self.get_logger().error("Failed to create episode.")
                    else:
                        RECORD_RUNNING = False
                        self.recorder.save_episode()
                        self.get_logger().info("💾 EPISODE SAVED")

                if RECORD_RUNNING:
                    states, actions = self._build_data_dict()
                    if states is None or actions is None:
                        continue

                    colors = {}
                    depths = {}
                    
                    # Обработка головной камеры
                    if 'head_camera' in self.camera_clients:
                        head_frame = self.camera_clients['head_camera'].get_frame()
                        if head_frame.bgr is not None:
                            if self.camera_config['head_camera'].get('binocular'):
                                w = self.camera_config['head_camera']['image_shape'][1] // 2
                                colors["color_0"] = head_frame.bgr[:, :w]
                                colors["color_1"] = head_frame.bgr[:, w:]
                            else:
                                colors["color_0"] = head_frame.bgr

                    # Обработка камер на запястьях (динамическое назначение индексов)
                    color_idx = 2 if self.camera_config.get('head_camera', {}).get('binocular') else 1
                    
                    for cam_name in ['left_wrist_camera', 'right_wrist_camera']:
                        if cam_name in self.camera_clients:
                            frame = self.camera_clients[cam_name].get_frame()
                            if frame.bgr is not None:
                                colors[f"color_{color_idx}"] = frame.bgr
                                color_idx += 1

                    self.recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)

                time_elapsed = time.time() - start_time
                time.sleep(max(0.0, (1.0 / self.args.frequency) - time_elapsed))
                
        except KeyboardInterrupt:
            self.get_logger().info("⛔ KeyboardInterrupt received.")
        finally:
            self.cleanup()

    def cleanup(self):
        global RECORD_RUNNING
        self.get_logger().info("Shutting down...")
        if RECORD_RUNNING:
            self.recorder.save_episode()
        self.recorder.close()
        
        # Закрытие всех клиентов камер (НОВОЕ API)
        for topic, client in self.camera_clients.items():
            client.close()
            
        stop_listening()
        self.kb_thread.join(timeout=1.0)
        self.get_logger().info("✅ Cleanup completed.")


def main():
    parser = argparse.ArgumentParser(description="ROS 2 + DDS Teleoperation Demo Recorder")
    parser.add_argument('--frequency', type=float, default=30.0, help='Recording frequency (Hz)')
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='IP of teleimager server')
    parser.add_argument('--task-dir', type=str, default='./data/', help='Path to save data')
    parser.add_argument('--task-name', type=str, default='demo_task', help='Task folder name')
    parser.add_argument('--task-goal', type=str, default='record demonstration', help='Task goal description')
    parser.add_argument('--task-desc', type=str, default='teleop recording', help='Task description')
    parser.add_argument('--task-steps', type=str, default='step1: approach; step2: grasp;', help='Task steps')
    parser.add_argument('--sim', action='store_true', help='Enable simulation mode (DDS domain_id=1)')
    parser.add_argument('--network-interface', type=str, default='enxc84d4427fee8', help='Network interface for DDS (e.g., eth0, wlan0)')
    
    args = parser.parse_args()
    rclpy.init()
    node = DemoRecorderNode(args)
    try:
        node.run()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()