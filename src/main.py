#!/usr/bin/env python3
import os
import sys
import time
import argparse
import threading
from enum import IntEnum

import logging
logging.getLogger("RH56DFTP").setLevel(logging.WARNING)

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from sshkeyboard import listen_keyboard, stop_listening

# Импорты робота и записи
from unitree_g1.Unitree_g1 import UnitreeG1
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as hg_LowState
from teleimager.client import TeleImageClient
from utils.episode_writer import EpisodeWriter

# Импорт класса ретаргета (предполагается, что он лежит в mocap_retargeter.py)
from mocap_retargeter import MocapRetargeter

# --- Константы ---
ARM_COMMAND_TOPIC = '/unitree_noitom_teleop/arms_waist_cmd'
ENABLE_DURATION = 5.0

LEFT_ARM_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
]
RIGHT_ARM_NAMES = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
WAIST_NAMES = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]

G1_FULL_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint", "R_thumb_MCP_joint1", "R_thumb_MCP_joint2",
    "R_thumb_PIP_joint", "R_thumb_DIP_joint", "R_index_MCP_joint", "R_index_DIP_joint",
    "R_middle_MCP_joint", "R_middle_DIP_joint", "R_ring_MCP_joint", "R_ring_DIP_joint",
    "R_pinky_MCP_joint", "R_pinky_DIP_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "L_thumb_MCP_joint1", "L_thumb_MCP_joint2", "L_thumb_PIP_joint",
    "L_thumb_DIP_joint", "L_index_MCP_joint", "L_index_DIP_joint", "L_middle_MCP_joint",
    "L_middle_DIP_joint", "L_ring_MCP_joint", "L_ring_DIP_joint", "L_pinky_MCP_joint", "L_pinky_DIP_joint"
]

# --- DDS State Reader ---
class G1_29_JointArmIndex(IntEnum):
    kLeftShoulderPitch = 15; kLeftShoulderRoll = 16; kLeftShoulderYaw = 17; kLeftElbow = 18
    kLeftWristRoll = 19; kLeftWristPitch = 20; kLeftWristyaw = 21
    kRightShoulderPitch = 22; kRightShoulderRoll = 23; kRightShoulderYaw = 24; kRightElbow = 25
    kRightWristRoll = 26; kRightWristPitch = 27; kRightWristYaw = 28

class G1_29_JointWaistIndex(IntEnum):
    kWaistYaw = 12; kWaistRoll = 13; kWaistPitch = 14

class G1_29_StateReader:
    def __init__(self, domain_id: int = 0, ifname: str = None):
        if ifname:
            ChannelFactoryInitialize(domain_id, ifname)
        else:
            ChannelFactoryInitialize(domain_id)
        
        self.latest_q = np.zeros(14)
        self.latest_dq = np.zeros(14)
        self.latest_waist_q = np.zeros(3)
        self.latest_waist_dq = np.zeros(3)
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

    def _read_loop(self):
        while True:
            msg = self.sub.Read()
            if msg is not None:
                with self.lock:
                    for idx, joint_id in enumerate(G1_29_JointArmIndex):
                        self.latest_q[idx] = msg.motor_state[joint_id].q
                        self.latest_dq[idx] = msg.motor_state[joint_id].dq
                    for idx, joint_id in enumerate(G1_29_JointWaistIndex):
                        self.latest_waist_q[idx] = msg.motor_state[joint_id].q
                        self.latest_waist_dq[idx] = msg.motor_state[joint_id].dq
                self.sub_ready = True
            time.sleep(0.002)

    def get_current_dual_arm_q(self): return self.latest_q.copy()
    def get_current_dual_arm_dq(self): return self.latest_dq.copy()
    def get_current_waist_q(self): return self.latest_waist_q.copy()
    def get_current_waist_dq(self): return self.latest_waist_dq.copy()


class MainController:
    def __init__(self, args):
        self.args = args
        self.is_running = True
        self.robot_enabled = False
        self.is_recording = False

        # 1. Ретаргет
        self.retargeter = MocapRetargeter()
        self.received_first_cmd = False

        # 2. ROS 2
        rclpy.init()
        self.ros_node = Node("teleop_main_node")
        self.ros_pub_joints = self.ros_node.create_publisher(JointState, ARM_COMMAND_TOPIC, 10)

        # 3. Робот (опционально)
        self.g1 = None
        if self.args.control_robot:
            self._init_robot()

        # 4. Запись датасета (только если включено)
        self.recorder = None
        self.camera_clients = {}
        self.camera_config = None
        self.arm_state_reader = None
        
        if self.args.record_dataset:
            self.arm_state_reader = G1_29_StateReader(domain_id=1 if args.sim else 0, ifname=args.network_interface)
            self._init_recorder()

        # 5. Клавиатура
        self.kb_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self.kb_thread.start()

    def _init_robot(self):
        print("Инициализация UnitreeG1...")
        self.g1 = UnitreeG1(domain_id=1 if self.args.sim else 0, ifname=self.args.network_interface, control_dt=0.02)
        print("g1 inited")

    def _init_recorder(self):
        print("Инициализация рекордера и камер...")
        self.camera_config, from_server = TeleImageClient.scan(server_host=self.args.img_server_ip)
        if from_server:
            for topic, cfg in self.camera_config.items():
                if cfg.get('enable_zmq'):
                    self.camera_clients[topic] = TeleImageClient(
                        topic, server_host=self.args.img_server_ip,
                        zmq_port=cfg['zmq_port'], request_bgr=True
                    )
        else:
            print("Внимание: не удалось получить конфиг камер.")

        task_path = os.path.join(self.args.task_dir, self.args.task_name)
        self.recorder = EpisodeWriter(
            task_dir=task_path, task_goal=self.args.task_goal,
            task_desc=self.args.task_desc, task_steps=self.args.task_steps,
            frequency=self.args.record_frequency, rerun_log=False
        )

    def _keyboard_loop(self):
        def on_press(key):
            if key == 'e':
                if not self.received_first_cmd:
                    print("can't enable: no comands were yet receiced")
                elif self.args.control_robot and self.g1:
                    self.robot_enabled = not self.robot_enabled
                    if self.robot_enabled:
                        print("will start enabling hands in 5 seconds")
                        time.sleep(5.0)
                        self.g1.enable_arm_sdk(duration=ENABLE_DURATION)
                        print(f"включаю arm_sdk ({ENABLE_DURATION} c)...")
                        time.sleep(ENABLE_DURATION + 0.2)
                    else:
                        print("выключаю arm_sdk...")
                        self.g1.disable_arm_sdk(duration=ENABLE_DURATION)
                        self.g1.hand_r.open()
                        self.g1.hand_l.open()
                        time.sleep(ENABLE_DURATION + 0.2)
                else:
                    print("Управление роботом не включено (используйте --control-robot).")
                    
            elif key == 's':
                if not self.args.record_dataset:
                    print("Запись датасета отключена. Запустите программу с флагом --record-dataset")
                elif not self.is_recording:
                    if self.recorder.create_episode():
                        self.is_recording = True
                        print("🔴 RECORDING STARTED")
                        if self.g1:
                            self.g1.say_ep_start() # narrator
                else:
                    self.is_recording = False
                    self.recorder.save_episode()
                    print("💾 EPISODE SAVED")
                    if self.g1:
                        self.g1.say_ep_saved() # narrator
                    
            elif key == 'q':
                print("Завершение работы...")
                self.is_running = False

        listen_keyboard(on_press=on_press, sequential=False, until=None)

    def _publish_body_to_ros2(self, arm_waist_dict):
        js = JointState()
        js.header.stamp = self.ros_node.get_clock().now().to_msg()
        js.name = G1_FULL_JOINT_NAMES
        joint_positions = [0.0] * len(G1_FULL_JOINT_NAMES)
        
        for name, value in arm_waist_dict.items():
            if name in G1_FULL_JOINT_NAMES:
                joint_positions[G1_FULL_JOINT_NAMES.index(name)] = float(value)
                
        js.position = joint_positions
        self.ros_pub_joints.publish(js)

    def _send_to_robot(self, arm_waist_dict, q_l_hand, q_r_hand):

        # send nothing if command is not valid
        required_names = LEFT_ARM_NAMES + RIGHT_ARM_NAMES + WAIST_NAMES
        if not all(name in arm_waist_dict for name in required_names):
            print("hell nah") # debug
            return

        l_arm = [arm_waist_dict[n] for n in LEFT_ARM_NAMES]
        r_arm = [arm_waist_dict[n] for n in RIGHT_ARM_NAMES]
        waist = [arm_waist_dict[n] for n in WAIST_NAMES]
        
        self.g1.set_arm_l(l_arm)
        self.g1.set_arm_r(r_arm)
        self.g1.set_waist(waist)
        
        if len(q_l_hand) == 6 and len(q_r_hand) == 6:
            q_l = np.array(q_l_hand) * 1800 * 4 / 3.14
            q_r = np.array(q_r_hand) * 1800 * 4 / 3.14
            self.g1.set_hand_l(q_l)
            self.g1.set_hand_r(q_r)

    def _build_data_dict(self):
        try:
            current_lr_arm_q = self.arm_state_reader.get_current_dual_arm_q()
            current_lr_arm_dq = self.arm_state_reader.get_current_dual_arm_dq()
            current_waist_q = self.arm_state_reader.get_current_waist_q()
            current_waist_dq = self.arm_state_reader.get_current_waist_dq()
            
            half = len(current_lr_arm_q) // 2
            states = {
                "left_arm": {"qpos": current_lr_arm_q[:half].tolist(), "qvel": current_lr_arm_dq[:half].tolist(), "torque": []},
                "right_arm": {"qpos": current_lr_arm_q[half:].tolist(), "qvel": current_lr_arm_dq[half:].tolist(), "torque": []},
                "body": {"qpos": current_waist_q.tolist(), "qvel": current_waist_dq.tolist(), "torque": []} 
            }
        except Exception:
            return None, None

        retarget_res = self.retargeter.get_latest_retarget()
        if retarget_res is None:
            arm_waist_dict, q_l_hand, q_r_hand = {}, [0.0]*6, [0.0]*6
        else:
            arm_waist_dict, q_l_hand, q_r_hand = retarget_res

        # TODO: states in ee should be observations, but currently are just copied actions
        states["left_ee"] = {"qpos": q_l_hand, "qvel": [], "torque": []}
        states["right_ee"] = {"qpos": q_r_hand, "qvel": [], "torque": []}

        actions = {
            "left_arm": {"qpos": [arm_waist_dict.get(n, 0.0) for n in LEFT_ARM_NAMES], "qvel": [], "torque": []},
            "right_arm": {"qpos": [arm_waist_dict.get(n, 0.0) for n in RIGHT_ARM_NAMES], "qvel": [], "torque": []},
            "left_ee": {"qpos": q_l_hand, "qvel": [], "torque": []},
            "right_ee": {"qpos": q_r_hand, "qvel": [], "torque": []},
            "body": {"qpos": [arm_waist_dict.get(n, 0.0) for n in WAIST_NAMES], "qvel": [], "torque": []} # append 0.0 for the waist_roll
        }
        return states, actions

    def _record_loop(self):
        rate = 1.0 / self.args.record_frequency
        while self.is_running:
            start_time = time.perf_counter()
            if self.is_recording:
                states, actions = self._build_data_dict()
                if states and actions:
                    colors = {}
                    color_idx = 0
                    for cam_name in ['head_camera', 'left_wrist_camera', 'right_wrist_camera']:
                        if cam_name in self.camera_clients:
                            frame = self.camera_clients[cam_name].get_frame()
                            if frame and frame.bgr is not None:
                                if self.camera_config[cam_name].get('binocular'):
                                    w = self.camera_config[cam_name]['image_shape'][1] // 2
                                    colors[f"color_{color_idx}"] = frame.bgr[:, :w]; color_idx += 1
                                    colors[f"color_{color_idx}"] = frame.bgr[:, w:]; color_idx += 1
                                else:
                                    colors[f"color_{color_idx}"] = frame.bgr; color_idx += 1
                    self.recorder.add_item(colors=colors, depths={}, states=states, actions=actions)
            
            elapsed = time.perf_counter() - start_time
            if elapsed < rate:
                time.sleep(rate - elapsed)

    def control_loop(self):
        rate = 1.0 / self.args.frequency
        while self.is_running:
            start_time = time.perf_counter()
            
            rclpy.spin_once(self.ros_node, timeout_sec=0.0)
            
            retarget_res = self.retargeter.get_latest_retarget()

            if retarget_res is not None:
                if not self.received_first_cmd:
                    self.received_first_cmd = True
                arm_waist_dict, q_l_hand, q_r_hand = retarget_res
                self._publish_body_to_ros2(arm_waist_dict)

                if self.args.control_robot:
                    self._send_to_robot(arm_waist_dict, q_l_hand, q_r_hand)
            
            elapsed = time.perf_counter() - start_time
            if elapsed < rate:
                time.sleep(rate - elapsed)

    def run(self):
        self.retargeter.start()
        if self.args.record_dataset:
            threading.Thread(target=self._record_loop, daemon=True).start()
        try:
            self.control_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

    def cleanup(self):
        self.is_running = False
        print("Остановка и сохранение...")
        
        if self.args.record_dataset:
            if self.is_recording and self.recorder:
                self.recorder.save_episode()
            if self.recorder:
                self.recorder.close()
            for client in self.camera_clients.values():
                client.close()
            
        if self.args.control_robot and self.g1:
            if self.robot_enabled:
                print("выключаю arm_sdk...")
                self.g1.disable_arm_sdk(duration=ENABLE_DURATION)
            print("Останавливаю сервисы, пожалуйста, подождите")
            self.g1.hand_r.open()
            self.g1.hand_l.open()
            time.sleep(ENABLE_DURATION + 0.2)
            self.g1.shutdown()
            
        stop_listening()
        self.ros_node.destroy_node()
        rclpy.shutdown()
        self.retargeter.stop()
        print("Завершено.")


def main():
    parser = argparse.ArgumentParser(description="Монолитный скрипт телеуправления Unitree G1 с записью датасета")
    parser.add_argument('--control-robot', action='store_true', help='Включить физическое управление роботом Unitree G1')
    parser.add_argument('--network-interface', type=str, default='enxc84d4427fee8', help='Сетевой интерфейс для DDS (например, eth0, enxc84d4427fee8)')
    parser.add_argument('--record-dataset', action='store_true', help='Включить запись датасета (инициализация камер и EpisodeWriter)')
    parser.add_argument('--task-dir', type=str, default='./data/', help='Директория для сохранения данных датасета')
    parser.add_argument('--task-name', type=str, default='demo_task', help='Имя папки/задачи для сохранения эпизода')
    parser.add_argument('--task-goal', type=str, default='record demonstration', help='Описание цели задачи (для метаданных датасета)')
    parser.add_argument('--task-desc', type=str, default='teleop recording', help='Текстовое описание телеоперации (для метаданных)')
    parser.add_argument('--task-steps', type=str, default='step1: approach; step2: grasp;', help='Шаги задачи, разделенные точкой с запятой (для метаданных)')
    parser.add_argument('--frequency', type=float, default=50.0, help='Частота управления и публикации в ROS 2 (Гц)')
    parser.add_argument('--record-frequency', type=float, default=50.0, help='Частота записи датасета (Гц)')
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='IP-адрес сервера камер (teleimager-server)')
    parser.add_argument('--sim', action='store_true', help='Режим симуляции (использует domain_id=1 для DDS)')
    
    args = parser.parse_args()
    controller = MainController(args)
    print("🟢 Press [e] to Enable/Disable robot. Press [s] to Start/Stop recording. Press [q] to Quit.")
    controller.run()

if __name__ == '__main__':
    main()