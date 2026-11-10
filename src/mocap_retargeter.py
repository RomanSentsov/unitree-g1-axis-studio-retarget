from pin_opti import G1_29_ArmIK
from utils.axis_studio_bvh import AxisStudioFK
import time
import threading
import numpy as np

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hands_retargeting.retargeting_wrapper import HandRetargeterWrapper
from mocap_api import *
import logging

from pathlib import Path

logging.getLogger("yourdfpy").setLevel(logging.ERROR)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "hands"

CONFIG_PATH_RIGHT = PROJECT_ROOT / "src" / "hands_retargeting" / "configs" / "teleop" / "rh56dftp_right.yml"
CONFIG_PATH_LEFT = PROJECT_ROOT / "src" / "hands_retargeting" / "configs" / "teleop" / "rh56dftp_left.yml"

RIGHT_HAND_NODES = [
    "RightHand", "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "EndSiteRightHandThumb3",
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "EndSiteRightHandIndex3",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "EndSiteRightHandMiddle3",
    "RightHandRing1", "RightHandRing2", "RightHandRing3", "EndSiteRightHandRing3",
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3", "EndSiteRightHandPinky3",
]

LEFT_HAND_NODES = [
    "LeftHand", "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3", "EndSiteLeftHandThumb3",
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "EndSiteLeftHandIndex3",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "EndSiteLeftHandMiddle3",
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "EndSiteLeftHandRing3",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3", "EndSiteLeftHandPinky3",
]

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

left_arm = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
]
right_arm = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


class MocapRetargeter:
    def __init__(self):
        self.app = None
        self.running = False
        self.freq = 50
        
        self.arm_ik = G1_29_ArmIK(False)
        
        self.retargeter_right = HandRetargeterWrapper(
            config_path=CONFIG_PATH_RIGHT,
            joint_names=RIGHT_HAND_NODES,
            assets_dir=ASSETS_DIR,
        )
        self.retargeter_right.optimizer.opt.set_maxeval(5)
        
        self.retargeter_left = HandRetargeterWrapper(
            config_path=CONFIG_PATH_LEFT,
            joint_names=LEFT_HAND_NODES,
            assets_dir=ASSETS_DIR,
        )
        self.retargeter_left.optimizer.opt.set_maxeval(5)

        # Событие от mocap: последний известный кадр (Lock + переменная)
        self._latest_event = None
        self._event_lock = threading.Lock()

        # Результат ретаргета: последнее известное решение (Lock + переменная)
        self._latest_result = None
        self._result_lock = threading.Lock()

    def start(self, udp_port=7012):
        self.app = MCPApplication()
        settings = MCPSettings()
        settings.set_udp(udp_port)
        settings.set_bvh_rotation(MCPBvhRotation.XYZ)
        self.app.set_settings(settings)
        self.app.open()
        print(f"Mocap application initialized, UDP port: {udp_port}")
        
        self.running = True
        self.poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.retarget_thread = threading.Thread(target=self._retarget_loop, daemon=True)
        self.poll_thread.start()
        self.retarget_thread.start()

    def stop(self):
        self.running = False
        if self.app:
            self.app.close()
            print("Mocap application closed")

    def _poll_loop(self):
        while self.running:
            evts = self.app.poll_next_event()
            for evt in evts:
                if evt.event_type == MCPEventType.AvatarUpdated:
                    # Просто перезаписываем последний кадр. Если retarget не успел — он обработает его повторно
                    with self._event_lock:
                        self._latest_event = evt
            time.sleep(0.001)

    def _retarget_loop(self):
        while self.running:
            start_time = time.perf_counter()
            
            # Читаем последний известный evt. Если его еще не было — пропускаем итерацию
            with self._event_lock:
                evt = self._latest_event
            
            if evt is not None:
                self._process_avatar_data(evt)
            
            elapsed = time.perf_counter() - start_time
            if elapsed < 1.0 / self.freq:
                time.sleep(1.0 / self.freq - elapsed)

    def _process_avatar_data(self, evt):
        avatar = MCPAvatar(evt.event_data.avatar_handle)
        joints = avatar.get_joints()
        joints_dict = {j.get_name(): j for j in joints}
        
        bvh_frame = AxisStudioFK.compute_all_fk(joints_dict)
        
        q, _, q_bend = self.arm_ik.solve_ik_bvh_frame(bvh_frame, BendToBalance=True)
        q_r_hand = self.retargeter_right.retarget(bvh_frame)[[2, 6, 4, 0, 9, 8]]
        q_l_hand = self.retargeter_left.retarget(bvh_frame)[[2, 6, 4, 0, 9, 8]]

        arm_waist_dict = {}
        arm_waist_dict["waist_yaw_joint"] = float(q[0])
        arm_waist_dict["waist_pitch_joint"] = float(q_bend)
        arm_waist_dict["waist_roll_joint"] = float(0.0)
        
        for name, value in zip(left_arm, q[1:8]):
            arm_waist_dict[name] = float(value)
        for name, value in zip(right_arm, q[8:15]):
            arm_waist_dict[name] = float(value)

        with self._result_lock:
            self._latest_result = (arm_waist_dict, q_l_hand.tolist(), q_r_hand.tolist())

    def get_latest_retarget(self):
        """Возвращает последние данные. Можно вызывать сколько угодно раз, данные не удаляются."""
        with self._result_lock:
            return self._latest_result


if __name__ == "__main__":
    retargeter = MocapRetargeter()
    retargeter.start()

    print("Ретаргет запущен. Нажмите Ctrl+C для остановки.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nОстановка...")
        retargeter.stop()