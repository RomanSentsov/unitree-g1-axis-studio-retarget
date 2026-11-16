import time
import sys, os
import numpy as np

parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent2_dir)

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
from utils.wav import read_wav, play_pcm_stream

WAV_EP_FINISHED = "audio/episode_saved.wav"
WAV_EP_STARTED = "audio/episode_started.wav"

class G1JointIndex:
    # Left leg
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleB = 4
    LeftAnkleRoll = 5
    LeftAnkleA = 5

    # Right leg
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleB = 10
    RightAnkleRoll = 11
    RightAnkleA = 11

    WaistYaw = 12
    WaistRoll = 13
    WaistA = 13
    WaistPitch = 14
    WaistB = 14

    # Left arm
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21

    # Right arm
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28

    kNotUsedJoint = 29


LEFT_ARM_JOINTS = [
    G1JointIndex.LeftShoulderPitch,
    G1JointIndex.LeftShoulderRoll,
    G1JointIndex.LeftShoulderYaw,
    G1JointIndex.LeftElbow,
    G1JointIndex.LeftWristRoll,
    G1JointIndex.LeftWristPitch,
    G1JointIndex.LeftWristYaw,
]

RIGHT_ARM_JOINTS = [
    G1JointIndex.RightShoulderPitch,
    G1JointIndex.RightShoulderRoll,
    G1JointIndex.RightShoulderYaw,
    G1JointIndex.RightElbow,
    G1JointIndex.RightWristRoll,
    G1JointIndex.RightWristPitch,
    G1JointIndex.RightWristYaw,
]

WAIST_JOINTS = [
    G1JointIndex.WaistYaw,
    G1JointIndex.WaistRoll,
    G1JointIndex.WaistPitch,
]


# ---------------- Inspire Hand Constants & DDS Wrapper ----------------

try:
    from RH56DFTP.RH56DFTP_TCP import RH56DFTP_TCP
    from Register.RegisterKey.ftp_registers_keys import (
        POS_SET_0, POS_SET_1, POS_SET_2, POS_SET_3, POS_SET_4, POS_SET_5
    )
    HAND_POS_REGISTERS = [POS_SET_0, POS_SET_1, POS_SET_2, POS_SET_3, POS_SET_4, POS_SET_5]
    MODBUS_TCP_AVAILABLE = True
except ImportError:
    MODBUS_TCP_AVAILABLE = False
    HAND_POS_REGISTERS = list(range(6))

HAND_MIN = 0
HAND_MAX = 1800

kTopicInspireCommand = "rt/inspire/cmd"
kTopicInspireState   = "rt/inspire/state"

Inspire_Right_Hand_JointIndex = [0, 1, 2, 3, 4, 5]
Inspire_Left_Hand_JointIndex  = [6, 7, 8, 9, 10, 11]

# Limits in radians for Inspire RH56DFTP (URDF order: [pinky, ring, middle, index, thumb_bend, thumb_rot])
INSPIRE_JOINT_LIMITS_MAX = np.array([1.4381, 1.4381, 1.4381, 1.4381, 0.5864, 1.1641], dtype=np.float64)


class SingleHandDDSWrapper:
    """Compatibility wrapper preserving .open(), .close(), .set_positions() interface."""

    def __init__(self, g1, side: str):
        self.g1 = g1
        self.side = side.lower()

    def set_positions(self, positions_6):
        if self.side in ("r", "right"):
            self.g1.set_hand_r(positions_6)
        else:
            self.g1.set_hand_l(positions_6)

    def open(self):
        if self.side in ("r", "right"):
            self.g1.open_hand_r()
        else:
            self.g1.open_hand_l()

    def close(self):
        pass


class UnitreeG1:

    # TODO: add --sim argument that makes for /lowcmd topic when domain set to 1 (but not to 0!!!)
    def __init__(self, domain_id=0, ifname="enxc84d4427fee8", control_dt=0.02):
        if ifname is not None:
            ChannelFactoryInitialize(domain_id, ifname)
        else:
            ChannelFactoryInitialize(domain_id)

        self.audioClient = AudioClient()
        self.audioClient.Init()
        self.audioClient.SetTimeout(4.0)
        self.audioClient.SetVolume(70)

        self.ep_started_audio, sample_rate, num_channels, is_ok = read_wav(WAV_EP_STARTED)
        if not is_ok or sample_rate != 16000 or num_channels != 1:
            print("[ERROR] WAV must be 16kHz mono")
        self.ep_finished_audio, sample_rate, num_channels, is_ok = read_wav(WAV_EP_FINISHED)
        if not is_ok or sample_rate != 16000 or num_channels != 1:
            print("[ERROR] WAV must be 16kHz mono")

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.first_update = False
        self.crc = CRC()

        self.kp = 80.0
        self.kd = 1.5
        self.control_dt = control_dt

        self.publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.publisher.Init()

        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.subscriber.Init(self._low_state_handler, 10)

        # Hands DDS (rt/inspire/cmd, rt/inspire/state)
        self.hand_cmd_pub = ChannelPublisher(kTopicInspireCommand, MotorCmds_)
        self.hand_cmd_pub.Init()

        self.hand_msg = MotorCmds_()
        self.hand_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(12)]

        self.hand_state_sub = ChannelSubscriber(kTopicInspireState, MotorStates_)
        self.hand_state_sub.Init(self._hand_state_handler, 10)

        self.target_hand_r = [0.0] * 6
        self.target_hand_l = [0.0] * 6
        self.latest_hand_state_r = [0.0] * 6
        self.latest_hand_state_l = [0.0] * 6
        self.has_hand_state = False

        # Physical Modbus TCP hands (transparent hardware bridge for RH56DFTP on real robot)
        self.tcp_hand_r = None
        self.tcp_hand_l = None
        if MODBUS_TCP_AVAILABLE:
            try:
                self.tcp_hand_r = RH56DFTP_TCP(host="192.168.123.211", port=6000)
            except Exception:
                pass
            try:
                self.tcp_hand_l = RH56DFTP_TCP(host="192.168.123.210", port=6000)
            except Exception:
                pass

        self.hand_r = SingleHandDDSWrapper(self, "right")
        self.hand_l = SingleHandDDSWrapper(self, "left") 

        # Целевые позиции — все три группы устроены одинаково
        self.target_pos_l = None       # 7 значений
        self.target_pos_r = None       # 7 значений
        self.target_pos_waist = None   # 3 значения

        # Плавное включение/выключение arm_sdk
        self.enable_current = 0.0
        self.enable_target = 0.0
        self.enable_start_time = None
        self.enable_start_val = 0.0
        self.enable_duration = 2.0

        self.thread = None
        self.running = False

        while not self.first_update:
            time.sleep(0.1)

        # Стартовые цели = текущее положение
        self.target_pos_l = [self.low_state.motor_state[j].q for j in LEFT_ARM_JOINTS]
        self.target_pos_r = [self.low_state.motor_state[j].q for j in RIGHT_ARM_JOINTS]
        self.target_pos_waist = [self.low_state.motor_state[j].q for j in WAIST_JOINTS]

        self.thread = RecurrentThread(
            interval=self.control_dt,
            target=self._low_cmd_write,
            name="g1_control",
        )
        self.thread.Start()
        self.running = True

    def _low_state_handler(self, msg: LowState_):
        self.low_state = msg
        if not self.first_update:
            self.first_update = True

    def _hand_state_handler(self, msg: MotorStates_):
        if len(msg.states) >= 12:
            self.latest_hand_state_r = [float(msg.states[j].q) for j in Inspire_Right_Hand_JointIndex]
            self.latest_hand_state_l = [float(msg.states[j].q) for j in Inspire_Left_Hand_JointIndex]
            self.has_hand_state = True

    def _low_cmd_write(self):
        if not self.first_update:
            return

        # Плавное включение/выключение arm_sdk
        if self.enable_current != self.enable_target:
            if self.enable_start_time is None:
                self.enable_start_time = time.time()
                self.enable_start_val = self.enable_current
            elapsed = time.time() - self.enable_start_time
            ratio = min(elapsed / self.enable_duration, 1.0)
            self.enable_current = self.enable_start_val + (self.enable_target - self.enable_start_val) * ratio
            if ratio >= 1.0:
                self.enable_start_time = None
        self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = self.enable_current

        # Руки и торс — одинаково
        self._fill(self.target_pos_l, LEFT_ARM_JOINTS, self.kp, self.kd)
        self._fill(self.target_pos_r, RIGHT_ARM_JOINTS, self.kp, self.kd)
        self._fill(self.target_pos_waist, WAIST_JOINTS, 350.0, 2.5)

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)

    def _fill(self, target, joints, kp, kd):
        if target is None:
            return
        for i, joint in enumerate(joints):
            self.low_cmd.motor_cmd[joint].q = target[i]
            self.low_cmd.motor_cmd[joint].dq = 0.0
            self.low_cmd.motor_cmd[joint].tau = 0.0
            self.low_cmd.motor_cmd[joint].kp = kp
            self.low_cmd.motor_cmd[joint].kd = kd

    # ---- Публичный API ----

    def set_arm_l(self, q):
        if len(q) != 7:
            raise ValueError("set_arm_l: нужно 7 значений")
        self.target_pos_l = list(q)

    def set_arm_r(self, q):
        if len(q) != 7:
            raise ValueError("set_arm_r: нужно 7 значений")
        self.target_pos_r = list(q)

    def set_waist(self, q):
        '''
        Args: (waist_yaw(== rot around z), waist_roll, waist_pitch)
        '''
        if len(q) != 3:
            raise ValueError("set_waist: нужно 3 значения")
        self.target_pos_waist = list(q)

    def enable_arm_sdk(self, duration=2.0):
        self.enable_target = 1.0
        self.enable_duration = duration
        self.enable_start_time = None
        self.enable_start_val = self.enable_current

    def disable_arm_sdk(self, duration=2.0):
        self.enable_target = 0.0
        self.enable_duration = duration
        self.enable_start_time = None
        self.enable_start_val = self.enable_current

    def say_ep_start(self):
        play_pcm_stream(self.audioClient, self.ep_started_audio, "start episode")

    def say_ep_saved(self):
        play_pcm_stream(self.audioClient, self.ep_finished_audio, "start saved")

    def shutdown(self):
        if self.thread is not None:
            self.thread.Stop()
            self.running = False
        try:
            self.open_hands()
        except Exception:
            pass
        if self.tcp_hand_r is not None:
            try:
                self.tcp_hand_r.close()
            except Exception:
                pass
        if self.tcp_hand_l is not None:
            try:
                self.tcp_hand_l.close()
            except Exception:
                pass

    # ---- Inspire Hand API ----

    def set_hand_r(self, angles_r):
        """angles_r: 6 joint values in radians in canonical order:
        [pinky, ring, middle, index, thumb_bend, thumb_rot]."""
        if len(angles_r) != 6:
            raise ValueError("set_hand_r: нужно ровно 6 значений")
        q = np.asarray(angles_r, dtype=np.float64)
        if np.any(q > 3.14):
            q = np.clip(q / 1800.0, 0.0, 1.0) * INSPIRE_JOINT_LIMITS_MAX
        else:
            q = np.clip(q, 0.0, INSPIRE_JOINT_LIMITS_MAX)

        self.target_hand_r = q.tolist()

        # 1. DDS (simulation / xr_teleoperate standard)
        for idx, joint_id in enumerate(Inspire_Right_Hand_JointIndex):
            self.hand_msg.cmds[joint_id].q = float(q[idx])
        self.hand_cmd_pub.Write(self.hand_msg)

        # 2. Modbus TCP (physical RH56DFTP hardware on real robot)
        if self.tcp_hand_r is not None:
            try:
                reg_vals = np.clip(q / INSPIRE_JOINT_LIMITS_MAX, 0.0, 1.0) * 1800.0
                for reg, val in zip(HAND_POS_REGISTERS, reg_vals):
                    self.tcp_hand_r.set(reg, int(val))
            except Exception:
                pass

    def set_hand_l(self, angles_l):
        """angles_l: 6 joint values in radians in canonical order:
        [pinky, ring, middle, index, thumb_bend, thumb_rot]."""
        if len(angles_l) != 6:
            raise ValueError("set_hand_l: нужно ровно 6 значений")
        q = np.asarray(angles_l, dtype=np.float64)
        if np.any(q > 3.14):
            q = np.clip(q / 1800.0, 0.0, 1.0) * INSPIRE_JOINT_LIMITS_MAX
        else:
            q = np.clip(q, 0.0, INSPIRE_JOINT_LIMITS_MAX)

        self.target_hand_l = q.tolist()

        # 1. DDS (simulation / xr_teleoperate standard)
        for idx, joint_id in enumerate(Inspire_Left_Hand_JointIndex):
            self.hand_msg.cmds[joint_id].q = float(q[idx])
        self.hand_cmd_pub.Write(self.hand_msg)

        # 2. Modbus TCP (physical RH56DFTP hardware on real robot)
        if self.tcp_hand_l is not None:
            try:
                reg_vals = np.clip(q / INSPIRE_JOINT_LIMITS_MAX, 0.0, 1.0) * 1800.0
                for reg, val in zip(HAND_POS_REGISTERS, reg_vals):
                    self.tcp_hand_l.set(reg, int(val))
            except Exception:
                pass

    def get_hand_r(self) -> list[float]:
        """Returns current right hand joint angles in radians:
        [pinky, ring, middle, index, thumb_bend, thumb_rot]."""
        if self.has_hand_state:
            return list(self.latest_hand_state_r)
        return list(self.target_hand_r)

    def get_hand_l(self) -> list[float]:
        """Returns current left hand joint angles in radians:
        [pinky, ring, middle, index, thumb_bend, thumb_rot]."""
        if self.has_hand_state:
            return list(self.latest_hand_state_l)
        return list(self.target_hand_l)

    def open_hand_r(self):
        self.set_hand_r([0.0] * 6)

    def open_hand_l(self):
        self.set_hand_l([0.0] * 6)

    def open_hands(self):
        self.open_hand_r()
        self.open_hand_l()

    def __del__(self):
        self.shutdown()