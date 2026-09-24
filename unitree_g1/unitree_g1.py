import time

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

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


# Inspire hand

from RH56DFTP.RH56DFTP_TCP import RH56DFTP_TCP
from Register.RegisterKey.ftp_registers_keys import (
    POS_SET_0, POS_SET_1, POS_SET_2, POS_SET_3, POS_SET_4, POS_SET_5
)

# Порядок пальцев Inspire (совпадает с примером):
# [pinky, ring, middle, index, thumb_bend, thumb_rot]
HAND_POS_REGISTERS = [POS_SET_0, POS_SET_1, POS_SET_2, POS_SET_3, POS_SET_4, POS_SET_5]

HAND_MIN = 0
HAND_MAX = 1800     # полный ход по примеру


class InspireHandTCP:
    """Обёртка над RH56DFTP_TCP для одной кисти."""

    def __init__(self, host: str, port: int = 6000):
        self.client = RH56DFTP_TCP(host=host, port=port)

    def set_positions(self, positions_6):
        """positions_6: список из 6 значений 0..1800 в порядке
        [pinky, ring, middle, index, thumb_bend, thumb_rot]."""
        if len(positions_6) != 6:
            raise ValueError("нужно ровно 6 значений")
        for reg, val in zip(HAND_POS_REGISTERS, positions_6):
            v = int(max(HAND_MIN, min(HAND_MAX, val)))
            self.client.set(reg, v)

    def open(self):
        self.set_positions([0] * 6)

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass


class UnitreeG1:
    def __init__(self, ifname="enxc84d4427fee8", control_dt=0.02):
        if ifname is not None:
            ChannelFactoryInitialize(0, ifname)
        else:
            ChannelFactoryInitialize(0)

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.first_update = False
        self.crc = CRC()

        self.kp = 60.0
        self.kd = 1.5
        self.control_dt = control_dt

        self.publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.publisher.Init()

        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.subscriber.Init(self._low_state_handler, 10)

        # Hands:
        self.hand_r = InspireHandTCP(host="192.168.123.211", port=6000)
        self.hand_l = InspireHandTCP(host="192.168.123.210", port=6000) 

        #


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
        self._fill(self.target_pos_waist, WAIST_JOINTS, 100.0, self.kd)

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

    def shutdown(self):
        if self.thread is not None:
            self.thread.Stop()
            self.running = False
        try:
            self.hand_r.close()
            self.hand_l.close()
        except Exception:
            pass

    # Inpire hand API

    def set_hand_r(self, angles_r):
        """angles_r: 6 значений 0..1800 в порядке
        [pinky, ring, middle, index, thumb_bend, thumb_rot]."""
        self.hand_r.set_positions(angles_r)


    def set_hand_l(self, angles_l):
        self.hand_l.set_positions(angles_l)

    def __del__(self):
        self.shutdown()