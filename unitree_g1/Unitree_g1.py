import time
import numpy as np

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
    unitree_go_msg_dds__MotorCmd_,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
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


# ---------------- Inspire Hand DDS & TCP Controllers ----------------

kTopicInspireDFXCommand = "rt/inspire/cmd"
kTopicInspireDFXState   = "rt/inspire/state"

Inspire_Right_Hand_JointIndex = [0, 1, 2, 3, 4, 5]
Inspire_Left_Hand_JointIndex  = [6, 7, 8, 9, 10, 11]

# Limits in radians for RH56DFTP / DFX (from URDF)
# Canonical hardware order: [pinky, ring, middle, index, thumb_bend, thumb_rot]
INSPIRE_JOINT_LIMITS_MAX = np.array([1.4381, 1.4381, 1.4381, 1.4381, 0.5864, 1.1641], dtype=np.float64)


class InspireHandDDS:
    """
    DDS controller for dual Inspire dexterous hands.
    1-to-1 compatible with unitreerobotics/xr_teleoperate and unitree_sim_isaaclab.
    Publishes to DDS topic 'rt/inspire/cmd' using unitree_go.msg.dds_.MotorCmds_.
    """

    def __init__(self, cmd_topic: str = kTopicInspireDFXCommand, state_topic: str = kTopicInspireDFXState):
        self.cmd_topic = cmd_topic
        self.state_topic = state_topic

        self.publisher = ChannelPublisher(self.cmd_topic, MotorCmds_)
        self.publisher.Init()

        # Initialize 12 motor command slots (6 right, 6 left)
        self.hand_msg = MotorCmds_()
        self.hand_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(12)]

        # Cache target positions in radians
        self.target_q_r = np.zeros(6, dtype=np.float64)
        self.target_q_l = np.zeros(6, dtype=np.float64)

    def ctrl_dual_hand(self, left_q_target=None, right_q_target=None):
        """Set current left and right hand motor state target q in radians.
        1-to-1 matching Inspire_Controller_DFX.ctrl_dual_hand in xr_teleoperate.
        """
        if left_q_target is not None:
            self.target_q_l = np.asarray(left_q_target, dtype=np.float64)
            for idx, joint_id in enumerate(Inspire_Left_Hand_JointIndex):
                self.hand_msg.cmds[joint_id].q = float(self.target_q_l[idx])

        if right_q_target is not None:
            self.target_q_r = np.asarray(right_q_target, dtype=np.float64)
            for idx, joint_id in enumerate(Inspire_Right_Hand_JointIndex):
                self.hand_msg.cmds[joint_id].q = float(self.target_q_r[idx])

        self.publisher.Write(self.hand_msg)

    def set_hand_r(self, q_r):
        """Set target angles for right hand in radians. Auto-converts from registers if values > 3.14."""
        q = np.asarray(q_r, dtype=np.float64)
        if np.any(q > 3.14):
            q = np.clip(q / 1800.0, 0.0, 1.0) * INSPIRE_JOINT_LIMITS_MAX
        self.ctrl_dual_hand(left_q_target=None, right_q_target=q)

    def set_hand_l(self, q_l):
        """Set target angles for left hand in radians. Auto-converts from registers if values > 3.14."""
        q = np.asarray(q_l, dtype=np.float64)
        if np.any(q > 3.14):
            q = np.clip(q / 1800.0, 0.0, 1.0) * INSPIRE_JOINT_LIMITS_MAX
        self.ctrl_dual_hand(left_q_target=q, right_q_target=None)

    def open(self):
        """Open both hands."""
        self.ctrl_dual_hand(left_q_target=np.zeros(6), right_q_target=np.zeros(6))

    def close(self):
        """Close/shutdown DDS channel."""
        pass


class SingleHandDDSWrapper:
    """Provides set_positions() interface for a single hand bound to InspireHandDDS."""

    def __init__(self, dds_controller: InspireHandDDS, side: str):
        self.dds = dds_controller
        self.side = side.lower()

    def set_positions(self, angles_6):
        if self.side in ("r", "right"):
            self.dds.set_hand_r(angles_6)
        else:
            self.dds.set_hand_l(angles_6)

    def open(self):
        self.set_positions([0.0] * 6)

    def close(self):
        pass


# ---------------- Inspire Hand TCP (Modbus TCP) ----------------

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
HAND_MAX = 1800     # полный ход по примеру


class InspireHandTCP:
    """Обёртка над RH56DFTP_TCP для одной кисти."""

    def __init__(self, host: str, port: int = 6000):
        if not MODBUS_TCP_AVAILABLE:
            raise RuntimeError("plusml-rh56dftp package not found, cannot use InspireHandTCP.")
        self.client = RH56DFTP_TCP(host=host, port=port)

    def set_positions(self, positions_6):
        """positions_6: список из 6 значений (радианы или 0..1800 регистры) в порядке
        [pinky, ring, middle, index, thumb_bend, thumb_rot]."""
        if len(positions_6) != 6:
            raise ValueError("нужно ровно 6 значений")
        pos = np.asarray(positions_6, dtype=np.float64)
        if np.all(pos <= 3.14):
            # Input is in radians -> convert to 0..1800 registers
            pos = np.clip(pos / INSPIRE_JOINT_LIMITS_MAX, 0.0, 1.0) * 1800.0
        for reg, val in zip(HAND_POS_REGISTERS, pos):
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
    """
    Robot and dexterous hands controller.

    real: UnitreeG1(ifname="enxc84d4427fee8", hand_interface="dds")
          -> ChannelFactoryInitialize(0, ifname), arms: rt/arm_sdk, hands: rt/inspire/cmd

    sim:  UnitreeG1(sim=True)
          -> ChannelFactoryInitialize(1), arms: rt/lowcmd, hands: rt/inspire/cmd
    """

    def __init__(
        self,
        ifname="enxc84d4427fee8",
        control_dt=0.02,
        hand_interface="dds",   # "dds" (default, xr-teleoperate style) or "tcp" (Modbus TCP)
        sim=False,
        dds_channel=None,
        cmd_topic=None,
        state_topic="rt/lowstate",
        state_timeout=None,
    ):
        self.sim = sim
        self.control_dt = control_dt
        self.hand_interface = "dds" if sim else hand_interface

        # --- DDS Channel & Arm Topic ---
        if sim:
            channel = 1 if dds_channel is None else dds_channel
            ChannelFactoryInitialize(channel)
            topic = cmd_topic or "rt/lowcmd"
        else:
            channel = 0 if dds_channel is None else dds_channel
            if ifname is not None:
                ChannelFactoryInitialize(channel, ifname)
            else:
                ChannelFactoryInitialize(channel)
            topic = cmd_topic or "rt/arm_sdk"

        self.cmd_topic = topic
        print(f"[UnitreeG1] mode={'SIM' if sim else 'REAL'}, channel={channel}, cmd_topic={topic}, hand_interface={self.hand_interface}")

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.first_update = False
        self.crc = CRC()

        self.kp = 80.0
        self.kd = 1.5
        self.waist_kp = 350.0
        self.waist_kd = 2.5

        self.publisher = ChannelPublisher(topic, LowCmd_)
        self.publisher.Init()

        self.subscriber = ChannelSubscriber(state_topic, LowState_)
        self.subscriber.Init(self._low_state_handler, 10)

        # --- Hands ---
        if self.hand_interface == "dds":
            self.hand_dds = InspireHandDDS(cmd_topic=kTopicInspireDFXCommand, state_topic=kTopicInspireDFXState)
            self.hand_r = SingleHandDDSWrapper(self.hand_dds, side="right")
            self.hand_l = SingleHandDDSWrapper(self.hand_dds, side="left")
        else:
            self.hand_dds = None
            self.hand_r = InspireHandTCP(host="192.168.123.211", port=6000)
            self.hand_l = InspireHandTCP(host="192.168.123.210", port=6000)

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

        # Ожидание первого состояния
        t0 = time.time()
        while not self.first_update:
            if state_timeout is not None and time.time() - t0 > state_timeout:
                raise TimeoutError(
                    f"Не получен {state_topic} за {state_timeout} c "
                    f"(mode={'SIM' if sim else 'REAL'}, channel={channel})"
                )
            time.sleep(0.1)

        print("[UnitreeG1] received a low_state: robot connected")

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

        # Плавное включение/выключение arm_sdk (только на реальном роботе)
        if not self.sim:
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

    def shutdown(self):
        if self.thread is not None:
            self.thread.Stop()
            self.running = False
        try:
            self.hand_r.close()
            self.hand_l.close()
        except Exception:
            pass

    # Inspire hand API

    def open_hands(self):
        try:
            self.hand_r.open()
            self.hand_l.open()
        except Exception:
            pass

    def close_hands(self):
        try:
            self.hand_r.close()
            self.hand_l.close()
        except Exception:
            pass

    def set_hand_r(self, angles_r):
        """angles_r: 6 значений (радианы или 0..1800) в порядке
        [pinky, ring, middle, index, thumb_bend, thumb_rot]."""
        self.hand_r.set_positions(angles_r)

    def set_hand_l(self, angles_l):
        """angles_l: 6 значений (радианы или 0..1800) в порядке
        [pinky, ring, middle, index, thumb_bend, thumb_rot]."""
        self.hand_l.set_positions(angles_l)

    def __del__(self):
        self.shutdown()