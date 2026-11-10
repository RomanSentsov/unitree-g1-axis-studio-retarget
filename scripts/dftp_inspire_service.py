#!/usr/bin/env python3
"""
Inspire Hand RH56DFTP to Unitree DDS Bridge Service.

Drop-in replacement for Unitree's official 'dfx_inspire_service' for robots
equipped with Ethernet/Modbus TCP hands (RH56DFTP) instead of Serial/RS485 (RH56DFX).

Architecture:
  xr-teleoperate / teleop_ros2 / policy
                  │
                  ▼ DDS (rt/inspire/cmd, unitree_go.msg.dds_.MotorCmds_)
       [dftp_inspire_service]  <── this daemon
                  │
                  ▼ Modbus TCP (port 6000)
    RH56DFTP Left (192.168.123.210) & Right (192.168.123.211)

DDS Joint Mapping (12 motors total):
  0..5:  Right Hand [pinky, ring, middle, index, thumb_bend, thumb_rot]
  6..11: Left Hand  [pinky, ring, middle, index, thumb_bend, thumb_rot]
"""

import sys
import os
import time
import signal
import threading
import numpy as np
import tyro

# Add repo root to sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorState_

from unitree_g1.Unitree_g1 import (
    InspireHandTCP,
    INSPIRE_JOINT_LIMITS_MAX,
    kTopicInspireDFXCommand,
    kTopicInspireDFXState,
    Inspire_Right_Hand_JointIndex,
    Inspire_Left_Hand_JointIndex,
)


class InspireDftpBridge:
    def __init__(
        self,
        ifname: str = "enxc84d4427fee8",
        channel: int = 0,
        r_host: str = "192.168.123.211",
        l_host: str = "192.168.123.210",
        port: int = 6000,
        frequency: float = 50.0,
        cmd_topic: str = kTopicInspireDFXCommand,
        state_topic: str = kTopicInspireDFXState,
    ):
        self.r_host = r_host
        self.l_host = l_host
        self.port = port
        self.frequency = frequency
        self.period = 1.0 / frequency
        self.cmd_topic = cmd_topic
        self.state_topic = state_topic

        # Target angles in radians [pinky, ring, middle, index, thumb_bend, thumb_rot]
        self.target_q_r = np.zeros(6, dtype=np.float64)
        self.target_q_l = np.zeros(6, dtype=np.float64)
        self.has_new_cmd_r = False
        self.has_new_cmd_l = False
        self.lock = threading.Lock()
        self.running = False

        # 1. Connect to TCP Hands
        print(f"[InspireDftpBridge] Connecting to Modbus TCP hands: Right={r_host}:{port}, Left={l_host}:{port}...")
        self.hand_r = None
        self.hand_l = None
        try:
            self.hand_r = InspireHandTCP(host=r_host, port=port)
            print(f"[InspireDftpBridge] Right hand connected ({r_host}:{port})")
        except Exception as e:
            print(f"[InspireDftpBridge] WARNING: Could not connect to right hand ({r_host}): {e}")

        try:
            self.hand_l = InspireHandTCP(host=l_host, port=port)
            print(f"[InspireDftpBridge] Left hand connected ({l_host}:{port})")
        except Exception as e:
            print(f"[InspireDftpBridge] WARNING: Could not connect to left hand ({l_host}): {e}")

        # 2. Initialize DDS
        print(f"[InspireDftpBridge] Initializing DDS channel {channel} on ifname '{ifname}'...")
        if ifname:
            ChannelFactoryInitialize(channel, ifname)
        else:
            ChannelFactoryInitialize(channel)

        self.subscriber = ChannelSubscriber(self.cmd_topic, MotorCmds_)
        self.subscriber.Init(self._on_dds_cmd, 10)
        print(f"[InspireDftpBridge] Subscribed to DDS command topic: '{self.cmd_topic}'")

        self.publisher = ChannelPublisher(self.state_topic, MotorStates_)
        self.publisher.Init()
        print(f"[InspireDftpBridge] Publishing DDS state topic: '{self.state_topic}'")

        # Pre-allocate state message (12 motors)
        self.state_msg = MotorStates_()
        self.state_msg.states = [unitree_go_msg_dds__MotorState_() for _ in range(12)]

    def _on_dds_cmd(self, msg: MotorCmds_):
        """DDS callback triggered when MotorCmds_ is received on rt/inspire/cmd."""
        if len(msg.cmds) < 12:
            return

        with self.lock:
            # Right hand: indices 0..5
            for i, joint_id in enumerate(Inspire_Right_Hand_JointIndex):
                self.target_q_r[i] = float(msg.cmds[joint_id].q)
            self.has_new_cmd_r = True

            # Left hand: indices 6..11
            for i, joint_id in enumerate(Inspire_Left_Hand_JointIndex):
                self.target_q_l[i] = float(msg.cmds[joint_id].q)
            self.has_new_cmd_l = True

    def start(self):
        """Main loop: pushes latest targets to Modbus TCP and publishes state feedback."""
        self.running = True
        print(f"[InspireDftpBridge] Service started at {self.frequency:.1f} Hz. Press Ctrl+C to stop.")

        last_sent_r = None
        last_sent_l = None

        while self.running:
            t0 = time.perf_counter()

            # Copy latest targets under lock
            with self.lock:
                q_r = self.target_q_r.copy() if self.has_new_cmd_r else None
                q_l = self.target_q_l.copy() if self.has_new_cmd_l else None
                self.has_new_cmd_r = False
                self.has_new_cmd_l = False

            # Send to Right Hand if new
            if q_r is not None and self.hand_r is not None:
                if last_sent_r is None or np.max(np.abs(q_r - last_sent_r)) > 1e-4:
                    try:
                        self.hand_r.set_positions(q_r)
                        last_sent_r = q_r.copy()
                    except Exception as e:
                        print(f"[InspireDftpBridge] Error sending to right hand: {e}")

            # Send to Left Hand if new
            if q_l is not None and self.hand_l is not None:
                if last_sent_l is None or np.max(np.abs(q_l - last_sent_l)) > 1e-4:
                    try:
                        self.hand_l.set_positions(q_l)
                        last_sent_l = q_l.copy()
                    except Exception as e:
                        print(f"[InspireDftpBridge] Error sending to left hand: {e}")

            # Publish feedback state to rt/inspire/state
            # (Matches xr_teleoperate conventions: current commanded angles are reflected back)
            for i, joint_id in enumerate(Inspire_Right_Hand_JointIndex):
                self.state_msg.states[joint_id].q = float(self.target_q_r[i])
            for i, joint_id in enumerate(Inspire_Left_Hand_JointIndex):
                self.state_msg.states[joint_id].q = float(self.target_q_l[i])
            self.publisher.Write(self.state_msg)

            # Sleep remainder of period
            elapsed = time.perf_counter() - t0
            sleep_time = self.period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self):
        print("\n[InspireDftpBridge] Stopping service and opening hands...")
        self.running = False
        if self.hand_r is not None:
            try:
                self.hand_r.open()
                self.hand_r.close()
            except Exception:
                pass
        if self.hand_l is not None:
            try:
                self.hand_l.open()
                self.hand_l.close()
            except Exception:
                pass
        print("[InspireDftpBridge] Stopped.")


def main(
    ifname: str = "enxc84d4427fee8",
    channel: int = 0,
    r_host: str = "192.168.123.211",
    l_host: str = "192.168.123.210",
    port: int = 6000,
    frequency: float = 50.0,
):
    """Launch the RH56DFTP Modbus TCP to Unitree DDS bridge service.

    Args:
        ifname: Network interface for DDS communication (default wired interface to G1).
        channel: Unitree DDS channel (0 for real robot).
        r_host: IP address of the Right RH56DFTP hand.
        l_host: IP address of the Left RH56DFTP hand.
        port: Modbus TCP port on the hands (default: 6000).
        frequency: Update rate in Hz for pushing commands to Modbus TCP sockets.
    """
    bridge = InspireDftpBridge(
        ifname=ifname,
        channel=channel,
        r_host=r_host,
        l_host=l_host,
        port=port,
        frequency=frequency,
    )

    def sig_handler(sig, frame):
        bridge.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    bridge.start()


if __name__ == "__main__":
    tyro.cli(main)
