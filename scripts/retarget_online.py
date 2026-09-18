from scipy.spatial.transform import Rotation as R
from pin_opti import G1_29_ArmIK
import pinocchio as pin
import numpy as np
import time

from utils.axis_studio_bvh import AxisStudioFK

from mocap_api import *

# Application class
class MocapAxisDemo:

    def __init__(self):

        self.app = None
        self.running = False

        self.prev_posture_time_ms = None

        # init arm ik (retargeter)
        self.arm_ik = G1_29_ArmIK(
            Unit_Test=True,
            Visualization=True
        )

        # Joint names
        self.r_hand_name = "RightHand"
        self.l_hand_name = "LeftHand"

        self.r_mid_name = "RightHandMiddle1"
        self.l_mid_name = "LeftHandMiddle1"

        self.r_pinky_name = "RightHandPinky1"
        self.l_pinky_name = "LeftHandPinky1"

        self.r_elbow_name = "RightForeArm"
        self.l_elbow_name = "LeftForeArm"

        # Positions
        self.r_hand_pos = None
        self.l_hand_pos = None

        self.r_mid_pos = None
        self.l_mid_pos = None

        self.r_pinky_pos = None
        self.l_pinky_pos = None

        self.r_elbow_pos = None
        self.l_elbow_pos = None

        self.debug_printed = False


    def start(self, udp_port=7012):

        self.app = MCPApplication()
        settings = MCPSettings()
        settings.set_udp(udp_port)
        # Should be coherent with Axis Studio BVH stream settings
        settings.set_bvh_rotation(
            MCPBvhRotation.YXZ
        )
        self.app.set_settings(settings)
        self.app.open()
        print(
            f"Mocap application initialized, "
            f"UDP port: {udp_port}"
        )
        self.running = True

        try:
            while self.running:
                evts = self.app.poll_next_event()
                for evt in evts:

                    if evt.event_type == MCPEventType.AvatarUpdated:
                        self._handle_avatar_data(evt)

        except KeyboardInterrupt:
            print("Program interrupted by user")
        finally:
            self.stop()

    def stop(self):
        self.running = False
        if self.app:
            self.app.close()
            print("Mocap application closed")

    def _handle_avatar_data(self, evt):

        avatar = MCPAvatar(evt.event_data.avatar_handle)

        joints = avatar.get_joints()

        current_time_ms = time.time() * 1000.0

        # limit retarget to 30 hz. Can be omited when Visualization is off
        if self.prev_posture_time_ms is not None:

            delta_ms = (
                current_time_ms
                - self.prev_posture_time_ms
            )

            if delta_ms < 33.0:
                return

        self.prev_posture_time_ms = current_time_ms

        # joint names
        joints_dict = {
            j.get_name(): j
            for j in joints
        }

        _cache = {}

        self.r_hand_pos, self.r_hand_quat = AxisStudioFK.get_global_transform(joints_dict, self.r_hand_name, relative_to_hips=True, _cache=_cache)
        self.l_hand_pos, self.l_hand_quat = AxisStudioFK.get_global_transform(joints_dict, self.l_hand_name, relative_to_hips=True, _cache=_cache)

        self.r_mid_pos, _ = AxisStudioFK.get_global_transform(joints_dict, self.r_mid_name, relative_to_hips=True, _cache=_cache)
        self.l_mid_pos, _ = AxisStudioFK.get_global_transform(joints_dict, self.l_mid_name, relative_to_hips=True, _cache=_cache)

        self.r_pinky_pos, _ = AxisStudioFK.get_global_transform(joints_dict, self.r_pinky_name, relative_to_hips=True, _cache=_cache)
        self.l_pinky_pos, _ = AxisStudioFK.get_global_transform(joints_dict, self.l_pinky_name, relative_to_hips=True, _cache=_cache)

        self.r_elbow_pos, _ = AxisStudioFK.get_global_transform(joints_dict, self.r_elbow_name, relative_to_hips=True, _cache=_cache)
        self.l_elbow_pos, _ = AxisStudioFK.get_global_transform(joints_dict, self.l_elbow_name, relative_to_hips=True, _cache=_cache)

        # HAND ORIENTATION

        r_hand_x = (self.r_mid_pos - self.r_hand_pos)
        r_hand_y = (self.r_pinky_pos - self.r_mid_pos)

        l_hand_x = (self.l_mid_pos - self.l_hand_pos)
        l_hand_y = (self.l_pinky_pos - self.l_mid_pos)

        # aligning x and y axis of a hand with x and z of the frame of DFTP in urdf  
        r_hand_rot = R.align_vectors(np.stack([r_hand_x, r_hand_y], axis=0), np.array([1, 0, 0], [0, 0, -1]))[0]
        l_hand_rot = R.align_vectors(np.stack([l_hand_x, l_hand_y], axis=0), np.array([1, 0, 0], [0, 0, -1]))[0]

        r_quat = r_hand_rot.as_quat()
        l_quat = l_hand_rot.as_quat()

        # Convert points in cpin format

        R_tf_target = pin.SE3(
            pin.Quaternion(r_quat),
            self.r_hand_pos / 150.0,
        )

        L_tf_target = pin.SE3(
            pin.Quaternion(l_quat),
            self.l_hand_pos / 150.0,
        )

        R_tf_elbow_target = pin.SE3(
            pin.Quaternion(1, 0, 0, 0),
            self.r_elbow_pos / 150.0,
        )

        L_tf_elbow_target = pin.SE3(
            pin.Quaternion(1, 0, 0, 0),
            self.l_elbow_pos / 150.0,
        )

        # Retarget

        until = time.time()

        self.arm_ik.solve_ik(L_tf_target.homogeneous, R_tf_target.homogeneous, L_tf_elbow_target.homogeneous, R_tf_elbow_target.homogeneous)

        elapsed = time.time() - until

        print(f"Retarget solved in: {elapsed * 1000.0:.2f} ms")


if __name__ == "__main__":

    demo = MocapAxisDemo()
    print("Starting Mocap Axis Studio demo...")
    demo.start()