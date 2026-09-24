from pin_opti import G1_29_ArmIK
from utils.axis_studio_bvh import AxisStudioFK
import time

from hands_retargeting.retargeting_wrapper import HandRetargeterWrapper
from hands_retargeting.viser_wrapper import ViserHandsVisualizer

from mocap_api import *
import logging

logging.getLogger("yourdfpy").setLevel(logging.ERROR)

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

ASSETS_DIR = PROJECT_ROOT / "hands"
BVH_FILEPATH = PROJECT_ROOT / "recs" / "bvh_recs" / "bvh_misha_no_xyz_chr01_MAYA.bvh"

CONFIG_PATH_RIGHT = PROJECT_ROOT / "scripts" / "hands_retargeting" / "configs" / "teleop" / "rh56dftp_right.yml"
CONFIG_PATH_LEFT = PROJECT_ROOT / "scripts" / "hands_retargeting" / "configs" / "teleop" / "rh56dftp_left.yml"

URDF_PATH_RIGHT = ASSETS_DIR / "rh56dftp" / "rh56dftp_modified_right.urdf"
URDF_PATH_LEFT = ASSETS_DIR / "rh56dftp" / "rh56dftp_modified_left.urdf"

RIGHT_HAND_NODES = [
    "RightHand",
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "EndSiteRightHandThumb3",
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "EndSiteRightHandIndex3",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "EndSiteRightHandMiddle3",
    "RightHandRing1", "RightHandRing2", "RightHandRing3", "EndSiteRightHandRing3",
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3", "EndSiteRightHandPinky3",
]

LEFT_HAND_NODES = [
    "LeftHand",
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3", "EndSiteLeftHandThumb3",
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "EndSiteLeftHandIndex3",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "EndSiteLeftHandMiddle3",
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "EndSiteLeftHandRing3",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3", "EndSiteLeftHandPinky3",
]

# Application class
class MocapAxisDemo:

    def __init__(self):

        self.app = None
        self.running = False

        self.prev_posture_time_ms = None

        # init arm ik (retargeter)
        self.arm_ik = G1_29_ArmIK(Unit_Test=True, Visualization=True)

        print(">>> Initialising retargeters...")
        self.retargeter_right = HandRetargeterWrapper(
            config_path=CONFIG_PATH_RIGHT,
            joint_names=RIGHT_HAND_NODES,
            assets_dir=ASSETS_DIR,
        )
        self.retargeter_left = HandRetargeterWrapper(
            config_path=CONFIG_PATH_LEFT,
            joint_names=LEFT_HAND_NODES,
            assets_dir=ASSETS_DIR,
        )

        # viser
        self.visualizer = ViserHandsVisualizer(
            urdf_left=URDF_PATH_LEFT,
            urdf_right=URDF_PATH_RIGHT,
            dof_names_left=self.retargeter_left.dof_joint_names,
            dof_names_right=self.retargeter_right.dof_joint_names,
        )

        self.all_hand_nodes = RIGHT_HAND_NODES + LEFT_HAND_NODES


        self.debug_printed = False


    def start(self, udp_port=7012):

        self.app = MCPApplication()
        settings = MCPSettings()
        settings.set_udp(udp_port)
        # Should be coherent with Axis Studio BVH stream settings
        settings.set_bvh_rotation(
            MCPBvhRotation.XYZ
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

        # print("received")

        # event data
        avatar = MCPAvatar(evt.event_data.avatar_handle)

        # joint data
        joints = avatar.get_joints()

        current_time_ms = time.time() * 1000.0 # TODO change to time from joint_data

        # limit retarget to 30 hz. Can be omited when Visualization is off
        if self.prev_posture_time_ms is not None:

            delta_ms = (
                current_time_ms
                - self.prev_posture_time_ms
            )

            if delta_ms < 100.0:
                return

        self.prev_posture_time_ms = current_time_ms

        # joint names
        joints_dict = {
            j.get_name(): j
            for j in joints
        }

        _cache = {}

        # Retarget
        bvh_frame = {name: AxisStudioFK.get_global_transform(joints_dict, name, relative_to_hips=True, _cache=_cache)[0] for name in AxisStudioFK.NODE_NAMES}
        until = time.time()

        # Solve ik in this block
        q, _ = self.arm_ik.solve_ik_bvh_frame(bvh_frame)
        q_r_hand = self.retargeter_right.retarget(bvh_frame)
        q_l_hand = self.retargeter_left.retarget(bvh_frame)
        print(q_r_hand)

        self.visualizer.update(q_l_hand, q_r_hand)

        elapsed = time.time() - until
        print(f"Whole retarget solved in: {elapsed * 1000.0:.2f} ms")


if __name__ == "__main__":

    demo = MocapAxisDemo()
    print("Starting Mocap Axis Studio demo...")
    demo.start()