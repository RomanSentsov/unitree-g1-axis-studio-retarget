import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
from pathlib import Path
import pybvh
import numpy as np
import tyro
import logging
logging.getLogger("yourdfpy").setLevel(logging.ERROR)

from hands_retargeting.retargeting_wrapper import HandRetargeterWrapper
from hands_retargeting.viser_wrapper import ViserHandsVisualizer
from utils.axis_studio_receiver import AxisStudioReceiver

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

def main(
    live: bool = False,
    udp_port: int = 7012,
    bvh_path: Path = BVH_FILEPATH,
):    
    # init retargeters
    print(">>> Initialising retargeters...")
    retargeter_right = HandRetargeterWrapper(
        config_path=CONFIG_PATH_RIGHT,
        joint_names=RIGHT_HAND_NODES,
        assets_dir=ASSETS_DIR,
    )
    retargeter_left = HandRetargeterWrapper(
        config_path=CONFIG_PATH_LEFT,
        joint_names=LEFT_HAND_NODES,
        assets_dir=ASSETS_DIR,
    )

    # viser
    visualizer = ViserHandsVisualizer(
        urdf_left=URDF_PATH_LEFT,
        urdf_right=URDF_PATH_RIGHT,
        dof_names_left=retargeter_left.dof_joint_names,
        dof_names_right=retargeter_right.dof_joint_names,
    )
    
    all_hand_nodes = RIGHT_HAND_NODES + LEFT_HAND_NODES
    
    if live:
        print(f">>> Starting live (Axis Studio UDP: {udp_port})...")
        with AxisStudioReceiver(joint_names=all_hand_nodes, udp_port=udp_port) as receiver:
            try:
                while True:
                    frame_dict = receiver.get_frame()
                    if frame_dict is None:
                        time.sleep(0.001)
                        continue

                    q_r = retargeter_right.retarget(frame_dict)
                    q_l = retargeter_left.retarget(frame_dict)

                    visualizer.update(q_l, q_r)

            except KeyboardInterrupt:
                print("\n>>> Stopped by user.")
    else:
        # load bvh
        print(f">>> Loading BVH: {BVH_FILEPATH}")
        bvh = pybvh.read_bvh_file(str(BVH_FILEPATH))
        bvh = bvh.reorient_world_up("+z")
        bvh = bvh.rotate_vertical(np.pi / 2)

        poses = bvh.node_positions(centered="skeleton")
        node_names = list(bvh.node_index.keys())
        total_frames = poses.shape[0]
        dt = getattr(bvh, "frame_time", 1.0 / 240.0)
        fps = int(round(1.0 / dt)) if dt > 0 else 60
        
        print(f">>> Viser started. Frames: {total_frames} | FPS: {fps}")

        frame_idx = 0
        try:
            while True:
                t_start = time.time()

                frame_dict = {
                    name: poses[frame_idx, bvh.node_index[name], :]
                    for name in node_names
                }

                q_r = retargeter_right.retarget(frame_dict)
                q_l = retargeter_left.retarget(frame_dict)

                visualizer.update(q_l, q_r)
                
                frame_idx = (frame_idx + 1) % total_frames

                elapsed = time.time() - t_start
                delay = dt - elapsed
                if delay > 0:
                    time.sleep(delay)

        except KeyboardInterrupt:
            print("\n>>> Stopped by user.")

if __name__ == "__main__":
    tyro.cli(main)