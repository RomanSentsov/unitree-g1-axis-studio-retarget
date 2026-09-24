from pathlib import Path
import warnings
import numpy as np

from hands_retargeting.retargeting_config import RetargetingConfig
R_hamer2urdf = np.array([
    [-1, 0, 0],
    [ 0,-1, 0],
    [ 0, 0, 1] 
])

END_SITE_MAP = {
    "EndSiteRightHandThumb3": ("RightHandThumb3", "RightHandThumb2"),
    "EndSiteRightHandIndex3": ("RightHandIndex3", "RightHandIndex2"),
    "EndSiteRightHandMiddle3": ("RightHandMiddle3", "RightHandMiddle2"),
    "EndSiteRightHandRing3": ("RightHandRing3", "RightHandRing2"),
    "EndSiteRightHandPinky3": ("RightHandPinky3", "RightHandPinky2"),
    "EndSiteLeftHandThumb3": ("LeftHandThumb3", "LeftHandThumb2"),
    "EndSiteLeftHandIndex3": ("LeftHandIndex3", "LeftHandIndex2"),
    "EndSiteLeftHandMiddle3": ("LeftHandMiddle3", "LeftHandMiddle2"),
    "EndSiteLeftHandRing3": ("LeftHandRing3", "LeftHandRing2"),
    "EndSiteLeftHandPinky3": ("LeftHandPinky3", "LeftHandPinky2"),
}

def get_hand_frame(keypoint_3d_array: np.ndarray) -> np.ndarray:
    """
    Originates from dex-retargeting repo
    
    Compute the 3D coordinate frame (orientation only) from detected 3d key points
    :param points: keypoints detected with HaMeR. Order: [wrist, index, middle, pinky]
    :return: the coordinate frame of wrist in MANO convention
    """
    assert keypoint_3d_array.shape == (21, 3)
    points = keypoint_3d_array[[0, 5, 9], :]

    # Compute vector from palm to the first joint of middle finger
    x_vector = points[0] - points[2]

    # Normal fitting with SVD
    points = points - np.mean(points, axis=0, keepdims=True)
    u, s, v = np.linalg.svd(points)

    normal = v[2, :]

    # Gram–Schmidt Orthonormalize
    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / np.linalg.norm(x)
    z = np.cross(x, normal)

    # We assume that the vector from pinky to index is similar the z axis in MANO convention
    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1
        z *= -1
    frame = np.stack([x, normal, z], axis=1)
    return frame

class HandRetargeterWrapper:
    """
    BVH data-> robot joint positions
    """

    def __init__(
        self,
        config_path: str | Path,
        joint_names: list[str],
        assets_dir: str | Path | None = None,
        scale: float = 0.01,
        r_hamer2urdf: np.ndarray | None = None,
    ):
        self.joint_names = list(joint_names)
        assert len(self.joint_names) == 21, f">>> Expected  21 joints, get {len(self.joint_names)}"

        self.scale = scale
        self.r_hamer2urdf = R_hamer2urdf

        if assets_dir is not None:
            RetargetingConfig.set_default_urdf_dir(str(assets_dir))

        self.retargeter = RetargetingConfig.load_from_file(str(config_path)).build()
        self.optimizer = self.retargeter.optimizer

        self.dof_joint_names = list(self.optimizer.robot.dof_joint_names)
        self.num_dofs = len(self.dof_joint_names)

        # last successful position cache
        self.last_qpos: np.ndarray = np.zeros(self.num_dofs, dtype=np.float64)
        self.has_valid_pose: bool = False

    def retarget(self, frame_dict: dict[str, np.ndarray]) -> np.ndarray:
        """
        :param frame_dict: {joint_name: [x, y, z]}
        :return: target joint positions array (qpos)
        """
        frame = dict(frame_dict)
        
        # add end site joints (for online mode)
        for tip_name, (j3_name, j2_name) in self.END_SITE_MAP.items():
            if tip_name in self.joint_names and tip_name not in frame:
                if j3_name in frame and j2_name in frame:
                    p3 = np.asarray(frame[j3_name], dtype=np.float64)
                    p2 = np.asarray(frame[j2_name], dtype=np.float64)
                    frame[tip_name] = p3 + (p3 - p2) * 0.75
        
        # check missing joints
        missing_joints = [name for name in self.joint_names if name not in frame_dict]
        if missing_joints:
            warnings.warn(
                f">>> [HandRetargeterWrapper] Missing joints: {missing_joints}. "
                f">>> Returning last succesful hand position.",
                RuntimeWarning
            )
            return self.last_qpos.copy()

        # get joints positions & scale them
        raw_points = np.array([frame_dict[name] for name in self.joint_names], dtype=np.float64)
        joints = raw_points * self.scale

        # prepare frame
        joints_centered = joints - joints[0]
        rot_frame = get_hand_frame(joints_centered)
        joints_canonical = joints_centered @ rot_frame
        joints_urdf = joints_canonical @ self.r_hamer2urdf.T

        # compute target vectors
        if self.optimizer.retargeting_type == "POSITION":
            ref_value = joints_urdf[self.optimizer.target_link_human_indices, :]
        else:
            origin_idx = self.optimizer.target_link_human_indices[0, :]
            task_idx = self.optimizer.target_link_human_indices[1, :]
            ref_value = joints_urdf[task_idx, :] - joints_urdf[origin_idx, :]

        # compute qpos & cache
        qpos = self.retargeter.retarget(ref_value)
        self.last_qpos = np.asarray(qpos, dtype=np.float64).copy()
        self.has_valid_pose = True

        return self.last_qpos.copy()

    def reset(self) -> None:
        if hasattr(self.retargeter, "reset"):
            self.retargeter.reset()
        self.last_qpos = np.zeros(self.num_dofs, dtype=np.float64)
        self.has_valid_pose = False