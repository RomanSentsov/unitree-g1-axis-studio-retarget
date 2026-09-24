from scipy.spatial.transform import Rotation as R
import numpy as np
from mocap_api import *

class AxisStudioFK:

    PARENT_MAP = {
        # Root
        "Hips": None,

        # Right leg
        "RightUpLeg": "Hips",
        "RightLeg": "RightUpLeg",
        "RightFoot": "RightLeg",

        # Left leg
        "LeftUpLeg": "Hips",
        "LeftLeg": "LeftUpLeg",
        "LeftFoot": "LeftLeg",
                                    
        # Spine
        "Spine": "Hips",
        "Spine1": "Spine",
        "Spine2": "Spine1",

        # Neck / head
        "Neck": "Spine2",
        "Neck1": "Neck",
        "Head": "Neck1",

        # Right arm

        "RightShoulder": "Spine2",
        "RightArm": "RightShoulder",
        "RightForeArm": "RightArm",
        "RightHand": "RightForeArm",

        "RightHandThumb1": "RightHand",
        "RightHandThumb2": "RightHandThumb1",
        "RightHandThumb3": "RightHandThumb2",

        "RightInHandIndex": "RightHand",
        "RightHandIndex1": "RightInHandIndex",
        "RightHandIndex2": "RightHandIndex1",
        "RightHandIndex3": "RightHandIndex2",

        "RightInHandMiddle": "RightHand",
        "RightHandMiddle1": "RightInHandMiddle",
        "RightHandMiddle2": "RightHandMiddle1",
        "RightHandMiddle3": "RightHandMiddle2",

        "RightInHandRing": "RightHand",
        "RightHandRing1": "RightInHandRing",
        "RightHandRing2": "RightHandRing1",
        "RightHandRing3": "RightHandRing2",

        "RightInHandPinky": "RightHand",
        "RightHandPinky1": "RightInHandPinky",
        "RightHandPinky2": "RightHandPinky1",
        "RightHandPinky3": "RightHandPinky2",

        # Left arm

        "LeftShoulder": "Spine2",
        "LeftArm": "LeftShoulder",
        "LeftForeArm": "LeftArm",
        "LeftHand": "LeftForeArm",

        "LeftHandThumb1": "LeftHand",
        "LeftHandThumb2": "LeftHandThumb1",
        "LeftHandThumb3": "LeftHandThumb2",

        "LeftInHandIndex": "LeftHand",
        "LeftHandIndex1": "LeftInHandIndex",
        "LeftHandIndex2": "LeftHandIndex1",
        "LeftHandIndex3": "LeftHandIndex2",

        "LeftInHandMiddle": "LeftHand",
        "LeftHandMiddle1": "LeftInHandMiddle",
        "LeftHandMiddle2": "LeftHandMiddle1",
        "LeftHandMiddle3": "LeftHandMiddle2",

        "LeftInHandRing": "LeftHand",
        "LeftHandRing1": "LeftInHandRing",
        "LeftHandRing2": "LeftHandRing1",
        "LeftHandRing3": "LeftHandRing2",

        "LeftInHandPinky": "LeftHand",
        "LeftHandPinky1": "LeftInHandPinky",
        "LeftHandPinky2": "LeftHandPinky1",
        "LeftHandPinky3": "LeftHandPinky2",
    }

    NODE_NAMES = list(PARENT_MAP.keys())

    # CONSTANT AXIS -> OUR WORLD TRANSFORM
    # +90 deg around X   +90 deg around Z

    AXIS_TO_TARGET = np.array([
        [0, -1, 0, 0],
        [1,  0, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1],
    ], dtype=float) @ np.array([
        [1, 0,  0, 0],
        [0, 0, -1, 0],
        [0, 1,  0, 0],
        [0, 0,  0, 1],
    ], dtype=float)

    @staticmethod
    def axis_quaternion_to_scipy(q):
        """
        (w, x, y, z) -> R
        """
        q = np.asarray(q, dtype=float).reshape(-1)

        w, x, y, z = q

        return R.from_quat([x, y, z, w])

    @staticmethod
    def quat_to_matrix(q):
        """
        R -> 3x3 matrix.
        """

        return AxisStudioFK.axis_quaternion_to_scipy(q).as_matrix()

    @staticmethod
    def matrix_to_scipy_quat(self, mat):
        """
        3x3 matrix -> (x,y,z,w)
        """

        return R.from_matrix(mat).as_quat()

    # FK
    @staticmethod
    def get_global_transform(
        joints_dict,
        joint_name,
        parent_map=None,
        relative_to_hips=True,
        _cache=None,
    ):
        """
        Calculate global joint transform.

        ## Args:

            joint_dict - dict of (joint_idx, joint_name)
            joint_name - name of joint to find global transform of
            parent map, relative_to_hips - do not change for unitree_g1 and axis_studio
            _cache - if in a single time frame FK is being calculated for multiple joints, pass {} to speed up calscs.
            CLEAN CACHE EVERY TIME FRAME

        ## Returns:

            position: (x, y, z)
            orientation as quat (x, y, z, w)
        """

        if parent_map is None:
            parent_map = AxisStudioFK.PARENT_MAP

        if _cache is None:
            _cache = {}

        cache_key = (joint_name, relative_to_hips)

        if cache_key in _cache:
            return _cache[cache_key]

        joint = joints_dict[joint_name]

        # Local translation
        local_pos = np.asarray(
            joint.get_local_position(),
            dtype=float
        )

        # Local rotation
        local_quat = joint.get_local_rotation()

        local_rot = AxisStudioFK.axis_quaternion_to_scipy(local_quat)

        local_mat = np.eye(4)

        local_mat[:3, :3] = local_rot.as_matrix()
        local_mat[:3, 3] = local_pos

        # Parent
        parent_name = parent_map.get(joint_name)

        if parent_name is None:

            if relative_to_hips:

                global_mat = AxisStudioFK.AXIS_TO_TARGET.copy()
                # Root translation set to 0
                global_mat[:3, 3] = 0.0

            else:
                # currently not working
                global_mat = local_mat

        else:
        # recurrent
            parent_pos, parent_quat = AxisStudioFK.get_global_transform(
                joints_dict,
                parent_name,
                parent_map,
                relative_to_hips,
                _cache,
            )

            parent_rot = R.from_quat(parent_quat)

            parent_mat = np.eye(4)
            parent_mat[:3, :3] = parent_rot.as_matrix()
            parent_mat[:3, 3] = parent_pos

            global_mat = parent_mat @ local_mat

        # Result
        global_pos = global_mat[:3, 3].copy()

        global_rot = R.from_matrix(
            global_mat[:3, :3]
        )

        # scipy format:
        # (x,y,z,w)
        global_quat = global_rot.as_quat()

        result = (
            global_pos,
            global_quat,
        )

        _cache[cache_key] = result

        return result

