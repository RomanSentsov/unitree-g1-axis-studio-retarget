from scipy.spatial.transform import Rotation as R
from pin_opti import G1_29_ArmIK
import pinocchio as pin
import numpy as np
import time

from mocap_api import *


# ============================================================
# AXIS STUDIO SKELETON HIERARCHY
# ============================================================

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

    # --------------------------------------------------------
    # Right arm
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Left arm
    # --------------------------------------------------------

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


# ============================================================
# CONSTANT AXIS -> OUR WORLD TRANSFORM
# ============================================================

# This is the same coordinate conversion you were using:
#
# first:
#     +90 deg around X
#
# then:
#     +90 deg around Z
#
# IMPORTANT:
# This transform is applied ONCE at the root.
#

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


# ============================================================
# QUATERNION HELPERS
# ============================================================

def axis_quaternion_to_scipy(q):
    """
    Axis Studio MocapApi quaternion -> scipy Rotation.

    Axis API:
        (w, x, y, z)

    scipy:
        (x, y, z, w)
    """

    q = np.asarray(q, dtype=float).reshape(-1)

    if q.size != 4:
        raise ValueError(
            f"Expected quaternion with 4 components, got {q}"
        )

    w, x, y, z = q

    norm = np.linalg.norm(q)

    if norm < 1e-10:
        raise ValueError(
            f"Zero-length quaternion received: {q}"
        )

    q = q / norm

    w, x, y, z = q

    return R.from_quat([x, y, z, w])


def quat_to_matrix(q):
    """
    Axis quaternion (w,x,y,z) -> 3x3 matrix.
    """

    return axis_quaternion_to_scipy(q).as_matrix()


def matrix_to_scipy_quat(mat):
    """
    3x3 matrix -> scipy quaternion (x,y,z,w)
    """

    return R.from_matrix(mat).as_quat()


# ============================================================
# FK
# ============================================================

def get_global_transform(
    joints_dict,
    joint_name,
    parent_map=None,
    relative_to_hips=True,
    _cache=None,
):
    """
    Calculate global joint transform.

    IMPORTANT:
    We DO NOT use get_local_rotation_by_euler().

    We use Axis Studio's native local quaternion:
        joint.get_local_rotation()

    Parent composition:

        T_global =
            T_parent_global @ T_local

    Rotation:

        R_global =
            R_parent_global @ R_local
    """

    if parent_map is None:
        parent_map = PARENT_MAP

    if _cache is None:
        _cache = {}

    cache_key = (joint_name, relative_to_hips)

    if cache_key in _cache:
        return _cache[cache_key]

    joint = joints_dict[joint_name]

    # --------------------------------------------------------
    # Local translation
    # --------------------------------------------------------

    local_pos = np.asarray(
        joint.get_local_position(),
        dtype=float
    )

    # --------------------------------------------------------
    # Local rotation
    #
    # DO NOT use:
    #
    #     get_local_rotation_by_euler()
    #
    # because that introduces an Euler-order dependency.
    #
    # --------------------------------------------------------

    local_quat = joint.get_local_rotation()

    local_rot = axis_quaternion_to_scipy(local_quat)

    local_mat = np.eye(4)

    local_mat[:3, :3] = local_rot.as_matrix()
    local_mat[:3, 3] = local_pos

    # --------------------------------------------------------
    # Parent
    # --------------------------------------------------------

    parent_name = parent_map.get(joint_name)

    if parent_name is None:

        if relative_to_hips:

            # Put Hips at origin and apply our world conversion.
            #
            # We intentionally DO NOT use Hips local rotation here.
            # This matches the behavior of your previous code.

            global_mat = AXIS_TO_TARGET.copy()

            # If you want root translation:
            #
            # global_mat[:3, 3] = (
            #     AXIS_TO_TARGET[:3, :3] @ local_pos
            # )

            global_mat[:3, 3] = 0.0

        else:

            global_mat = local_mat

    else:

        parent_pos, parent_quat = get_global_transform(
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

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

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


# ============================================================
# DEBUG FUNCTIONS
# ============================================================

def print_joint_rotation(joint):
    """
    Print Axis quaternion and Euler representation.

    This is ONLY for debugging.
    """

    q = np.asarray(
        joint.get_local_rotation(),
        dtype=float
    )

    euler = np.asarray(
        joint.get_local_rotation_by_euler(),
        dtype=float
    )

    print()
    print("JOINT:", joint.get_name())
    print("Axis quaternion (w,x,y,z):", q)
    print("Axis Euler:", euler)

    try:
        rot = axis_quaternion_to_scipy(q)

        print(
            "Quaternion -> Euler xyz:",
            rot.as_euler("xyz", degrees=True)
        )

        print(
            "Quaternion -> Euler yxz:",
            rot.as_euler("yxz", degrees=True)
        )

        print(
            "Quaternion -> Euler yzx:",
            rot.as_euler("yzx", degrees=True)
        )

    except Exception as e:
        print("Quaternion conversion error:", e)


def check_quaternion(joint):
    """
    Check whether quaternion returned by Axis is valid.
    """

    q = np.asarray(
        joint.get_local_rotation(),
        dtype=float
    )

    norm = np.linalg.norm(q)

    print(
        f"{joint.get_name():20s} "
        f"q={q} "
        f"norm={norm:.6f}"
    )


# ============================================================
# AXIS EVENT
# ============================================================

def get_event_type_name(event_type_value):

    event_type_map = {
        MCPEventType.InvalidEvent: 'InvalidEvent',
        MCPEventType.AvatarUpdated: 'AvatarUpdated',
        MCPEventType.TrackerUpdated: 'TrackerUpdated',
        MCPEventType.AliceIMUUpdated: 'AliceIMUUpdated',
        MCPEventType.AliceRigidbodyUpdated: 'AliceRigidbodyUpdated',
        MCPEventType.AliceTrackerUpdated: 'AliceTrackerUpdated',
        MCPEventType.AliceMarkerUpdated: 'AliceMarkerUpdated',
    }

    return event_type_map.get(
        event_type_value,
        f'Unknown({event_type_value})'
    )


# ============================================================
# MAIN AXIS STUDIO CLASS
# ============================================================

class MocapAxisDemo:

    def __init__(self):

        self.app = None
        self.running = False

        self.prev_posture_time_ms = None

        # ----------------------------------------------------
        # IK
        # ----------------------------------------------------

        self.arm_ik = G1_29_ArmIK(
            Unit_Test=True,
            Visualization=True
        )

        # ----------------------------------------------------
        # Joint names
        # ----------------------------------------------------

        self.r_hand_name = "RightHand"
        self.l_hand_name = "LeftHand"

        self.r_mid_name = "RightHandMiddle1"
        self.l_mid_name = "LeftHandMiddle1"

        self.r_pinky_name = "RightHandPinky1"
        self.l_pinky_name = "LeftHandPinky1"

        self.r_elbow_name = "RightForeArm"
        self.l_elbow_name = "LeftForeArm"

        # ----------------------------------------------------
        # Cached positions
        # ----------------------------------------------------

        self.r_hand_pos = None
        self.l_hand_pos = None

        self.r_mid_pos = None
        self.l_mid_pos = None

        self.r_pinky_pos = None
        self.l_pinky_pos = None

        self.r_elbow_pos = None
        self.l_elbow_pos = None

        self.debug_printed = False

    # ========================================================
    # START
    # ========================================================

    def start(self, udp_port=7012):

        self.app = MCPApplication()

        settings = MCPSettings()

        settings.set_udp(udp_port)

        # ----------------------------------------------------
        # IMPORTANT
        # ----------------------------------------------------
        #
        # This setting controls the Euler order of BVH data.
        #
        # We don't use Euler values below, but keep this
        # consistent with Axis Studio's BVH output.
        #
        # ----------------------------------------------------

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

                    else:

                        print(
                            'Other events:',
                            get_event_type_name(
                                evt.event_type
                            )
                        )

        except KeyboardInterrupt:

            print(
                "Program interrupted by user"
            )

        finally:

            self.stop()

    # ========================================================
    # FRAME
    # ========================================================

    def _handle_avatar_data(self, evt):

        avatar = MCPAvatar(
            evt.event_data.avatar_handle
        )

        joints = avatar.get_joints()

        # ----------------------------------------------------
        # Limit processing to ~30 FPS
        # ----------------------------------------------------

        current_time_ms = time.time() * 1000.0

        if self.prev_posture_time_ms is not None:

            delta_ms = (
                current_time_ms
                - self.prev_posture_time_ms
            )

            if delta_ms < 33.0:
                return

        self.prev_posture_time_ms = current_time_ms

        # ----------------------------------------------------
        # Dictionary
        # ----------------------------------------------------

        joints_dict = {
            j.get_name(): j
            for j in joints
        }

        _cache = {}

        # ----------------------------------------------------
        # DEBUG ON FIRST FRAME
        # ----------------------------------------------------

        if not self.debug_printed:

            print("\n========== AXIS DEBUG ==========\n")

            for name in [
                "Hips",
                "Spine2",
                "RightArm",
                "RightForeArm",
                "RightHand",
                "LeftArm",
                "LeftForeArm",
                "LeftHand",
            ]:

                if name in joints_dict:

                    check_quaternion(
                        joints_dict[name]
                    )

            if "RightHand" in joints_dict:

                print_joint_rotation(
                    joints_dict["RightHand"]
                )

            print(
                "\n================================\n"
            )

            self.debug_printed = True

        # ====================================================
        # GLOBAL POSITIONS
        # ====================================================

        (
            self.r_hand_pos,
            self.r_hand_quat,
        ) = get_global_transform(
            joints_dict,
            self.r_hand_name,
            relative_to_hips=True,
            _cache=_cache,
        )

        (
            self.l_hand_pos,
            self.l_hand_quat,
        ) = get_global_transform(
            joints_dict,
            self.l_hand_name,
            relative_to_hips=True,
            _cache=_cache,
        )

        (
            self.r_mid_pos,
            _
        ) = get_global_transform(
            joints_dict,
            self.r_mid_name,
            relative_to_hips=True,
            _cache=_cache,
        )

        (
            self.l_mid_pos,
            _
        ) = get_global_transform(
            joints_dict,
            self.l_mid_name,
            relative_to_hips=True,
            _cache=_cache,
        )

        (
            self.r_pinky_pos,
            _
        ) = get_global_transform(
            joints_dict,
            self.r_pinky_name,
            relative_to_hips=True,
            _cache=_cache,
        )

        (
            self.l_pinky_pos,
            _
        ) = get_global_transform(
            joints_dict,
            self.l_pinky_name,
            relative_to_hips=True,
            _cache=_cache,
        )

        (
            self.r_elbow_pos,
            _
        ) = get_global_transform(
            joints_dict,
            self.r_elbow_name,
            relative_to_hips=True,
            _cache=_cache,
        )

        (
            self.l_elbow_pos,
            _
        ) = get_global_transform(
            joints_dict,
            self.l_elbow_name,
            relative_to_hips=True,
            _cache=_cache,
        )

        # ====================================================
        # HAND ORIENTATION
        # ====================================================

        r_hand_x = (
            self.r_mid_pos
            - self.r_hand_pos
        )

        r_hand_y = (
            self.r_pinky_pos
            - self.r_mid_pos
        )

        l_hand_x = (
            self.l_mid_pos
            - self.l_hand_pos
        )

        l_hand_y = (
            self.l_pinky_pos
            - self.l_mid_pos
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # We keep EXACTLY the same hand-frame construction
        # that worked for your BVH file.
        # ----------------------------------------------------

        r_hand_rot = R.align_vectors(
            np.stack(
                [
                    r_hand_x,
                    r_hand_y
                ],
                axis=0
            ),
            np.array([
                [1, 0, 0],
                [0, 0, -1]
            ])
        )[0]

        l_hand_rot = R.align_vectors(
            np.stack(
                [
                    l_hand_x,
                    l_hand_y
                ],
                axis=0
            ),
            np.array([
                [1, 0, 0],
                [0, 0, -1]
            ])
        )[0]

        r_quat = r_hand_rot.as_quat()
        l_quat = l_hand_rot.as_quat()

        # ====================================================
        # PINOCCHIO TARGETS
        # ====================================================

        R_tf_target = pin.SE3(
            pin.Quaternion(r_quat),
            self.r_hand_pos / 150.0,
        )

        L_tf_target = pin.SE3(
            pin.Quaternion(l_quat),
            self.l_hand_pos / 150.0,
        )

        # ----------------------------------------------------
        # Elbow position targets
        # ----------------------------------------------------

        R_tf_elbow_target = pin.SE3(
            pin.Quaternion(
                1,
                0,
                0,
                0
            ),
            self.r_elbow_pos / 150.0,
        )

        L_tf_elbow_target = pin.SE3(
            pin.Quaternion(
                1,
                0,
                0,
                0
            ),
            self.l_elbow_pos / 150.0,
        )

        # ====================================================
        # IK
        # ====================================================

        until = time.time()

        self.arm_ik.solve_ik(
            L_tf_target.homogeneous,
            R_tf_target.homogeneous,
            L_tf_elbow_target.homogeneous,
            R_tf_elbow_target.homogeneous
        )

        elapsed = time.time() - until

        print(
            f"IK: {elapsed * 1000.0:.2f} ms"
        )

    # ========================================================
    # STOP
    # ========================================================

    def stop(self):

        self.running = False

        if self.app:

            self.app.close()

            print(
                "Mocap application closed"
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    demo = MocapAxisDemo()

    print(
        "Starting Mocap Axis Studio demo..."
    )

    demo.start()