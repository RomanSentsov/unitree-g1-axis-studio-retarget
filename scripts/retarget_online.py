from scipy.spatial.transform import Rotation as R
from pin_opti import G1_29_ArmIK
import pinocchio as pin
import numpy as np
import time
from mocap_api import *



PARENT_MAP = {
    # Корень
    "Hips": None,

    # Правая нога
    "RightUpLeg": "Hips",
    "RightLeg": "RightUpLeg",
    "RightFoot": "RightLeg",

    # Левая нога
    "LeftUpLeg": "Hips",
    "LeftLeg": "LeftUpLeg",
    "LeftFoot": "LeftLeg",

    # Позвоночник и голова
    "Spine": "Hips",
    "Spine1": "Spine",
    "Spine2": "Spine1",
    "Neck": "Spine2",
    "Neck1": "Neck",
    "Head": "Neck1",

    # Правая рука
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

    # Левая рука
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

def get_global_transform(joints_dict, joint_name, parent_map=None,
                         relative_to_hips=True, _cache=None, ignore_rot=True):
    """
    Возвращает глобальную позицию и кватернион сустава.

    Если relative_to_hips=True, трансформация считается в системе координат
    Hips (Hips помещается в начало координат без поворота, его собственные
    смещение и ориентация игнорируются).
    """
    if parent_map is None:
        parent_map = PARENT_MAP
    if _cache is None:
        _cache = {}

    # Ключ кэша зависит от флага, иначе можно получить неверный результат
    cache_key = (joint_name, relative_to_hips)
    if cache_key in _cache:
        return _cache[cache_key]

    joint = joints_dict[joint_name]

    # ---- Локальная позиция ----
    local_pos = np.array(joint.get_local_position(), dtype=float)

    # quat_rot = np.array(joint.get_local_rotation(), dtype=float)
    euler_rot = joint.get_local_rotation_by_euler()

    # ---- Локальная матрица 4×4 ----
    rot = R.from_euler('yxz', euler_rot, degrees=True)
    # rot = R.from_quat(quat_rot)

    local_mat = np.eye(4)
    local_mat[:3, :3] = rot.as_matrix()
    local_mat[:3, 3] = local_pos

    parent_name = parent_map.get(joint_name)

    if parent_name is None:
        # Root
        if relative_to_hips:
            # reorient
            # local_mat[:3, 3] = np.zeros(3)

            global_mat = np.array([[1, 0, 0, 0], # 90 degs arond x axis
                                   [0, 0, -1, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 0, 1]])


            global_mat = np.array([[0, -1, 0, 0], # 90 degs arond z axis
                                    [1, 0, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]]) @ global_mat

            # global_mat = local_mat @ global_mat # orient of Hips
            
        else:
            # not working currently
            global_mat = local_mat
    else:
        parent_pos, parent_quat = get_global_transform(
            joints_dict, parent_name, parent_map, relative_to_hips, _cache
        )
        parent_rot = R.from_quat(parent_quat)
        parent_mat = np.eye(4)
        parent_mat[:3, :3] = parent_rot.as_matrix()
        parent_mat[:3, 3] = parent_pos

        global_mat = parent_mat @ local_mat

    global_pos = global_mat[:3, 3]
    global_quat = R.from_matrix(global_mat[:3, :3]).as_quat()

    result = (global_pos, global_quat)
    _cache[cache_key] = result
    return result

# Axis Studio stuff

def get_event_type_name(event_type_value):
    """
    Convert event type value to corresponding enum name
    
    Args:
        event_type_value: The numeric event type value
        
    Returns:
        str: The corresponding event type name
    """
    event_type_map = {
        MCPEventType.InvalidEvent: 'InvalidEvent',
        MCPEventType.AvatarUpdated: 'AvatarUpdated',
        MCPEventType.TrackerUpdated: 'TrackerUpdated',
        MCPEventType.AliceIMUUpdated: 'AliceIMUUpdated',
        MCPEventType.AliceRigidbodyUpdated: 'AliceRigidbodyUpdated',  
        MCPEventType.AliceTrackerUpdated: 'AliceTrackerUpdated',
        MCPEventType.AliceMarkerUpdated: 'AliceMarkerUpdated',
    }
    return event_type_map.get(event_type_value, f'Unknown({event_type_value})')


class MocapAxisDemo:
    """
    Mocap Axis Studio Demo class for demonstrating how to get Axis Studio data through Mocap API
    """
    
    def __init__(self):
        """
        Initialize Mocap Axis Studio Demo instance
        """
        self.app = None
        self.running = False
        self.prev_posture_time_ms = None

        # Retarget init

        self.arm_ik = G1_29_ArmIK(Unit_Test = True, Visualization = True)

        self.r_hand_name = "RightHand"
        self.l_hand_name = "LeftHand"

        self.r_mid_name = "RightHandMiddle1"
        self.l_mid_name = "LeftHandMiddle1"

        self.r_pinky_name = "RightHandPinky1"
        self.l_pinky_name = "LeftHandPinky1"

        self.r_elbow_name = "RightForeArm"
        self.l_elbow_name = "LeftForeArm"

        self.r_hand_pos = None
        self.l_hand_pos = None

        self.r_mid_pos = None
        self.l_mid_pos = None

        self.r_pinky_pos = None
        self.l_pinky_pos = None

        self.r_elbow_pos = None
        self.l_elbow_pos = None

    def start(self, udp_port=7012):
        """
        Start Mocap application and handle event loop
        
        Args:
            udp_port: UDP port number, default is 7012
        """
        # Initialize Mocap application
        self.app = MCPApplication()
        settings = MCPSettings()
        settings.set_udp(udp_port)
        settings.set_bvh_rotation(MCPBvhRotation.YXZ)
        settings.set_bvh_data
        self.app.set_settings(settings)
        self.app.open()
        print(f"Mocap application initialized, UDP port: {udp_port}")
        
        self.running = True
        try:
            while self.running:
                evts = self.app.poll_next_event()
                for evt in evts:
                    if evt.event_type == MCPEventType.AvatarUpdated: # avatar bvh(人体BVH)
                        self._handle_avatar_data(evt)
                    else:
                        print('Other events:', get_event_type_name(evt.event_type))
        except KeyboardInterrupt:
            print("Program interrupted by user")
        finally:
            self.stop()


    def _handle_avatar_data(self, evt):
        """
        Handle avatar data
        """
        avatar = MCPAvatar(evt.event_data.avatar_handle)


        joints = avatar.get_joints()  # Get all joint data

        # TIME because 90 fps is too much for retargeter with visualization

        # hour, minute, second, millisecond = avatar.get_avatar_posture_time()      # error if stream BVH edit
        # current_time_ms = ((hour * 3600 + minute * 60 + second) * 1000 + millisecond)


        current_time_ms = time.time() * 1000
        
        if self.prev_posture_time_ms is not None:
            delta_ms = current_time_ms - self.prev_posture_time_ms
            if delta_ms < 33.0:
                return
        self.prev_posture_time_ms = current_time_ms

        # for the Forward kinematic

        joints_dict = {j.get_name(): j for j in joints}
        _cache = {}

        for joint in joints:
            link_name = joint.get_name()

            if link_name == self.r_hand_name:
                self.r_hand_pos, _ = get_global_transform(
                    joints_dict, self.r_hand_name, relative_to_hips=True, _cache=_cache)
            elif link_name == self.l_hand_name:
                self.l_hand_pos, _ = get_global_transform(
                    joints_dict, self.l_hand_name, relative_to_hips=True, _cache=_cache)

            elif link_name == self.r_mid_name:
                self.r_mid_pos, _ = get_global_transform(
                    joints_dict, self.r_mid_name, relative_to_hips=True, _cache=_cache)
            elif link_name == self.l_mid_name:
                self.l_mid_pos, _ = get_global_transform(
                    joints_dict, self.l_mid_name, relative_to_hips=True, _cache=_cache)

            elif link_name == self.r_pinky_name:
                self.r_pinky_pos, _ = get_global_transform(
                    joints_dict, self.r_pinky_name, relative_to_hips=True, _cache=_cache)
            elif link_name == self.l_pinky_name:
                self.l_pinky_pos, _ = get_global_transform(
                    joints_dict, self.l_pinky_name, relative_to_hips=True, _cache=_cache)

            elif link_name == self.r_elbow_name:
                self.r_elbow_pos, _ = get_global_transform(
                    joints_dict, self.r_elbow_name, relative_to_hips=True, _cache=_cache)
            elif link_name == self.l_elbow_name:
                self.l_elbow_pos, _ = get_global_transform(
                    joints_dict, self.l_elbow_name, relative_to_hips=True, _cache=_cache)

        # print(self.r_hand_pos)

        r_hand_x = self.r_mid_pos - self.r_hand_pos
        r_hand_y = self.r_pinky_pos - self.r_mid_pos

        l_hand_x = self.l_mid_pos - self.l_hand_pos
        l_hand_y = self.l_pinky_pos - self.l_mid_pos

        # Orientation of hands

        r_quat = R.align_vectors(np.stack([r_hand_x, r_hand_y], axis=0),
                                np.array([[1, 0, 0], [0, 0, -1]]))[0].as_quat()

        l_quat = R.align_vectors(np.stack([l_hand_x, l_hand_y], axis=0),
                                np.array([[1, 0, 0], [0, 0, -1]]))[0].as_quat()

        # Making SE3 

        R_tf_target = pin.SE3(
            pin.Quaternion(r_quat),
            self.r_hand_pos/150,
        )   

        L_tf_target = pin.SE3(
            pin.Quaternion(l_quat),
            self.l_hand_pos/150,
        )


        R_tf_elbow_target = pin.SE3(
            pin.Quaternion(1, 0, 0, 0),
            self.r_elbow_pos/150,
        )   

        L_tf_elbow_target = pin.SE3(
            pin.Quaternion(1, 0, 0, 0),
            self.l_elbow_pos/150,
        )


        until = time.time()
        self.arm_ik.solve_ik(L_tf_target.homogeneous, R_tf_target.homogeneous, L_tf_elbow_target.homogeneous, R_tf_elbow_target.homogeneous)
        now = time.time()
        elapsed = now - until
        print(now - until)
        

    def stop(self):
        """
        Close Mocap application
        """
        self.running = False
        if self.app:
            self.app.close()
            print("Mocap application closed")

if __name__ == "__main__":
    # Create and run demo instance
    demo = MocapAxisDemo()
    print("Starting Mocap Axis Studio demo...") 
    demo.start()

