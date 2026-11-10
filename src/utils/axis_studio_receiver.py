import numpy as np
from typing import Optional
try:
    from mocap_api import MCPApplication, MCPSettings, MCPBvhRotation, MCPEventType, MCPAvatar
    from utils.axis_studio_bvh import AxisStudioFK
    MOCAP_API_AVAILABLE = True
except ImportError:
    MOCAP_API_AVAILABLE = False

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

class AxisStudioReceiver:
    """
    Receiving UDP data from Axis Studio and calculating forward kinematics.
    """

    def __init__(self, joint_names: list[str], udp_port: int = 7012):
        if not MOCAP_API_AVAILABLE:
            raise RuntimeError("mocap_api or utils.axis_studio_bvh not found!")

        self.joint_names = joint_names
        self.udp_port = udp_port

        self.app = MCPApplication()
        settings = MCPSettings()
        settings.set_udp(self.udp_port)
        settings.set_bvh_rotation(MCPBvhRotation.YXZ)
        self.app.set_settings(settings)

    def open(self):
        self.app.open()
        print(f">>> [AxisStudioReceiver] Connected to UDP: {self.udp_port}")

    def close(self):
        if self.app:
            self.app.close()
            print(">>> [AxisStudioReceiver] Connection closed.")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_frame(self) -> Optional[dict[str, np.ndarray]]:
        evts = self.app.poll_next_event()
        for evt in evts:
            if evt.event_type == MCPEventType.AvatarUpdated:
                avatar = MCPAvatar(evt.event_data.avatar_handle)
                joints = avatar.get_joints()
                joints_dict = {j.get_name(): j for j in joints}

                _cache = {}
                frame_dict = {}

                real_joints = [name for name in self.joint_names if not name.startswith("EndSite")]
                for name in real_joints:
                    pos, _ = AxisStudioFK.get_global_transform(
                        joints_dict,
                        name,
                        relative_to_hips=True,
                        _cache=_cache,
                    )
                    frame_dict[name] = pos

                for tip_name, (j3_name, j2_name) in END_SITE_MAP.items():
                    if tip_name in self.joint_names:
                        if j3_name in frame_dict and j2_name in frame_dict:
                            p3 = frame_dict[j3_name]
                            p2 = frame_dict[j2_name]
                            frame_dict[tip_name] = p3 + (p3 - p2) * 0.75
                        else:
                            frame_dict[tip_name] = frame_dict.get(j3_name, np.zeros(3))

                return frame_dict