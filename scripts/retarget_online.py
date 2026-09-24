from pin_opti import G1_29_ArmIK
from utils.axis_studio_bvh import AxisStudioFK
import time

from mocap_api import *

# Application class
class MocapAxisDemo:

    def __init__(self):

        self.app = None
        self.running = False

        self.prev_posture_time_ms = None

        # init arm ik (retargeter)
        self.arm_ik = G1_29_ArmIK(Unit_Test=True, Visualization=True)

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

        print("received")

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

            if delta_ms < 33.0:
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

        elapsed = time.time() - until
        print(f"Retarget solved in: {elapsed * 1000.0:.2f} ms")


if __name__ == "__main__":

    demo = MocapAxisDemo()
    print("Starting Mocap Axis Studio demo...")
    demo.start()