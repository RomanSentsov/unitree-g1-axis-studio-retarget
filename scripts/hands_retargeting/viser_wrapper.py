import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import numpy as np
import viser
from viser.extras import ViserUrdf
import viser.transforms as tf

class ViserHandsVisualizer:
    def __init__(
        self,
        urdf_left: str | Path,
        urdf_right: str | Path,
        dof_names_left: list[str],
        dof_names_right: list[str],
    ):
        self.dof_names_left = dof_names_left
        self.dof_names_right = dof_names_right

        self.server = viser.ViserServer()
        self.server.scene.add_grid("/ground", width=1.5, height=1.5)
        self.server.initial_camera.position = (0.4, 0.0, 0.45)
        self.server.initial_camera.look_at = (0.0, 0.0, 0.1)

        self.server.scene.add_frame(
            "/hand_left",
            position=(0.0, 0.2, 0.3),
            wxyz=tf.SO3.from_x_radians(-np.pi / 3).wxyz,
            show_axes=False,
        )
        self.server.scene.add_frame(
            "/hand_right",
            position=(0.0, -0.2, 0.3),
            wxyz=tf.SO3.from_x_radians(np.pi / 3).wxyz,
            show_axes=False,
        )

        self.urdf_left = ViserUrdf(self.server, urdf_or_path=urdf_left, root_node_name="/hand_left")
        self.urdf_right = ViserUrdf(self.server, urdf_or_path=urdf_right, root_node_name="/hand_right")

        self._actuated_left = self.urdf_left.get_actuated_joint_names()
        self._actuated_right = self.urdf_right.get_actuated_joint_names()

    def _apply_qpos(self, urdf: ViserUrdf, actuated_names: tuple[str, ...], dof_names: list[str], qpos: np.ndarray):
        joint_map = dict(zip(dof_names, qpos))
        cfg = np.array([joint_map.get(name, 0.0) for name in actuated_names], dtype=np.float64)
        urdf.update_cfg(cfg)

    def update(self, q_l: np.ndarray, q_r: np.ndarray) -> None:
        self._apply_qpos(self.urdf_left, self._actuated_left, self.dof_names_left, q_l)
        self._apply_qpos(self.urdf_right, self._actuated_right, self.dof_names_right, q_r)