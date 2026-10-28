import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation as R


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

    # Матрица поворота из AXIS_TO_TARGET (3x3)
    _ROOT_ROT_MAT = AXIS_TO_TARGET[:3, :3].copy()
    # Кватернион корня (x, y, z, w) — постоянный
    _ROOT_QUAT = R.from_matrix(_ROOT_ROT_MAT).as_quat().astype(float)

    # Топологический порядок (вычисляется один раз при первом вызове)
    _TOPO_ORDER = None

    @classmethod
    def _init_topo_order(cls):
        """BFS от корня — вычисляется один раз."""
        if cls._TOPO_ORDER is not None:
            return

        # Находим корень
        root = next(node for node, parent in cls.PARENT_MAP.items() if parent is None)

        # Обратный словарь: parent -> [children]
        children = {node: [] for node in cls.PARENT_MAP}
        for child, parent in cls.PARENT_MAP.items():
            if parent is not None:
                children[parent].append(child)

        # BFS
        queue = deque([root])
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            queue.extend(children[node])

        cls._TOPO_ORDER = order

    # ==================== Сырая математика (без scipy) ====================

    @staticmethod
    def _quat_mul(q1, q2):
        """Умножение кватернионов (x, y, z, w). Возвращает новый массив."""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        ], dtype=float)

    @staticmethod
    def _rotate_vec_by_quat(v, q):
        """Поворот вектора v кватернионом q (x,y,z,w)."""
        ux, uy, uz = q[0], q[1], q[2]
        s = q[3]
        vx, vy, vz = v[0], v[1], v[2]

        # u x v
        cx = uy*vz - uz*vy
        cy = uz*vx - ux*vz
        cz = ux*vy - uy*vx

        # u . v
        dot = ux*vx + uy*vy + uz*vz

        t = 2.0 * dot
        u_sq = ux*ux + uy*uy + uz*uz
        k = s*s - u_sq

        return np.array([
            t*ux + k*vx + 2.0*s*cx,
            t*uy + k*vy + 2.0*s*cy,
            t*uz + k*vz + 2.0*s*cz,
        ], dtype=float)

    @staticmethod
    def _wxyz_to_xyzw(q_wxyz):
        """Конвертация (w, x, y, z) -> (x, y, z, w)."""
        return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=float)

    # ==================== Основной метод ====================

    @classmethod
    def compute_all_fk(cls, joints_dict):
        """
        Вычисляет глобальные позиции всех суставов за один проход.

        Args:
            joints_dict: dict {name: joint} из Axis Studio

        Returns:
            dict: {joint_name: np.array([x, y, z])}
        """
        # Lazy-инициализация топологического порядка
        cls._init_topo_order()

        global_pos = {}
        global_quat = {}

        root_quat = cls._ROOT_QUAT

        for joint_name in cls._TOPO_ORDER:
            joint = joints_dict[joint_name]

            local_pos = np.asarray(joint.get_local_position(), dtype=float)
            local_quat = cls._wxyz_to_xyzw(joint.get_local_rotation())

            parent_name = cls.PARENT_MAP[joint_name]

            if parent_name is None:
                # Корень: позиция обнуляется (relative_to_hips=True),
                # ориентация = AXIS_TO_TARGET
                global_pos[joint_name] = np.zeros(3, dtype=float)
                global_quat[joint_name] = root_quat
            else:
                p_pos = global_pos[parent_name]
                p_quat = global_quat[parent_name]

                global_pos[joint_name] = p_pos + cls._rotate_vec_by_quat(local_pos, p_quat)
                global_quat[joint_name] = cls._quat_mul(p_quat, local_quat)

        return global_pos

    # ==================== Совместимость со старым API ====================

    @classmethod
    def get_global_transform(
        cls,
        joints_dict,
        joint_name,
        parent_map=None,
        relative_to_hips=True,
        _cache=None,
    ):
        """
        Старый интерфейс. Работает через compute_all_fk.
        Используйте compute_all_fk напрямую для максимальной производительности.
        """
        if _cache is None:
            _cache = {}

        cache_key = (joint_name, relative_to_hips)
        if cache_key in _cache:
            return _cache[cache_key]

        all_pos = cls.compute_all_fk(joints_dict)

        # Для обратной совместимости — упакуем в кэш все суставы
        for name in cls.NODE_NAMES:
            _cache[(name, relative_to_hips)] = (all_pos[name], None)

        return _cache[cache_key]