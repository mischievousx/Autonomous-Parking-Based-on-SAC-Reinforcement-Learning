import gymnasium as gym
from gymnasium import spaces
import numpy as np
from environment.vehicle import Vehicle
from environment.scene import ParkingScene

class ParkingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        self.vehicle = Vehicle()
        self.scene = ParkingScene()
        self.dt = 0.1
        self.max_steps = 300
        self.step_count = 0
        self.max_iou = 0.0
        self.init_dist = 1.0
        self.r_dist = 0.0
        
        # Action space: continuous [a, delta_rate].
        self.action_space = spaces.Box(
            low=np.array([-self.vehicle.max_a, -self.vehicle.max_delta_rate], dtype=np.float32),
            high=np.array([self.vehicle.max_a, self.vehicle.max_delta_rate], dtype=np.float32),
        )

        # Assume up to 5 obstacles (avoid zero-dim before reset).
        obs_dim = 5 + 3

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    # =====================================================
    # Gymnasium reset
    # =====================================================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0

        # Build scene.
        self.scene.build_scene(occupied=(True, False, True), layout="vertical")

        # Give a safe initial pose (avoid instant collision).
        x = self.np_random.uniform(8.0, 9.0)
        y = self.np_random.uniform(9.0, 10.0)
        theta = self.np_random.uniform(-np.pi/6, np.pi/6)

        self.vehicle.set_pose(x, y, theta)
        self.vehicle.v = 0.0
        self.vehicle.delta = 0.0
        self.r_dist = 0.0
        self.max_r_dist = 0.0
        self.pre_angle_reward = 0.0

        # Initialize distance and IoU references.
        slot = self.scene.parking_spaces[self.scene.target_idx]
        slot_center = slot["center"].copy()
        self.init_dist = np.linalg.norm(slot_center - np.array([x + self.vehicle.wheelbase/2 * np.cos(theta), y + self.vehicle.wheelbase/2 * np.sin(theta)]))
        self.max_iou = self.compute_iou()

        obs = self._get_state()

        return obs, {}

    # =====================================================
    # Gymnasium step
    # =====================================================

    def step(self, action):
        self.step_count += 1

        # ===============================
        # 1) Action and state update
        # ===============================
        a, delta_rate = action
        x, y, theta, v, delta = self.vehicle.update(a, delta_rate, self.dt)
        vehicle_poly = self.vehicle.get_vertices()

        slot = self.scene.parking_spaces[self.scene.target_idx]

        # ===============================
        # 2) Geometry metrics
        # ===============================
        # Distance.
        dist = np.linalg.norm(slot["center"] - np.array([x + self.vehicle.wheelbase/2 * np.cos(theta), y + self.vehicle.wheelbase/2 * np.sin(theta)]))

        # IoU.
        current_iou = self.compute_iou()

        # Collision.
        collision = self.scene.check_collision(vehicle_poly)

        # Success (geometry + speed threshold).
        success = self.scene.is_success(x, y, theta, v=v, v_tol=0.1)

        # ===============================
        # 3) Appendix B reward
        # ===============================
        reward = 0.0
        done = False

        # ---- (1) IoU incremental reward (core) ----
        delta_iou = max(current_iou - self.max_iou, 0.0)
        reward += 10.0 * delta_iou
        self.max_iou = max(self.max_iou, current_iou)

        # ---- (2) Distance reward ----
        r_dist = -(dist - self.init_dist) / max(self.init_dist, 1.0)
        delta_r_dist = max(r_dist - self.max_r_dist, 0.0)
        reward += 5.0 * delta_r_dist
        self.max_r_dist = max(self.max_r_dist, r_dist)

        # ---- (3) Time penalty ----
        time_penalty = -np.tanh(self.step_count / (10 * self.max_steps))
        reward += 0.05 * time_penalty

        # ---- (4) Heading alignment reward (distance-adaptive decay) ----
        angle_diff = abs(self.vehicle.normalize_angle(slot["theta"] - theta))
        angle_reward = abs(float(np.cos(angle_diff)))
        delta_angle_reward = angle_reward - self.pre_angle_reward
        reward += 0.5 * delta_angle_reward * r_dist
        self.pre_angle_reward = angle_reward

        # ---- (5) Collision penalty ----
        if collision:
            reward -= 5.0
            done = True

        # ---- (6) Success reward ----
        if success:
            # Base success reward.
            reward += 10.0
            done = True

        # ---- (7) Timeout termination ----
        truncated = False
        if self.step_count >= self.max_steps:
            truncated = True

        info = {"is_success": success}

        return self._get_state(), reward, done, truncated, info

    # =====================================================
    # IOU between vehicle and target slot
    # =====================================================
    def compute_iou(self):
        """Compute IoU of current vehicle footprint vs target parking slot."""
        vehicle_poly = self.vehicle.get_vertices()
        slot = self.scene.parking_spaces[self.scene.target_idx]
        slot_poly = self.scene._slot_polygon(slot)

        area_vehicle = self._polygon_area(np.array(vehicle_poly))
        area_slot = self._polygon_area(np.array(slot_poly))
        inter = self._overlap_area(vehicle_poly, slot_poly)
        union = area_vehicle + area_slot - inter
        if union <= 1e-9:
            return 0.0
        return inter / union

    # =====================================================
    # State
    # =====================================================
    def _get_state(self):
        x, y, theta, v, delta = self.vehicle.get_state()
        slot = self.scene.parking_spaces[self.scene.target_idx]

        dx, dy = slot["center"] - np.array([x + self.vehicle.wheelbase/2 * np.cos(theta), y + self.vehicle.wheelbase/2 * np.sin(theta)])
        dtheta = self.vehicle.normalize_angle(slot["theta"] - theta)

        state = np.array(
            [x, y, theta, v, delta, dx, dy, dtheta],
            dtype=np.float32,
        )
        return state

    # =====================================================
    # Geometry helpers
    # =====================================================
    @staticmethod
    def _polygon_area(poly):
        if len(poly) < 3:
            return 0.0
        x = poly[:, 0]
        y = poly[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    @staticmethod
    def _overlap_area(poly_1, poly_2):
        """Compute overlap area of two rectangles (AABB)."""
        x1_min, y1_min, x1_max, y1_max = ParkingEnv.get_aabb(np.array(poly_1))
        x2_min, y2_min, x2_max, y2_max = ParkingEnv.get_aabb(np.array(poly_2))
        
        # Compute overlap width and height.
        overlap_width = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        overlap_height = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        
        return overlap_width * overlap_height

    @staticmethod
    def get_aabb(poly):
        """Get axis-aligned bounding box (x_min, y_min, x_max, y_max)."""
        x_coords = poly[:, 0]
        y_coords = poly[:, 1]
        return x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max()
