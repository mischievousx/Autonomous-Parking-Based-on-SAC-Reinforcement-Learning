# vehicle.py
import numpy as np
import math

class Vehicle:
    """
    Vehicle kinematic model (Rear-axle bicycle model)

    State:
    - x, y    : rear axle center position (m)
    - theta   : heading angle (rad), CCW positive
    - v       : longitudinal velocity (m/s), v < 0 for reverse
    - delta   : front steering angle (rad), CW positive (per experiment)

    Control:
    - a           : acceleration (m/s^2)
    - delta_rate  : steering rate (rad/s)
    """

    def __init__(self):
        # ========== Vehicle geometry ==========
        self.wheelbase = 2.7          # L, axle distance (m)
        self.front_overhang = 0.9     # front overhang (m)
        self.rear_overhang = 0.9      # rear overhang (m)
        self.width = 1.8              # vehicle width (m)

        # Total length (for reference / visualization)
        self.length = (
            self.wheelbase
            + self.front_overhang
            + self.rear_overhang
        )

        # ========== State ==========
        self.x = 0.0                  # rear axle x
        self.y = 0.0                  # rear axle y
        self.theta = 0.0              # heading (rad)
        self.v = 0.0                  # velocity (m/s)
        self.delta = 0.0              # steering angle (rad)

        # ========== Constraints ==========
        self.max_delta = math.radians(35)
        self.max_delta_rate = 1.0   # Faster steering rate (was 0.5).
        self.max_v = 1.0            # Higher max speed (was 0.5).
        self.max_a = 1.0            # Higher max acceleration (was 0.2).

    # --------------------------------------------------
    # Basic state operations
    # --------------------------------------------------

    def set_pose(self, x, y, theta):
        """Set vehicle pose (rear axle center)."""
        self.x = x
        self.y = y
        self.theta = self.normalize_angle(theta)

    def get_state(self):
        """Return current vehicle state."""
        return self.x, self.y, self.theta, self.v, self.delta

    # --------------------------------------------------
    # Kinematic update
    # --------------------------------------------------

    def update(self, a, delta_rate, dt=0.1):
        """
        Update vehicle state using rear-axle bicycle model.
        """

        # --- Input saturation ---
        a = np.clip(a, -self.max_a, self.max_a)
        delta_rate = np.clip(
            delta_rate,
            -self.max_delta_rate,
            self.max_delta_rate
        )

        # --- Update velocity ---
        self.v += a * dt
        self.v = np.clip(self.v, -self.max_v, self.max_v)

        # --- Update steering angle ---
        self.delta += delta_rate * dt
        self.delta = np.clip(self.delta, -self.max_delta, self.max_delta)

        # --- Kinematic model (rear axle) ---
        self.x += self.v * math.cos(self.theta) * dt
        self.y += self.v * math.sin(self.theta) * dt

        if abs(self.delta) > 1e-6:
            self.theta -= (self.v / self.wheelbase) * math.tan(self.delta) * dt

        self.theta = self.normalize_angle(self.theta)

        return self.get_state()

    # --------------------------------------------------
    # Geometry (for collision / visualization)
    # --------------------------------------------------

    def get_vertices(self):
        """
        Get vehicle rectangle vertices in world frame.
        Reference point: rear axle center.
        """

        half_w = self.width / 2.0

        # Local coordinates (rear axle at origin)
        local_vertices = np.array([
            [ self.wheelbase + self.front_overhang,  half_w],   # front-right
            [ self.wheelbase + self.front_overhang, -half_w],   # front-left
            [-self.rear_overhang, -half_w],                     # rear-left
            [-self.rear_overhang,  half_w]                      # rear-right
        ])

        # Rotation matrix
        c = math.cos(self.theta)
        s = math.sin(self.theta)
        R = np.array([[c, -s],
                      [s,  c]])

        # Transform to world frame
        vertices = local_vertices @ R.T
        vertices[:, 0] += self.x
        vertices[:, 1] += self.y

        return vertices

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

    @staticmethod
    def normalize_angle(angle):
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

if __name__ == "__main__":
    # Simple test
    vehicle = Vehicle()
    vehicle.set_pose(10.0, 10.0, 0.0)

    for _ in range(10):
        state = vehicle.update(a=1.0, delta_rate=0.1, dt=0.1)
        print(f"State: x={state[0]:.2f}, y={state[1]:.2f}, theta={math.degrees(state[2]):.2f} deg, v={state[3]:.2f} m/s, delta={math.degrees(state[4]):.2f} deg")