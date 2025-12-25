# scene.py
import numpy as np
import math

EPS = 1e-6


class ParkingScene:
    """
    Autonomous parking scene (for reinforcement learning).
    - Coordinate system: x right, y up.
    - Vehicle state: rear-axle center (x, y, theta).
    """

    def __init__(self):
        self.obstacles = []
        self.parking_spaces = []
        self.target_idx = None
        self.boundary = None

        # Vehicle parameters (rear-axle model).
        self.car_length = 4.5
        self.car_width = 1.8
        self.wheelbase = 2.7

    # =====================================================
    # Scene construction
    # =====================================================
    def build_scene(self, occupied=(True, False, True), layout="parallel"):
        """
        Build the parking scene.
        occupied: which slots are occupied.
        layout: parallel / vertical
        """
        self.obstacles.clear()
        self.parking_spaces.clear()
        self.target_idx = None

        assert len(occupied) == 3
        assert occupied.count(False) == 1

        # Slot size.
        slot_L = 6.0
        slot_W = 2.5

        if layout == "parallel":
            slot_theta = 0.0
            offset = slot_L
        elif layout == "vertical":
            slot_theta = math.pi / 2
            offset = slot_W
        else:
            raise ValueError("Unknown layout")

        # Create slots.
        for i in range(3):
            if layout == "parallel":
                cx = i * offset + slot_L / 2
                cy = slot_W / 2
            else:
                cx = i * offset + slot_W / 2
                cy = slot_L / 2

            slot = dict(
                center=np.array([cx, cy]),
                theta=slot_theta,
                length=slot_L,
                width=slot_W,
                index=i
            )
            self.parking_spaces.append(slot)

            if occupied[i]:
                obs = self._create_vehicle_obstacle(cx, cy, slot_theta)
                self.obstacles.append(obs)
            else:
                self.target_idx = i

        self._add_walls(layout, offset, slot_L)
        return self.parking_spaces[self.target_idx]

    # =====================================================
    # Geometry modeling
    # =====================================================
    def vehicle_polygon(self, x, y, theta):
        """
        Generate vehicle rectangle in world coordinates from rear-axle center.
        """
        rear = np.array([x, y])
        front = rear + self.wheelbase * np.array([math.cos(theta), math.sin(theta)])
        center = (rear + front) / 2

        hl = self.car_length / 2
        hw = self.car_width / 2

        local = np.array([
            [ hl,  hw],
            [ hl, -hw],
            [-hl, -hw],
            [-hl,  hw]
        ])

        c, s = math.cos(theta), math.sin(theta)
        R = np.array([[c, -s], [s, c]])
        poly = local @ R.T + center
        return poly

    def _create_vehicle_obstacle(self, cx, cy, theta):
        """
        Use a parked vehicle as an obstacle.
        """
        hl = self.car_length / 2
        hw = self.car_width / 2

        local = np.array([
            [ hl,  hw],
            [ hl, -hw],
            [-hl, -hw],
            [-hl,  hw]
        ])

        c, s = math.cos(theta), math.sin(theta)
        R = np.array([[c, -s], [s, c]])
        poly = local @ R.T + np.array([cx, cy])
        return poly

    # =====================================================
    # Walls
    # =====================================================
    def _add_walls(self, layout, offset, slot_L):
        t = 0.1
        Lx = 3 * offset + 5

        # Bottom wall.
        self.obstacles.append(np.array([
            [0, -t], [Lx, -t], [Lx, 0], [0, 0]
        ]))

        # Left wall.
        Ly = slot_L + 10
        self.obstacles.append(np.array([
            [-t, 0], [0, 0], [0, Ly], [-t, Ly]
        ]))

        self.boundary = (-2, Lx, -2, Ly)

    # =====================================================
    # Collision detection
    # =====================================================
    def check_collision(self, vehicle_poly):
        for obs in self.obstacles:
            if self._sat_intersect(vehicle_poly, obs):
                return True
        return False

    def _sat_intersect(self, A, B):
        def axes(poly):
            for i in range(len(poly)):
                e = poly[(i + 1) % len(poly)] - poly[i]
                n = np.array([-e[1], e[0]])
                n /= np.linalg.norm(n) + EPS
                yield n

        def proj(poly, axis):
            p = poly @ axis
            return p.min(), p.max()

        for axis in list(axes(A)) + list(axes(B)):
            a1, a2 = proj(A, axis)
            b1, b2 = proj(B, axis)
            if a2 < b1 or b2 < a1:
                return False
        return True

    # =====================================================
    # Success check (pi symmetry)
    # =====================================================
    def is_success(self, x, y, theta, v=None, v_tol=0.1):
        """
        Success if the vehicle is fully inside the slot and the speed is below the threshold (default 0.05 m/s).
        v: current speed; optional. If None, only geometry is checked.
        """
        slot = self.parking_spaces[self.target_idx]
        poly = self.vehicle_polygon(x, y, theta)
        slot_poly = self._slot_polygon(slot)

        for p in poly:
            if not self._point_in_poly(p, slot_poly):
                return False
        if v is not None and abs(v) > v_tol:
            return False
        return True

    def _slot_polygon(self, slot):
        hl = slot["length"] / 2
        hw = slot["width"] / 2

        local = np.array([
            [ hl,  hw],
            [ hl, -hw],
            [-hl, -hw],
            [-hl,  hw]
        ])

        c, s = math.cos(slot["theta"]), math.sin(slot["theta"])
        R = np.array([[c, -s], [s, c]])
        return local @ R.T + slot["center"]

    def _point_in_poly(self, p, poly):
        cnt = 0
        x, y = p
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            if (y1 > y) != (y2 > y):
                xin = (y - y1) * (x2 - x1) / (y2 - y1 + EPS) + x1
                if x < xin:
                    cnt += 1
        return cnt % 2 == 1
