import numpy as np
from shapely.geometry import Point, Polygon

class Obstacle:
    def __init__(self, vertices):
        """
        vertices: list of (x,y) coordinates defining the polygon.
        """
        self.vertices = np.array(vertices)
        self.poly = Polygon(vertices)

    def contains(self, point):
        """
        True if point is inside or on boundary of polygon.
        """
        return self.poly.contains(Point(point)) or self.poly.touches(Point(point))

    def distance_and_normal(self, point):
        """
        Returns (distance, normal_vector) from point to obstacle boundary.
        The normal points outward.
        """
        p = Point(point)
        d = self.poly.exterior.distance(p)
        # find closest boundary point
        closest_point = self.poly.exterior.interpolate(self.poly.exterior.project(p))
        closest = np.array([closest_point.x, closest_point.y])

        # normal vector: from obstacle towards the point
        normal = (point - closest)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            return d, np.zeros(2)
        return d, normal / norm

def random_regular_polygon(
    rng,
    center,
    n_sides=None,
    min_radius=5.0,
    max_radius=15.0,
    angle_offset=None
):
    """
    Creates a random regular convex polygon.
    
    Parameters:
    - center: (x,y)
    - n_sides: number of sides (if None, sampled in [7,20])
    - radius: circumradius of polygon
    """

    if n_sides is None:
        n_sides = rng.integers(7, 21)

    radius = rng.uniform(min_radius, max_radius)

    if angle_offset is None:
        angle_offset = rng.uniform(0, 2*np.pi)

    angles = angle_offset + np.linspace(0, 2*np.pi, n_sides, endpoint=False)

    xs = center[0] + radius * np.cos(angles)
    ys = center[1] + radius * np.sin(angles)

    vertices = np.column_stack((xs, ys))

    return Obstacle(vertices)