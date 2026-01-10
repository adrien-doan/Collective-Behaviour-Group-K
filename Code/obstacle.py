import numpy as np
from shapely.geometry import Point, Polygon, LineString

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
    
    def blocks_segment(self, p1, p2):
        """
        Returns True if the line segment p1 -> p2 intersects the obstacle.
        """
        segment = LineString([p1, p2])
        return segment.intersects(self.poly)
    
    def ray_intersection_normal(self, origin, direction, max_dist):
        """
        Casts a ray from origin along direction.
        If it intersects the obstacle within max_dist,
        returns (distance, normal_at_intersection).
        Otherwise returns (None, None).
        """
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        end = origin + max_dist * direction

        ray = LineString([origin, end])
        inter = ray.intersection(self.poly.exterior)

        if inter.is_empty:
            return None, None

        # Handle multiple intersection types
        if isinstance(inter, Point):
            ip = inter
        else:
            # Take closest intersection point
            ip = min(
                inter.geoms,
                key=lambda p: np.linalg.norm(
                    np.array([p.x, p.y]) - origin
                )
            )

        hit_point = np.array([ip.x, ip.y])

        # Compute normal at intersection
        # Use small offset to approximate tangent
        eps = 1e-6
        proj = self.poly.exterior.project(ip)
        p1 = self.poly.exterior.interpolate(proj - eps)
        p2 = self.poly.exterior.interpolate(proj + eps)

        tangent = np.array([p2.x - p1.x, p2.y - p1.y])
        tangent /= np.linalg.norm(tangent) + 1e-12

        # Rotate tangent to get outward normal
        normal = np.array([-tangent[1], tangent[0]])

        # Ensure normal points outward
        if np.dot(normal, direction) > 0:
            normal = -normal

        dist = np.linalg.norm(hit_point - origin)
        return dist, normal

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