"""
cg_math_pipeline.py
===================
Supporting mathematics and computer-graphics algorithms for
solar_system_3d_tkinter.py.

Uses only the Python standard library.
"""

import math
from typing import List, Tuple

Vec3 = Tuple[float, float, float]
Mat4 = List[List[float]]


# ---------------------------------------------------------------------------
# Solar-system data
# Distances, radii and orbital values are illustrative simulation units.
# ---------------------------------------------------------------------------

CELESTIAL_BODIES = [
    {
        "name": "Sun", "tag": "Star", "au": "0 AU",
        "diameter": "1,392,700 km", "orbit_days": 0.0, "rot_days": 25.0,
        "tilt": 7.25, "info": "The star at the centre of our Solar System.",
        "distance": 0.0, "ecc": 0.0, "inc": 0.0, "radius": 10.0,
        "rgb": (255, 190, 70),
    },
    {
        "name": "Mercury", "tag": "Terrestrial", "au": "0.39 AU",
        "diameter": "4,879 km", "orbit_days": 87.97, "rot_days": 58.6,
        "tilt": 0.03, "info": "The smallest planet and the closest to the Sun.",
        "distance": 22.0, "ecc": 0.206, "inc": 7.0, "radius": 1.35,
        "rgb": (170, 155, 140),
    },
    {
        "name": "Venus", "tag": "Terrestrial", "au": "0.72 AU",
        "diameter": "12,104 km", "orbit_days": 224.7, "rot_days": -243.0,
        "tilt": 177.4, "info": "A hot planet with a dense carbon-dioxide atmosphere.",
        "distance": 32.0, "ecc": 0.007, "inc": 3.4, "radius": 2.0,
        "rgb": (225, 185, 115),
    },
    {
        "name": "Earth", "tag": "Terrestrial", "au": "1.00 AU",
        "diameter": "12,742 km", "orbit_days": 365.25, "rot_days": 1.0,
        "tilt": 23.44, "info": "Our home planet, with liquid water on its surface.",
        "distance": 45.0, "ecc": 0.017, "inc": 0.0, "radius": 2.15,
        "rgb": (65, 135, 235),
    },
    {
        "name": "Mars", "tag": "Terrestrial", "au": "1.52 AU",
        "diameter": "6,779 km", "orbit_days": 687.0, "rot_days": 1.03,
        "tilt": 25.19, "info": "The red planet, known for its dusty surface.",
        "distance": 59.0, "ecc": 0.093, "inc": 1.85, "radius": 1.7,
        "rgb": (220, 95, 65),
    },
    {
        "name": "Jupiter", "tag": "Gas Giant", "au": "5.20 AU",
        "diameter": "139,820 km", "orbit_days": 4331.0, "rot_days": 0.41,
        "tilt": 3.13, "info": "The largest planet, famous for its Great Red Spot.",
        "distance": 88.0, "ecc": 0.049, "inc": 1.3, "radius": 6.8,
        "rgb": (210, 170, 125),
    },
    {
        "name": "Saturn", "tag": "Gas Giant", "au": "9.58 AU",
        "diameter": "116,460 km", "orbit_days": 10747.0, "rot_days": 0.45,
        "tilt": 26.7, "info": "A gas giant recognised by its prominent ring system.",
        "distance": 119.0, "ecc": 0.057, "inc": 2.49, "radius": 5.8,
        "rgb": (220, 195, 135),
    },
    {
        "name": "Uranus", "tag": "Ice Giant", "au": "19.2 AU",
        "diameter": "50,724 km", "orbit_days": 30589.0, "rot_days": -0.72,
        "tilt": 97.8, "info": "An ice giant that rotates with an extreme axial tilt.",
        "distance": 151.0, "ecc": 0.046, "inc": 0.77, "radius": 4.0,
        "rgb": (125, 215, 220),
    },
    {
        "name": "Neptune", "tag": "Ice Giant", "au": "30.1 AU",
        "diameter": "49,244 km", "orbit_days": 59800.0, "rot_days": 0.67,
        "tilt": 28.3, "info": "A distant blue ice giant with very strong winds.",
        "distance": 181.0, "ecc": 0.010, "inc": 1.77, "radius": 3.9,
        "rgb": (75, 105, 225),
    },
    {
        "name": "Pluto", "tag": "Dwarf Planet", "au": "39.5 AU",
        "diameter": "2,377 km", "orbit_days": 90560.0, "rot_days": -6.39,
        "tilt": 122.5, "info": "A dwarf planet in the Kuiper Belt.",
        "distance": 207.0, "ecc": 0.249, "inc": 17.2, "radius": 1.15,
        "rgb": (190, 165, 145),
    },
]


# ---------------------------------------------------------------------------
# Vector and matrix mathematics
# Matrices are represented as row lists and multiply column vectors.
# ---------------------------------------------------------------------------

def vec3_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec3_cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def vec3_dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vec3_norm(v: Vec3) -> Vec3:
    length = math.sqrt(vec3_dot(v, v))
    if length <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def mat4_mul(a: Mat4, b: Mat4) -> Mat4:
    return [
        [sum(a[r][k] * b[k][c] for k in range(4)) for c in range(4)]
        for r in range(4)
    ]


def mat4_transform_point(m: Mat4, p: Vec3) -> Tuple[float, float, float, float]:
    x, y, z = p
    v = (x, y, z, 1.0)
    return tuple(sum(m[r][c] * v[c] for c in range(4)) for r in range(4))


def translation_matrix(tx: float, ty: float, tz: float) -> Mat4:
    return [
        [1.0, 0.0, 0.0, tx],
        [0.0, 1.0, 0.0, ty],
        [0.0, 0.0, 1.0, tz],
        [0.0, 0.0, 0.0, 1.0],
    ]


def rotation_y_matrix(angle: float) -> Mat4:
    c, s = math.cos(angle), math.sin(angle)
    return [
        [c, 0.0, s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def rotation_z_matrix(angle: float) -> Mat4:
    c, s = math.cos(angle), math.sin(angle)
    return [
        [c, -s, 0.0, 0.0],
        [s, c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def scaling_matrix(sx: float, sy: float, sz: float) -> Mat4:
    return [
        [sx, 0.0, 0.0, 0.0],
        [0.0, sy, 0.0, 0.0],
        [0.0, 0.0, sz, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def reflection_matrix(sx: float, sy: float, sz: float) -> Mat4:
    """Return reflection/sign-change matrix across selected coordinate planes."""
    return scaling_matrix(sx, sy, sz)


def look_at_matrix(eye: Vec3, target: Vec3, up: Vec3 = (0.0, 1.0, 0.0)) -> Mat4:
    forward = vec3_norm(vec3_sub(target, eye))
    side = vec3_norm(vec3_cross(forward, up))
    if side == (0.0, 0.0, 0.0):
        side = (1.0, 0.0, 0.0)
    true_up = vec3_cross(side, forward)
    return [
        [side[0], side[1], side[2], -vec3_dot(side, eye)],
        [true_up[0], true_up[1], true_up[2], -vec3_dot(true_up, eye)],
        [-forward[0], -forward[1], -forward[2], vec3_dot(forward, eye)],
        [0.0, 0.0, 0.0, 1.0],
    ]


def perspective_matrix(fov_degrees: float, aspect: float, near: float, far: float) -> Mat4:
    aspect = max(float(aspect), 1e-9)
    near = max(float(near), 1e-6)
    if far <= near:
        raise ValueError("far clipping plane must be greater than near clipping plane")
    f = 1.0 / math.tan(math.radians(fov_degrees) / 2.0)
    return [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) / (near - far), (2.0 * far * near) / (near - far)],
        [0.0, 0.0, -1.0, 0.0],
    ]


def orthographic_matrix(half_width: float, half_height: float,
                        near: float, far: float) -> Mat4:
    """Create a symmetric orthographic projection matrix.

    The main application supplies half-width, half-height, near and far.
    """
    if half_width == 0 or half_height == 0 or far == near:
        raise ValueError("orthographic bounds must have non-zero dimensions")
    return [
        [1.0 / half_width, 0.0, 0.0, 0.0],
        [0.0, 1.0 / half_height, 0.0, 0.0],
        [0.0, 0.0, -2.0 / (far - near), -(far + near) / (far - near)],
        [0.0, 0.0, 0.0, 1.0],
    ]


# ---------------------------------------------------------------------------
# Lighting and shading
# ---------------------------------------------------------------------------

def blinn_phong_shade(rgb, normal: Vec3, position: Vec3, light_pos: Vec3,
                      eye_pos: Vec3, mode: str = "phong") -> str:
    n = vec3_norm(normal)
    light = vec3_norm(vec3_sub(light_pos, position))
    view = vec3_norm(vec3_sub(eye_pos, position))
    ndotl = max(0.0, vec3_dot(n, light))

    mode = mode.lower()
    if mode == "unlit":
        intensity = 1.0
    elif mode == "normals":
        channels = [int(max(0.0, min(1.0, component * 0.5 + 0.5)) * 255)
                    for component in n]
        return "#{:02x}{:02x}{:02x}".format(*channels)
    elif mode == "lambert":
        intensity = 0.18 + 0.82 * ndotl
    elif mode == "toon":
        intensity = 0.25 if ndotl < 0.25 else 0.55 if ndotl < 0.6 else 1.0
    else:  # Blinn-Phong
        half_vec = vec3_norm((light[0] + view[0], light[1] + view[1], light[2] + view[2]))
        specular = max(0.0, vec3_dot(n, half_vec)) ** 24
        intensity = min(1.0, 0.16 + 0.72 * ndotl + 0.22 * specular)

    shaded = [max(0, min(255, int(component * intensity))) for component in rgb]
    return "#{:02x}{:02x}{:02x}".format(*shaded)


# ---------------------------------------------------------------------------
# Parametric curves
# ---------------------------------------------------------------------------

def _mix(a: Vec3, b: Vec3, t: float) -> Vec3:
    return tuple((1.0 - t) * a[i] + t * b[i] for i in range(3))


def eval_bezier(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: float) -> Vec3:
    u = 1.0 - t
    return tuple(
        u**3 * p0[i] + 3*u*u*t*p1[i] + 3*u*t*t*p2[i] + t**3*p3[i]
        for i in range(3)
    )


def eval_bspline(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: float) -> Vec3:
    t2, t3 = t * t, t * t * t
    b0 = (1 - 3*t + 3*t2 - t3) / 6.0
    b1 = (4 - 6*t2 + 3*t3) / 6.0
    b2 = (1 + 3*t + 3*t2 - 3*t3) / 6.0
    b3 = t3 / 6.0
    return tuple(b0*p0[i] + b1*p1[i] + b2*p2[i] + b3*p3[i] for i in range(3))


def eval_hermite(p0: Vec3, p1: Vec3, tangent0: Vec3,
                 tangent1: Vec3, t: float) -> Vec3:
    t2, t3 = t*t, t*t*t
    h00 = 2*t3 - 3*t2 + 1
    h10 = t3 - 2*t2 + t
    h01 = -2*t3 + 3*t2
    h11 = t3 - t2
    return tuple(
        h00*p0[i] + h10*tangent0[i] + h01*p1[i] + h11*tangent1[i]
        for i in range(3)
    )


# ---------------------------------------------------------------------------
# 2D rasterization algorithms
# ---------------------------------------------------------------------------

def dda_line(x1: int, y1: int, x2: int, y2: int) -> List[Tuple[int, int]]:
    dx, dy = x2 - x1, y2 - y1
    steps = max(abs(dx), abs(dy))
    if steps == 0:
        return [(int(round(x1)), int(round(y1)))]
    x_inc, y_inc = dx / steps, dy / steps
    x, y = float(x1), float(y1)
    points = []
    for _ in range(steps + 1):
        points.append((int(round(x)), int(round(y))))
        x += x_inc
        y += y_inc
    return points


def bresenham_line(x1: int, y1: int, x2: int, y2: int) -> List[Tuple[int, int]]:
    x, y = x1, y1
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    points = []
    while True:
        points.append((x, y))
        if x == x2 and y == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return points


def midpoint_circle(cx: int, cy: int, radius: int) -> List[Tuple[int, int]]:
    if radius < 0:
        raise ValueError("radius cannot be negative")
    x, y = 0, radius
    decision = 1 - radius
    points = set()
    while x <= y:
        points.update({
            (cx+x, cy+y), (cx-x, cy+y), (cx+x, cy-y), (cx-x, cy-y),
            (cx+y, cy+x), (cx-y, cy+x), (cx+y, cy-x), (cx-y, cy-x),
        })
        x += 1
        if decision < 0:
            decision += 2*x + 1
        else:
            y -= 1
            decision += 2*(x-y) + 1
    return sorted(points)


# Cohen-Sutherland outcode bits
_INSIDE, _LEFT, _RIGHT, _BOTTOM, _TOP = 0, 1, 2, 4, 8


def _clip_code(x: float, y: float, xmin: float, ymin: float,
               xmax: float, ymax: float) -> int:
    code = _INSIDE
    if x < xmin:
        code |= _LEFT
    elif x > xmax:
        code |= _RIGHT
    if y < ymin:
        code |= _BOTTOM
    elif y > ymax:
        code |= _TOP
    return code


def cohen_sutherland_clip(x1: float, y1: float, x2: float, y2: float,
                          xmin: float, ymin: float, xmax: float, ymax: float):
    """Clip a line; return clipped endpoints (x1,y1,x2,y2), or None if rejected."""
    if xmin > xmax or ymin > ymax:
        raise ValueError("invalid clipping rectangle")
    code1 = _clip_code(x1, y1, xmin, ymin, xmax, ymax)
    code2 = _clip_code(x2, y2, xmin, ymin, xmax, ymax)

    while True:
        if not (code1 | code2):
            return (x1, y1, x2, y2)
        if code1 & code2:
            return None

        out = code1 if code1 else code2
        if out & _TOP:
            if y2 == y1:
                return None
            x = x1 + (x2 - x1) * (ymax - y1) / (y2 - y1)
            y = ymax
        elif out & _BOTTOM:
            if y2 == y1:
                return None
            x = x1 + (x2 - x1) * (ymin - y1) / (y2 - y1)
            y = ymin
        elif out & _RIGHT:
            if x2 == x1:
                return None
            y = y1 + (y2 - y1) * (xmax - x1) / (x2 - x1)
            x = xmax
        else:  # LEFT
            if x2 == x1:
                return None
            y = y1 + (y2 - y1) * (xmin - x1) / (x2 - x1)
            x = xmin

        if out == code1:
            x1, y1 = x, y
            code1 = _clip_code(x1, y1, xmin, ymin, xmax, ymax)
        else:
            x2, y2 = x, y
            code2 = _clip_code(x2, y2, xmin, ymin, xmax, ymax)
