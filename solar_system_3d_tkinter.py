"""
solar_system_3d_tkinter.py
==========================
Zero-Dependency Python 3D Solar System Simulator (Built with standard library tkinter + math)
Computer Graphics Final Submission Project (5.5CE109E02H)

Features:
  - Custom 3D Software Rendering Pipeline (M -> V -> P -> NDC -> Viewport)
  - 3D Sphere Mesh Generation, Back-Face Culling & Painter's Z-Sorting
  - Blinn-Phong, Lambertian, Toon/Cel, Surface Normals & Wireframe Shading
  - Perspective vs Orthographic Projection + Dynamic FOV + Mouse Orbit/Zoom
  - 3D Reflection Matrices (XY, YZ, XZ) & Live 4x4 Matrix Inspector
  - Unit 5 Parametric Transfer Curve (Cubic Bezier / B-Spline / Hermite)
  - Built-in Unit 1-3 2D Algorithm Lab (DDA, Bresenham, Midpoint Circle, Cohen-Sutherland)
"""

import math
import tkinter as tk
from tkinter import ttk
from typing import List, Tuple, Dict, Any

from cg_math_pipeline import (
    CELESTIAL_BODIES,
    Vec3,
    Mat4,
    mat4_mul,
    mat4_transform_point,
    translation_matrix,
    rotation_y_matrix,
    rotation_z_matrix,
    scaling_matrix,
    reflection_matrix,
    look_at_matrix,
    perspective_matrix,
    orthographic_matrix,
    vec3_sub,
    vec3_cross,
    vec3_dot,
    vec3_norm,
    blinn_phong_shade,
    eval_bezier,
    eval_bspline,
    eval_hermite,
    dda_line,
    bresenham_line,
    midpoint_circle,
    cohen_sutherland_clip,
)


def build_uv_sphere(stacks: int = 10, slices: int = 14) -> Tuple[List[Vec3], List[Tuple[int, int, int]]]:
    """Generates unit sphere vertices and triangle indices for 3D software rasterization."""
    verts: List[Vec3] = []
    tris: List[Tuple[int, int, int]] = []
    for i in range(stacks + 1):
        phi = math.pi * i / stacks
        y = math.cos(phi)
        r = math.sin(phi)
        for j in range(slices + 1):
            theta = 2.0 * math.pi * j / slices
            x = r * math.cos(theta)
            z = r * math.sin(theta)
            verts.append((x, y, z))

    stride = slices + 1
    for i in range(stacks):
        for j in range(slices):
            a = i * stride + j
            b = a + 1
            c = (i + 1) * stride + j
            d = c + 1
            if i != 0:
                tris.append((a, c, b))
            if i != stacks - 1:
                tris.append((b, c, d))
    return verts, tris


class SolarSystemPythonApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SOLARIS 3D (Python) — Computer Graphics Final Project (5.5CE109E02H)")
        self.root.geometry("1360x820")
        self.root.configure(bg="#020617")

        # Simulation State
        self.is_playing = True
        self.orbit_dir = 1.0
        self.speed = 1.0
        self.scale_k = 1.0
        self.reflect_sign = [1, 1, 1]
        self.proj_mode = "perspective"  # 'perspective' or 'orthographic'
        self.fov = 50.0
        self.shading_mode = "phong"     # 'phong', 'lambert', 'toon', 'normals', 'wireframe', 'unlit'
        self.selected_name = "Earth"

        # Camera Spherical Coordinates
        self.cam_yaw = 0.65
        self.cam_pitch = 0.48
        self.cam_dist = 240.0
        self.cam_target: Vec3 = (0.0, 0.0, 0.0)
        self.drag_last = None

        # Visual Overlays
        self.var_orbits = tk.BooleanVar(value=True)
        self.var_grid = tk.BooleanVar(value=True)
        self.var_axes = tk.BooleanVar(value=False)
        self.var_light_ray = tk.BooleanVar(value=False)
        self.var_curve = tk.BooleanVar(value=False)
        self.curve_type = tk.StringVar(value="bezier")
        self.curve_t = 0.0

        # Geometry Cache
        self.sphere_verts, self.sphere_tris = build_uv_sphere(stacks=9, slices=12)
        self.angles: Dict[str, float] = {b["name"]: idx * 0.7 for idx, b in enumerate(CELESTIAL_BODIES)}
        self.spins: Dict[str, float] = {b["name"]: 0.0 for b in CELESTIAL_BODIES}
        self.world_positions: Dict[str, Vec3] = {}
        self.last_matrices: Dict[str, Mat4] = {}

        self._build_ui()
        self._select_body("Earth")
        self._tick()

    def _build_ui(self):
        # Top Header Bar
        header = tk.Frame(self.root, bg="#0f172a", height=48, bd=0)
        header.pack(side=tk.TOP, fill=tk.X)
        tk.Label(
            header, text="SOLARIS 3D  |  Python Software 3D Pipeline (5.5CE109E02H)",
            bg="#0f172a", fg="#38bdf8", font=("Segoe UI", 12, "bold"), padx=14
        ).pack(side=tk.LEFT, pady=10)

        tk.Button(
            header, text="2D Algo Lab (Units 1–3)", command=self._open_2d_lab,
            bg="#059669", fg="white", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, padx=10, pady=4
        ).pack(side=tk.RIGHT, padx=8, pady=8)

        tk.Button(
            header, text="Top View", command=lambda: self._set_cam(0.0, 1.48, 280),
            bg="#1e293b", fg="#e2e8f0", font=("Segoe UI", 9), relief=tk.FLAT, padx=8, pady=4
        ).pack(side=tk.RIGHT, padx=4, pady=8)

        tk.Button(
            header, text="Side View", command=lambda: self._set_cam(0.0, 0.05, 290),
            bg="#1e293b", fg="#e2e8f0", font=("Segoe UI", 9), relief=tk.FLAT, padx=8, pady=4
        ).pack(side=tk.RIGHT, padx=4, pady=8)

        tk.Button(
            header, text="Free Orbit", command=lambda: self._set_cam(0.65, 0.48, 240),
            bg="#2563eb", fg="white", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, padx=8, pady=4
        ).pack(side=tk.RIGHT, padx=4, pady=8)

        # Main 3-Column Body
        main_frame = tk.Frame(self.root, bg="#020617")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left Control Workbench
        left = tk.Frame(main_frame, bg="#0b1120", width=270, padx=10, pady=10)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        self._add_section_label(left, "SEC 01: 3D TRANSFORMATIONS (M = T·R·S)")
        btn_row = tk.Frame(left, bg="#0b1120")
        btn_row.pack(fill=tk.X, pady=4)
        self.btn_play = tk.Button(btn_row, text="Pause", command=self._toggle_play, bg="#2563eb", fg="white", width=8, relief=tk.FLAT)
        self.btn_play.pack(side=tk.LEFT, padx=2)
        tk.Button(btn_row, text="Reverse", command=self._reverse_dir, bg="#1e293b", fg="white", width=8, relief=tk.FLAT).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_row, text="Reset", command=self._reset_sim, bg="#1e293b", fg="white", width=8, relief=tk.FLAT).pack(side=tk.LEFT, padx=2)

        tk.Label(left, text="Orbital Speed (0.2x – 10x)", bg="#0b1120", fg="#cbd5e1", font=("Segoe UI", 9)).pack(anchor="w", pady=(6, 0))
        self.slider_speed = tk.Scale(left, from_=0.2, to=10.0, resolution=0.2, orient=tk.HORIZONTAL, bg="#0b1120", fg="#38bdf8", highlightthickness=0, command=self._on_speed)
        self.slider_speed.set(1.0)
        self.slider_speed.pack(fill=tk.X)

        tk.Label(left, text="Uniform Scale Factor (S_k)", bg="#0b1120", fg="#cbd5e1", font=("Segoe UI", 9)).pack(anchor="w", pady=(4, 0))
        self.slider_scale = tk.Scale(left, from_=0.5, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, bg="#0b1120", fg="#34d399", highlightthickness=0, command=self._on_scale)
        self.slider_scale.set(1.0)
        self.slider_scale.pack(fill=tk.X)

        ref_row = tk.Frame(left, bg="#0b1120")
        ref_row.pack(fill=tk.X, pady=4)
        tk.Button(ref_row, text="Reflect XY", command=lambda: self._toggle_reflect(2), bg="#1e293b", fg="#fda4af", font=("Segoe UI", 8), relief=tk.FLAT).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        tk.Button(ref_row, text="Reflect YZ", command=lambda: self._toggle_reflect(0), bg="#1e293b", fg="#fda4af", font=("Segoe UI", 8), relief=tk.FLAT).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        tk.Button(ref_row, text="Reflect XZ", command=lambda: self._toggle_reflect(1), bg="#1e293b", fg="#fda4af", font=("Segoe UI", 8), relief=tk.FLAT).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        self._add_section_label(left, "SEC 02: PROJECTION & CAMERA (P·V)")
        self.btn_proj = tk.Button(left, text="Projection: Perspective (50°)", command=self._toggle_proj, bg="#1e293b", fg="#67e8f9", relief=tk.FLAT, pady=4)
        self.btn_proj.pack(fill=tk.X, pady=4)
        self.slider_fov = tk.Scale(left, from_=25, to=90, resolution=1, orient=tk.HORIZONTAL, bg="#0b1120", fg="#c084fc", highlightthickness=0, label="Perspective FOV (deg)", command=self._on_fov)
        self.slider_fov.set(50)
        self.slider_fov.pack(fill=tk.X)

        self._add_section_label(left, "SEC 03: ILLUMINATION & SHADING")
        self.combo_shade = ttk.Combobox(left, values=["phong", "lambert", "toon", "normals", "wireframe", "unlit"], state="readonly")
        self.combo_shade.set("phong")
        self.combo_shade.bind("<<ComboboxSelected>>", lambda e: setattr(self, "shading_mode", self.combo_shade.get()))
        self.combo_shade.pack(fill=tk.X, pady=4)

        self._add_section_label(left, "SEC 04: GEOMETRY & UNIT 5 CURVES")
        for txt, var in [
            ("Parametric Orbit Ellipses", self.var_orbits),
            ("Ecliptic Reference Grid (XZ)", self.var_grid),
            ("Local 3D Axis Frames (RGB)", self.var_axes),
            ("Incident Light Vector Ray (L)", self.var_light_ray),
            ("Unit 5: Parametric Curve Probe", self.var_curve),
        ]:
            tk.Checkbutton(
                left, text=txt, variable=var, bg="#0b1120", fg="#e2e8f0",
                selectcolor="#1e293b", activebackground="#0b1120", activeforeground="white", anchor="w"
            ).pack(fill=tk.X)

        self.combo_curve = ttk.Combobox(left, textvariable=self.curve_type, values=["bezier", "bspline", "hermite"], state="readonly")
        self.combo_curve.pack(fill=tk.X, pady=4)

        # Right Inspector Panel (Planet Info + 4x4 Matrix)
        right = tk.Frame(main_frame, bg="#0b1120", width=290, padx=10, pady=10)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        self._add_section_label(right, "PLANET DOSSIER & TELEMETRY")
        self.lbl_body_title = tk.Label(right, text="Earth (Terrestrial)", bg="#0b1120", fg="white", font=("Segoe UI", 12, "bold"), anchor="w")
        self.lbl_body_title.pack(fill=tk.X, pady=(4, 2))
        self.lbl_body_stats = tk.Label(right, text="", bg="#0f172a", fg="#93c5fd", font=("Consolas", 9), justify=tk.LEFT, anchor="w", padx=8, pady=6)
        self.lbl_body_stats.pack(fill=tk.X, pady=4)
        self.lbl_body_desc = tk.Label(right, text="", bg="#0b1120", fg="#cbd5e1", font=("Segoe UI", 9), wraplength=265, justify=tk.LEFT, anchor="w")
        self.lbl_body_desc.pack(fill=tk.X, pady=4)

        self._add_section_label(right, "LIVE 4x4 MODEL MATRIX (M)")
        self.txt_matrix_m = tk.Label(right, text="", bg="#020617", fg="#38bdf8", font=("Consolas", 9), justify=tk.LEFT, padx=8, pady=6)
        self.txt_matrix_m.pack(fill=tk.X, pady=4)

        self._add_section_label(right, "LIVE 4x4 VIEW MATRIX (V)")
        self.txt_matrix_v = tk.Label(right, text="", bg="#020617", fg="#34d399", font=("Consolas", 9), justify=tk.LEFT, padx=8, pady=6)
        self.txt_matrix_v.pack(fill=tk.X, pady=4)

        self._add_section_label(right, "LIVE 4x4 PROJECTION MATRIX (P)")
        self.txt_matrix_p = tk.Label(right, text="", bg="#020617", fg="#fbbf24", font=("Consolas", 9), justify=tk.LEFT, padx=8, pady=6)
        self.txt_matrix_p.pack(fill=tk.X, pady=4)

        # Center 3D Viewport Canvas
        center = tk.Frame(main_frame, bg="#020617")
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(center, bg="#020617", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._zoom(-15))
        self.canvas.bind("<Button-5>", lambda e: self._zoom(15))

        # Bottom Planet Selector Dock
        dock = tk.Frame(center, bg="#0f172a", height=42)
        dock.pack(side=tk.BOTTOM, fill=tk.X)
        for b in CELESTIAL_BODIES:
            name = b["name"]
            tk.Button(
                dock, text=name, command=lambda n=name: self._select_body(n),
                bg="#1e293b", fg="white", font=("Segoe UI", 8, "bold"), relief=tk.FLAT, padx=8, pady=4
            ).pack(side=tk.LEFT, padx=3, pady=5)

    def _add_section_label(self, parent, text):
        tk.Label(
            parent, text=text, bg="#0b1120", fg="#38bdf8",
            font=("Consolas", 8, "bold"), anchor="w", pady=4
        ).pack(fill=tk.X, pady=(6, 1))

    # --- UI Callbacks ---
    def _toggle_play(self):
        self.is_playing = not self.is_playing
        self.btn_play.config(text="Pause" if self.is_playing else "Play")

    def _reverse_dir(self):
        self.orbit_dir *= -1.0

    def _reset_sim(self):
        for idx, b in enumerate(CELESTIAL_BODIES):
            self.angles[b["name"]] = idx * 0.7
            self.spins[b["name"]] = 0.0
        self.reflect_sign = [1, 1, 1]

    def _on_speed(self, val):
        self.speed = float(val)

    def _on_scale(self, val):
        self.scale_k = float(val)

    def _on_fov(self, val):
        self.fov = float(val)
        if self.proj_mode == "perspective":
            self.btn_proj.config(text=f"Projection: Perspective ({int(self.fov)}°)")

    def _toggle_reflect(self, axis_idx: int):
        self.reflect_sign[axis_idx] *= -1

    def _toggle_proj(self):
        if self.proj_mode == "perspective":
            self.proj_mode = "orthographic"
            self.btn_proj.config(text="Projection: Orthographic")
        else:
            self.proj_mode = "perspective"
            self.btn_proj.config(text=f"Projection: Perspective ({int(self.fov)}°)")

    def _set_cam(self, yaw: float, pitch: float, dist: float):
        self.cam_yaw, self.cam_pitch, self.cam_dist = yaw, pitch, dist

    def _select_body(self, name: str):
        self.selected_name = name
        body = next(b for b in CELESTIAL_BODIES if b["name"] == name)
        self.lbl_body_title.config(text=f"{body['name']} ({body['tag']})")
        self.lbl_body_stats.config(
            text=(
                f"Distance : {body['au']}\n"
                f"Diameter : {body['diameter']}\n"
                f"Orbit T  : {body['orbit_days']} days\n"
                f"Tilt (Rz): {body['tilt']}°"
            )
        )
        self.lbl_body_desc.config(text=body["info"])

    def _on_mouse_down(self, event):
        self.drag_last = (event.x, event.y)

    def _on_mouse_drag(self, event):
        if self.drag_last:
            dx = event.x - self.drag_last[0]
            dy = event.y - self.drag_last[1]
            self.cam_yaw -= dx * 0.008
            self.cam_pitch = max(-1.5, min(1.5, self.cam_pitch + dy * 0.008))
            self.drag_last = (event.x, event.y)

    def _on_mouse_wheel(self, event):
        self._zoom(-15 if event.delta > 0 else 15)

    def _zoom(self, delta: float):
        self.cam_dist = max(40.0, min(650.0, self.cam_dist + delta))

    # --- 3D Projection Helper ---
    def _project_point(self, p: Vec3, VP: Mat4, w: int, h: int) -> Tuple[float, float, float, bool]:
        cx, cy, cz, cw = mat4_transform_point(VP, p)
        if cw <= 1e-4:
            return (0.0, 0.0, 0.0, False)
        ndc_x = cx / cw
        ndc_y = cy / cw
        ndc_z = cz / cw
        if ndc_z < -1.0 or ndc_z > 1.0:
            return (0.0, 0.0, 0.0, False)
        sx = (ndc_x * 0.5 + 0.5) * w
        sy = (1.0 - (ndc_y * 0.5 + 0.5)) * h
        return (sx, sy, ndc_z, True)

    def _format_mat4(self, M: Mat4) -> str:
        return "\n".join(" ".join(f"{M[r][c]:6.2f}" for c in range(4)) for r in range(4))

    # --- Main 60 FPS Render Loop ---
    def _tick(self):
        dt = 0.033
        if self.is_playing:
            for b in CELESTIAL_BODIES:
                name = b["name"]
                if b["orbit_days"] > 0:
                    omega = (365.25 / b["orbit_days"]) * 0.35
                    self.angles[name] += omega * self.speed * self.orbit_dir * dt
                spin_dir = -1.0 if b["rot_days"] < 0 else 1.0
                self.spins[name] += (1.0 / max(abs(b["rot_days"]), 0.5)) * spin_dir * self.speed * dt
            self.curve_t = (self.curve_t + dt * 0.25) % 1.0

        self._render_frame()
        self.root.after(33, self._tick)

    def _render_frame(self):
        c = self.canvas
        w = max(c.winfo_width(), 640)
        h = max(c.winfo_height(), 480)
        c.delete("all")

        # Update World Positions
        for b in CELESTIAL_BODIES:
            name = b["name"]
            a = b["distance"]
            e = b["ecc"]
            inc = math.radians(b["inc"])
            th = self.angles[name]
            if a == 0.0:
                self.world_positions[name] = (0.0, 0.0, 0.0)
            else:
                b_semi = a * math.sqrt(max(1.0 - e * e, 0.01))
                x = a * math.cos(th) - a * e
                z = b_semi * math.sin(th)
                y = math.sin(th) * math.sin(inc) * (a * 0.35)
                self.world_positions[name] = (x, y, z)

        self.cam_target = self.world_positions.get(self.selected_name, (0.0, 0.0, 0.0))
        eye = (
            self.cam_target[0] + self.cam_dist * math.cos(self.cam_pitch) * math.sin(self.cam_yaw),
            self.cam_target[1] + self.cam_dist * math.sin(self.cam_pitch),
            self.cam_target[2] + self.cam_dist * math.cos(self.cam_pitch) * math.cos(self.cam_yaw),
        )

        V = look_at_matrix(eye, self.cam_target)
        aspect = w / float(h)
        if self.proj_mode == "perspective":
            P = perspective_matrix(self.fov, aspect, 1.0, 2000.0)
        else:
            half_h = self.cam_dist * 0.45
            P = orthographic_matrix(half_h * aspect, half_h, 1.0, 2000.0)
        VP = mat4_mul(P, V)

        # 1. Draw XZ Ecliptic Reference Grid
        if self.var_grid.get():
            for g in range(-220, 221, 44):
                p1 = self._project_point((g, -0.5, -220), VP, w, h)
                p2 = self._project_point((g, -0.5, 220), VP, w, h)
                if p1[3] and p2[3]:
                    c.create_line(p1[0], p1[1], p2[0], p2[1], fill="#0f172a", width=1)
                p3 = self._project_point((-220, -0.5, g), VP, w, h)
                p4 = self._project_point((220, -0.5, g), VP, w, h)
                if p3[3] and p4[3]:
                    c.create_line(p3[0], p3[1], p4[0], p4[1], fill="#0f172a", width=1)

        # 2. Draw Parametric Orbit Ellipses
        if self.var_orbits.get():
            for b in CELESTIAL_BODIES:
                a = b["distance"]
                if a <= 0:
                    continue
                e = b["ecc"]
                b_semi = a * math.sqrt(max(1.0 - e * e, 0.01))
                inc = math.radians(b["inc"])
                pts_2d = []
                for i in range(65):
                    th = 2.0 * math.pi * i / 64
                    px = a * math.cos(th) - a * e
                    pz = b_semi * math.sin(th)
                    py = math.sin(th) * math.sin(inc) * (a * 0.35)
                    sp = self._project_point((px, py, pz), VP, w, h)
                    if sp[3]:
                        pts_2d.extend([sp[0], sp[1]])
                if len(pts_2d) >= 4:
                    col = "#38bdf8" if b["name"] == self.selected_name else "#1e3a8a"
                    c.create_line(*pts_2d, fill=col, width=1)

        # 3. Draw Unit 5 Parametric Transfer Curve (Earth -> Mars)
        if self.var_curve.get():
            p0 = self.world_positions["Earth"]
            p3 = self.world_positions["Mars"]
            mid = ((p0[0] + p3[0]) * 0.5, (p0[1] + p3[1]) * 0.5, (p0[2] + p3[2]) * 0.5)
            p1 = (mid[0] + 10.0, mid[1] + 28.0, mid[2] + 15.0)
            p2 = (mid[0] - 12.0, mid[1] - 20.0, mid[2] - 18.0)
            ctype = self.curve_type.get()
            curve_2d = []
            for i in range(41):
                t = i / 40.0
                if ctype == "bezier":
                    pt = eval_bezier(p0, p1, p2, p3, t)
                elif ctype == "bspline":
                    pt = eval_bspline(p0, p1, p2, p3, t)
                else:
                    pt = eval_hermite(p0, p3, (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]), (p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]), t)
                sp = self._project_point(pt, VP, w, h)
                if sp[3]:
                    curve_2d.extend([sp[0], sp[1]])
            if len(curve_2d) >= 4:
                c.create_line(*curve_2d, fill="#a855f7", width=2)
            # Probe position
            if ctype == "bezier":
                pr = eval_bezier(p0, p1, p2, p3, self.curve_t)
            elif ctype == "bspline":
                pr = eval_bspline(p0, p1, p2, p3, self.curve_t)
            else:
                pr = eval_hermite(p0, p3, (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]), (p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]), self.curve_t)
            psp = self._project_point(pr, VP, w, h)
            if psp[3]:
                c.create_oval(psp[0]-4, psp[1]-4, psp[0]+4, psp[1]+4, fill="#fef08a", outline="white")

        # 4. Transform & Collect 3D Mesh Triangles for Painter's Z-Sort
        draw_tris = []
        for b in CELESTIAL_BODIES:
            name = b["name"]
            pos = self.world_positions[name]
            r = b["radius"] * self.scale_k

            T = translation_matrix(*pos)
            Rz = rotation_z_matrix(math.radians(b["tilt"]))
            Ry = rotation_y_matrix(self.spins[name])
            S = scaling_matrix(r * self.reflect_sign[0], r * self.reflect_sign[1], r * self.reflect_sign[2])

            M = mat4_mul(T, mat4_mul(Rz, mat4_mul(Ry, S)))
            self.last_matrices[name] = M

            # Transform sphere vertices to world & screen space
            world_vs = []
            screen_vs = []
            for v in self.sphere_verts:
                wx, wy, wz, _ = mat4_transform_point(M, v)
                world_vs.append((wx, wy, wz))
                screen_vs.append(self._project_point((wx, wy, wz), VP, w, h))

            for i0, i1, i2 in self.sphere_tris:
                s0, s1, s2 = screen_vs[i0], screen_vs[i1], screen_vs[i2]
                if not (s0[3] and s1[3] and s2[3]):
                    continue
                # 2D Back-face culling
                cross2d = (s1[0] - s0[0]) * (s2[1] - s0[1]) - (s1[1] - s0[1]) * (s2[0] - s0[0])
                if cross2d <= 0 and self.shading_mode != "wireframe":
                    continue
                w0, w1, w2 = world_vs[i0], world_vs[i1], world_vs[i2]
                avg_z = (s0[2] + s1[2] + s2[2]) / 3.0
                if name == "Sun":
                    fill_col = "#f59e0b"
                else:
                    nrm = vec3_norm(vec3_cross(vec3_sub(w1, w0), vec3_sub(w2, w0)))
                    centroid = ((w0[0]+w1[0]+w2[0])/3.0, (w0[1]+w1[1]+w2[1])/3.0, (w0[2]+w1[2]+w2[2])/3.0)
                    fill_col = blinn_phong_shade(b["rgb"], nrm, centroid, (0.0, 0.0, 0.0), eye, mode=self.shading_mode)
                draw_tris.append((avg_z, s0, s1, s2, fill_col))

        # Painter's Algorithm (Back-to-Front)
        draw_tris.sort(key=lambda item: item[0], reverse=True)
        for _, s0, s1, s2, fill_col in draw_tris:
            if self.shading_mode == "wireframe":
                c.create_polygon(s0[0], s0[1], s1[0], s1[1], s2[0], s2[1], fill="", outline="#38bdf8")
            else:
                c.create_polygon(s0[0], s0[1], s1[0], s1[1], s2[0], s2[1], fill=fill_col, outline=fill_col)

        # 5. Draw Local RGB Axes & Incident Light Ray + Planet Labels
        if self.var_light_ray.get() and self.selected_name != "Sun":
            sp0 = self._project_point((0.0, 0.0, 0.0), VP, w, h)
            sp1 = self._project_point(self.world_positions[self.selected_name], VP, w, h)
            if sp0[3] and sp1[3]:
                c.create_line(sp0[0], sp0[1], sp1[0], sp1[1], fill="#fde047", dash=(5, 4), width=2)

        for b in CELESTIAL_BODIES:
            name = b["name"]
            pos = self.world_positions[name]
            if self.var_axes.get():
                alen = b["radius"] * self.scale_k * 2.2
                o = self._project_point(pos, VP, w, h)
                ax = self._project_point((pos[0]+alen, pos[1], pos[2]), VP, w, h)
                ay = self._project_point((pos[0], pos[1]+alen, pos[2]), VP, w, h)
                az = self._project_point((pos[0], pos[1], pos[2]+alen), VP, w, h)
                if o[3]:
                    if ax[3]: c.create_line(o[0], o[1], ax[0], ax[1], fill="#f43f5e", width=2)
                    if ay[3]: c.create_line(o[0], o[1], ay[0], ay[1], fill="#22c55e", width=2)
                    if az[3]: c.create_line(o[0], o[1], az[0], az[1], fill="#3b82f6", width=2)

            lbl_pos = self._project_point((pos[0], pos[1] + b["radius"] * self.scale_k + 2.5, pos[2]), VP, w, h)
            if lbl_pos[3]:
                col = "#38bdf8" if name == self.selected_name else "#cbd5e1"
                c.create_text(lbl_pos[0], lbl_pos[1], text=name, fill=col, font=("Consolas", 8, "bold"))

        # 6. Update Live 4x4 Matrix Inspector Readouts
        M_sel = self.last_matrices.get(self.selected_name, [[0]*4 for _ in range(4)])
        self.txt_matrix_m.config(text=self._format_mat4(M_sel))
        self.txt_matrix_v.config(text=self._format_mat4(V))
        self.txt_matrix_p.config(text=self._format_mat4(P))

    # --- Unit 1-3: 2D Rasterization & Clipping Lab Window ---
    def _open_2d_lab(self):
        win = tk.Toplevel(self.root)
        win.title("2D Computer Graphics Algorithm Lab (Syllabus Units 1–3)")
        win.geometry("760x540")
        win.configure(bg="#0b1120")

        ctrl = tk.Frame(win, bg="#0f172a", width=240, padx=12, pady=12)
        ctrl.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(ctrl, text="Select Algorithm (Units 1–3)", bg="#0f172a", fg="#38bdf8", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 6))
        algo_var = tk.StringVar(value="DDA Line (Unit 1.3)")
        combo = ttk.Combobox(ctrl, textvariable=algo_var, values=[
            "DDA Line (Unit 1.3)",
            "Bresenham Line (Unit 1.3)",
            "Midpoint Circle (Unit 1.4)",
            "Cohen-Sutherland Clipping (Unit 3.3)"
        ], state="readonly")
        combo.pack(fill=tk.X, pady=4)

        entries = {}
        for lbl, default in [("X1 / Cx", "3"), ("Y1 / Cy", "4"), ("X2 / Radius", "25"), ("Y2", "22")]:
            f = tk.Frame(ctrl, bg="#0f172a")
            f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=lbl, bg="#0f172a", fg="#cbd5e1", width=12, anchor="w").pack(side=tk.LEFT)
            e = tk.Entry(f, bg="#020617", fg="white", insertbackground="white", width=8)
            e.insert(0, default)
            e.pack(side=tk.RIGHT)
            entries[lbl] = e

        log_lbl = tk.Label(ctrl, text="Trace ready.", bg="#020617", fg="#a7f3d0", font=("Consolas", 8), justify=tk.LEFT, anchor="nw", padx=6, pady=6, height=14, width=30)
        log_lbl.pack(fill=tk.BOTH, expand=True, pady=8)

        cv = tk.Canvas(win, bg="#020617", width=480, height=480, highlightthickness=1, highlightbackground="#334155")
        cv.pack(side=tk.RIGHT, padx=14, pady=14)

        def run_algo():
            cv.delete("all")
            N = 30
            cell = 480 / N
            for i in range(N + 1):
                cv.create_line(i * cell, 0, i * cell, 480, fill="#1e293b")
                cv.create_line(0, i * cell, 480, i * cell, fill="#1e293b")

            def plot(gx, gy, col="#38bdf8"):
                if 0 <= gx < N and 0 <= gy < N:
                    py = (N - 1 - gy) * cell
                    cv.create_rectangle(gx * cell + 1, py + 1, (gx + 1) * cell - 1, py + cell - 1, fill=col, outline="")

            x1 = int(entries["X1 / Cx"].get() or 3)
            y1 = int(entries["Y1 / Cy"].get() or 4)
            x2 = int(entries["X2 / Radius"].get() or 25)
            y2 = int(entries["Y2"].get() or 22)
            sel = algo_var.get()

            if "DDA" in sel:
                pts = dda_line(x1, y1, x2, y2)
                for p in pts:
                    plot(p[0], p[1], "#38bdf8")
                log_lbl.config(text=f"DDA Points ({len(pts)}):\n" + "\n".join(str(p) for p in pts[:12]))
            elif "Bresenham" in sel:
                pts = bresenham_line(x1, y1, x2, y2)
                for p in pts:
                    plot(p[0], p[1], "#34d399")
                log_lbl.config(text=f"Bresenham Points ({len(pts)}):\n" + "\n".join(str(p) for p in pts[:12]))
            elif "Midpoint" in sel:
                cx, cy, r = max(5, min(24, x1)), max(5, min(24, y1)), max(2, min(12, x2))
                pts = midpoint_circle(cx, cy, r)
                for p in pts:
                    plot(p[0], p[1], "#fbbf24")
                log_lbl.config(text=f"Midpoint Circle C=({cx},{cy}), R={r}\nPlotted {len(pts)} octant pixels.")
            else:
                xmin, ymin, xmax, ymax = 7, 7, 22, 22
                cv.create_rectangle(xmin * cell, (N - 1 - ymax) * cell, (xmax + 1) * cell, (N - ymin) * cell, outline="#a855f7", width=2)
                cv.create_line((x1 + 0.5) * cell, (N - 0.5 - y1) * cell, (x2 + 0.5) * cell, (N - 0.5 - y2) * cell, fill="#ef4444", dash=(4, 3))
                res = cohen_sutherland_clip(x1, y1, x2, y2, xmin, ymin, xmax, ymax)
                if res:
                    ax, ay, bx, by = res
                    cv.create_line((ax + 0.5) * cell, (N - 0.5 - ay) * cell, (bx + 0.5) * cell, (N - 0.5 - by) * cell, fill="#34d399", width=3)
                    log_lbl.config(text=f"Clipped Window [7,7]-[22,22]\nACCEPTED:\n({ax:.1f},{ay:.1f}) -> ({bx:.1f},{by:.1f})")
                else:
                    log_lbl.config(text="Clipped Window [7,7]-[22,22]\nTRIVIALLY REJECTED")

        tk.Button(ctrl, text="Run & Rasterize", command=run_algo, bg="#2563eb", fg="white", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, pady=5).pack(fill=tk.X, pady=4)
        run_algo()


if __name__ == "__main__":
    root = tk.Tk()
    app = SolarSystemPythonApp(root)
    root.mainloop()
