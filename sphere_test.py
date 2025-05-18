import sys
import site

# Add user site-packages for scipy
site.USER_SITE = '/Users/yitong/.local/lib/python3.11/site-packages'
sys.path.append(site.USER_SITE)

import bpy
import numpy as np
import random
from scipy.interpolate import RBFInterpolator

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# === CLEANUP ===
for name in ["TestSphere", "TestPlane"]:
    if name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
for obj in bpy.data.objects:
    if obj.name.startswith("Anchor_") or obj.name.startswith("Dither_"):
        bpy.data.objects.remove(obj, do_unlink=True)

# === CREATE SPHERE ===
bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=1.5)
obj = bpy.context.object
obj.name = "TestSphere"
mesh = obj.data

# === GET VERTICES ===
verts = np.array([v.co.to_tuple() for v in mesh.vertices])
num_anchors = 20
anchor_indices = random.sample(range(len(verts)), num_anchors)
original_positions = verts[anchor_indices]

# === APPLY DITHER ===
dither_strength = 0.2
dithered_positions = original_positions + dither_strength * np.random.randn(num_anchors, 3)

# === VISUALIZE POINTS ===
def add_sphere(location, name, color):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.05, location=location)
    sphere = bpy.context.object
    sphere.name = name

    mat = bpy.data.materials.new(name + "_Mat")
    mat.diffuse_color = (*color, 1.0)
    sphere.data.materials.append(mat)

# RED = original anchors, GREEN = dithered
for i, (orig, dith) in enumerate(zip(original_positions, dithered_positions)):
    add_sphere(orig, f"Anchor_{i}", (1.0, 0.1, 0.1))     # red
    add_sphere(dith, f"Dither_{i}", (0.1, 1.0, 0.1))     # green

print("✅ Original and dithered anchor points visualized.")

# === APPLY RBF DEFORMATION ===
displacements = dithered_positions - original_positions
print("Sample displacement vector:", displacements[0])

# Use degree=0 to avoid linear degeneracy
rbf = RBFInterpolator(original_positions, displacements, kernel='thin_plate_spline', degree=0)

predicted_displacements = rbf(verts)

# Apply deformation to the sphere
for i, v in enumerate(mesh.vertices):
    dx, dy, dz = predicted_displacements[i]
    v.co.x += dx
    v.co.y += dy
    v.co.z += dz

print("✅ Deformation applied to TestSphere.")
