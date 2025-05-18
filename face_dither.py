import sys
import site

# Add user site-packages for scipy
site.USER_SITE = '/Users/yitong/.local/lib/python3.11/site-packages'
sys.path.append(site.USER_SITE)

import bpy
import bmesh
import random
import numpy as np
from mathutils import Vector
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import pdist

# Set random seed for reproducibility
random.seed(5)
np.random.seed(5)

# === PARAMETERS ===
# DITHER_MAGNITUDE is now a range
MIN_DITHER = 0.01
MAX_DITHER = 0.2
EXEMPT_KEYWORDS = ["peripheral", "inside", 'eyelid_tops']
SYMMETRY_THRESHOLD = 0.01  # threshold for considering vertices symmetric

# === HELPER FUNCTIONS ===
def get_vertex_world_position(obj, index):
    return obj.matrix_world @ obj.data.vertices[index].co

def reflect_across_plane(P, P0, N):
    # Reflect point P across the plane defined by point P0 and normal N
    return P - 2 * ((P - P0).dot(N)) * N

def random_offset(min_magnitude=MIN_DITHER, max_magnitude=MAX_DITHER):
    # Generate a normalized random direction vector
    direction = Vector([random.uniform(-1, 1) for _ in range(3)]).normalized()
    # Choose a random magnitude within the specified range
    magnitude = random.uniform(min_magnitude, max_magnitude)
    return direction * magnitude

def add_sphere(location, name, color=(1, 0, 0, 1), radius=0.01):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location)
    sphere = bpy.context.object
    sphere.name = name
    mat_name = f"Mat_{color}"
    mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)
    mat.diffuse_color = color
    if not sphere.data.materials:
        sphere.data.materials.append(mat)
    else:
        sphere.data.materials[0] = mat
    return sphere

def signed_distance_to_plane(point, plane_origin, plane_normal):
    # Calculate signed distance from point to plane
    return (point - plane_origin).dot(plane_normal)

def is_symmetric_pair(pos1, pos2, plane_origin, plane_normal, threshold=SYMMETRY_THRESHOLD):
    # Test if two points are symmetric with respect to the plane
    reflected = reflect_across_plane(pos1, plane_origin, plane_normal)
    return (reflected - pos2).length < threshold

# === SETUP ===
# Ensure we're in Object Mode
bpy.ops.object.mode_set(mode='OBJECT')

# Get the face object
obj = bpy.data.objects["Yitong_Face"]
mesh = obj.data
verts = mesh.vertices

# === STEP 1: Get Plane of Symmetry ===
def get_named_vertex(name):
    group = obj.vertex_groups.get(name)
    if not group:
        raise ValueError(f"Vertex group '{name}' not found.")
    for v in verts:
        if any(g.group == group.index for g in v.groups):
            return get_vertex_world_position(obj, v.index)
    raise ValueError(f"No vertex assigned to group '{name}'.")

# Get the three points defining the symmetry plane
A = get_named_vertex("Nose_Smack_Dap_Middle")
B = get_named_vertex("Below_Lips")
C = get_named_vertex("Third_Eye")

# Calculate plane origin and normal
plane_origin = A
plane_normal = (B - A).cross(C - A).normalized()

print(f"Symmetry plane established at {plane_origin} with normal {plane_normal}")

# === STEP 2: Collect and Process Vertices ===
original_positions = []
dithered_positions = []
processed_vertices = set()  # Keep track of processed vertices

for vgroup in obj.vertex_groups:
    name = vgroup.name
    verts_in_group = [v.index for v in verts if any(g.group == vgroup.index for g in v.groups)]
    
    if not verts_in_group:
        continue
        
    # Check if this group should be exempt from dithering
    is_exempt = any(key.lower() in name.lower() for key in EXEMPT_KEYWORDS)
    
    print(f"Processing group '{name}' with {len(verts_in_group)} vertices. Exempt: {is_exempt}")
    
    if is_exempt:
        # For exempt groups, just add original positions (no dithering)
        for vert_idx in verts_in_group:
            if vert_idx in processed_vertices:
                continue
                
            pos = get_vertex_world_position(obj, vert_idx)
            original_positions.append(np.array(pos))
            dithered_positions.append(np.array(pos))  # No change
            
            add_sphere(pos, f"Anchor_{name}_{vert_idx}", (1, 0, 0, 1))
            processed_vertices.add(vert_idx)
    else:
        # Non-exempt group: Apply dithering based on vertex group size
        
        # Only process if the vertices haven't been processed yet
        unprocessed_verts = [v for v in verts_in_group if v not in processed_vertices]
        
        if len(unprocessed_verts) == 1:
            # Single vertex case: dither parallel to the plane
            vert_idx = unprocessed_verts[0]
            pos = get_vertex_world_position(obj, vert_idx)
            
            # Generate random offset
            raw_offset = random_offset()
            # Project offset onto the plane (make it parallel to the plane)
            plane_offset = raw_offset - raw_offset.dot(plane_normal) * plane_normal
            dithered = pos + plane_offset
            
            original_positions.append(np.array(pos))
            dithered_positions.append(np.array(dithered))
            
            add_sphere(pos, f"Anchor_{name}_{vert_idx}", (1, 0, 0, 1))
            add_sphere(dithered, f"Dither_{name}_{vert_idx}", (0, 1, 0, 1))
            processed_vertices.add(vert_idx)
            
        elif len(unprocessed_verts) == 2:
            # Two vertex case: dither first vertex, mirror for second
            v1_idx, v2_idx = unprocessed_verts
            v1_pos = get_vertex_world_position(obj, v1_idx)
            v2_pos = get_vertex_world_position(obj, v2_idx)
            
            # Apply random offset to first vertex
            offset = random_offset()
            v1_dithered = v1_pos + offset
            
            # Calculate mirror transform for second vertex
            mirrored_offset = reflect_across_plane(offset, Vector((0,0,0)), plane_normal)
            v2_dithered = v2_pos + mirrored_offset
            
            # Store positions for both vertices
            original_positions.append(np.array(v1_pos))
            original_positions.append(np.array(v2_pos))
            dithered_positions.append(np.array(v1_dithered))
            dithered_positions.append(np.array(v2_dithered))
            
            # Visualize with spheres
            add_sphere(v1_pos, f"Anchor_{name}_{v1_idx}", (1, 0, 0, 1))
            add_sphere(v2_pos, f"Anchor_{name}_{v2_idx}", (1, 0, 0, 1))
            add_sphere(v1_dithered, f"Dither_{name}_{v1_idx}", (0, 1, 0, 1))
            add_sphere(v2_dithered, f"Dither_{name}_{v2_idx}", (0, 1, 0, 1))
            
            processed_vertices.add(v1_idx)
            processed_vertices.add(v2_idx)
            
        elif len(unprocessed_verts) > 2:
            print(f"Warning: Vertex group '{name}' has {len(unprocessed_verts)} vertices. Only expected 1 or 2.")
            # Skip this group or handle as needed

print(f"✅ Collected {len(original_positions)} anchor pairs")

# === STEP 3: Apply RBF DEFORMATION ===
original_positions = np.array(original_positions)
dithered_positions = np.array(dithered_positions)
displacements = dithered_positions - original_positions

# Safety check to avoid singularities
if np.min(pdist(original_positions)) < 1e-6:
    print("⚠️ Warning: Some original anchor points are too close or identical.")
    # Add small jitter to positions
    jitter = np.random.normal(0, 1e-7, original_positions.shape)
    original_positions = original_positions + jitter

# Create RBF interpolator with some smoothing to help with numerical stability
rbf = RBFInterpolator(
    original_positions,
    displacements,
    kernel='thin_plate_spline',
    degree=0,
    smoothing=1e-4
)

# Apply deformation to all vertices
for v in mesh.vertices:
    pos = obj.matrix_world @ v.co
    displacement = rbf([[pos.x, pos.y, pos.z]])[0]
    v.co.x += displacement[0]
    v.co.y += displacement[1]
    v.co.z += displacement[2]

print("✅ RBF deformation applied.")
