"""

## Running the Script

Basic usage:

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background --python blender_rbf_script.py -- --input face_landmark_points.blend --dither_config face_individuals/example_params.json --output output.blend

```

### Command Line Arguments

- `--input`: Path to input .blend file containing the "Yitong_Face" object with vertex groups (required)
- `--dither_config`: Path to JSON configuration file specifying dithering parameters (required)
- `--output`: Path to save the modified .blend file (default: output.blend)
- `--exempt_keywords`: Comma-separated list of keywords for vertex groups to exempt from dithering (default: peripheral,inside,eyelid_tops)
- `--seed`: Random seed for reproducibility (optional)

## Configuration File Format

The configuration file should be a JSON file specifying dithering parameters for each vertex group:

```json
{
  "default": {
    "direction": [0, 0, 0],
    "magnitude": 0.0
  },
  "Left_Eyebrow": {
    "direction": [0, 0, 1],
    "magnitude": 0.15,
    "parallel_to_plane": false
  },
  "Right_Eyebrow": {
    "direction": [0, 0, 1],
    "magnitude": 0.15,
    "parallel_to_plane": false
  }
}


## Headless Blender RBF Deformation Script - Usage Guide

This script allows you to apply RBF (Radial Basis Function) deformation to a Blender model in headless mode, with precise control over dithering direction and magnitude for different vertex groups.

## Prerequisites

1. Blender installation (tested with Blender 3.x)
2. Python 3.x with the following packages:
   - numpy
   - scipy

## Installation

1. Make sure the scipy package is available to Blender's Python interpreter:
   ```bash
   /Applications/Blender.app/Contents/MacOS/Blender --background --python-expr "import sys; print(sys.executable)" | xargs -I{} {} -m pip install scipy
   ```
   
   Alternatively, you can set the `site.USER_SITE` environment variable to point to a directory with scipy installed.

2. Save the script as `blender_rbf_script.py`

3. Create a JSON configuration file for dithering parameters (see example below)

```

### Configuration Parameters

- `direction`: A 3D vector [x, y, z] indicating the direction of the dithering
- `magnitude`: The amount of displacement to apply in the specified direction
- `parallel_to_plane`: (Optional) If true, the direction will be projected onto the symmetry plane

You can specify a `default` entry for vertex groups not explicitly included in the configuration.

## Example

To apply specific dithering to eyebrows and nose:

1. Create a configuration file `dither_config.json`:
   ```json
   {
     "default": {
       "direction": [0, 0, 0],
       "magnitude": 0.0
     },
     "Left_Eyebrow": {
       "direction": [0, 0, 1],
       "magnitude": 0.15
     },
     "Right_Eyebrow": {
       "direction": [0, 0, 1],
       "magnitude": 0.15
     },
     "Nose_Tip": {
       "direction": [0, 1, 0],
       "magnitude": 0.2
     }
   }
   ```

2. Run the script:
   ```bash
   /Applications/Blender.app/Contents/MacOS/Blender --background --python blender_rbf_script.py -- --input face_model.blend --dither_config dither_config.json
   ```

## Notes

- The script requires three special vertex groups to define the symmetry plane: "Nose_Smack_Dap_Middle", "Below_Lips", and "Third_Eye"
- For symmetric features, pairs of vertices will be dithered with mirrored offsets
- Groups with keywords listed in `--exempt_keywords` will not be dithered
- Deformation is applied using Radial Basis Function interpolation, which smoothly distributes the effect across the mesh
"""


#!/usr/bin/env python3
import sys
import site

# Add user site-packages for scipy
site.USER_SITE = '/Users/yitong/.local/lib/python3.11/site-packages'
sys.path.append(site.USER_SITE)
import bpy
import bmesh
import sys
import site
import os
import json
import argparse
import numpy as np
import random
from mathutils import Vector
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import pdist
import struct

# === ARGUMENT PARSING ===
def get_args():
    # Define a custom argument parser that ignores blender's args
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]  # Get all args after "--"
    else:
        argv = []  # No custom args found
    
    parser = argparse.ArgumentParser(description='Headless Blender RBF Deformation Script')
    
    # Required arguments
    parser.add_argument('--input', type=str, required=True, help='Path to input .blend file')
    
    # Configure dithering parameters via JSON
    parser.add_argument('--dither_config', type=str, required=True, 
                        help='JSON configuration with dithering parameters for vertex groups')
    
    # Optional parameters
    parser.add_argument('--output', type=str, default='output.blend', 
                        help='Path to output the modified .blend file')
    parser.add_argument('--exempt_keywords', type=str, default='sparse_peripherals', 
                        help='Comma-separated list of keywords for vertex groups to exempt from dithering')
    parser.add_argument('--symmetry_threshold', type=float, default=0.01,
                        help='Threshold for considering vertices symmetric')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args(argv)
    return args

# === HELPER FUNCTIONS ===

def enable_addon(addon_name):
    """Enable an addon if it isn't already enabled."""
    try:
        # Try to enable the addon
        bpy.ops.preferences.addon_enable(module=addon_name)
        print(f"Enabled addon: {addon_name}")
        return True
    except Exception as e:
        print(f"Error enabling addon {addon_name}: {e}")
        return False

def get_vertex_world_position(obj, index):
    return obj.matrix_world @ obj.data.vertices[index].co

def reflect_across_plane(P, P0, N):
    # Reflect point P across the plane defined by point P0 and normal N
    return P - 2 * ((P - P0).dot(N)) * N

def apply_dither_offset(pos, direction, magnitude):
    # Apply dither in specified direction with specified magnitude
    if isinstance(direction, list):
        direction_vector = Vector(direction).normalized()
    else:  # Assume it's already a Vector
        direction_vector = direction.normalized()
    
    return pos + direction_vector * magnitude

def signed_distance_to_plane(point, plane_origin, plane_normal):
    # Calculate signed distance from point to plane
    return (point - plane_origin).dot(plane_normal)

def is_symmetric_pair(pos1, pos2, plane_origin, plane_normal, threshold):
    # Test if two points are symmetric with respect to the plane
    reflected = reflect_across_plane(pos1, plane_origin, plane_normal)
    return (reflected - pos2).length < threshold

# === MAIN FUNCTION ===
def main():
    # Get command line arguments
    args = get_args()
    
    # Set random seed if specified
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Load the .blend file
    bpy.ops.wm.open_mainfile(filepath=args.input)
    
    # Load dithering configuration from JSON
    with open(args.dither_config, 'r') as f:
        dither_config = json.load(f)
    
    # if not export_stl_path and "_config" in dither_config and dither_config["_config"].get("export_stl", False):
    # If enabled in config but no path given, use the output path with .stl extension
    export_stl_path = os.path.splitext(args.output)[0] + ".stl"
    
    # Parse exempt keywords
    EXEMPT_KEYWORDS = [keyword.strip() for keyword in args.exempt_keywords.split(',')]
    SYMMETRY_THRESHOLD = args.symmetry_threshold
    
    print(f"Loaded configuration. Starting processing.")
    
    # Ensure we're in Object Mode
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Get the face object - make sure it exists first
    face_obj = None
    for obj in bpy.data.objects:
        if "eyes_open_mask" in obj.name:
            face_obj = obj
            print(f"Found face object: {obj.name}")
            break
    
    if not face_obj:
        print("Error: Could not find 'Yitong_Face' object. Available objects:")
        for obj in bpy.data.objects:
            print(f"  - {obj.name}")
        raise ValueError("Face object not found")
    
    # Select the face object and make it active
    bpy.ops.object.select_all(action='DESELECT')
    face_obj.select_set(True)
    bpy.context.view_layer.objects.active = face_obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    
    obj = face_obj
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
    A = get_named_vertex("Middle_Head")
    B = get_named_vertex("Third_Eye")
    C = get_named_vertex("Chin")
    
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
        
        # Get dithering parameters for this vertex group from config, or use defaults
        group_config = dither_config.get(name, dither_config.get('default', {}))
        
        # If direction is not specified, use [0,0,0] (no dithering)
        direction = group_config.get('direction', [0, 0, 0])
        magnitude = group_config.get('magnitude', 0.0)
        
        # Handle exempt groups
        if is_exempt:
            magnitude = 0.0  # Force no dithering for exempt groups
        
        print(f"Processing group '{name}' with {len(verts_in_group)} vertices. " + 
              f"Direction: {direction}, Magnitude: {magnitude}, Exempt: {is_exempt}")
        
        # If magnitude is 0 or group is exempt, don't dither
        if magnitude == 0 or is_exempt:
            for vert_idx in verts_in_group:
                if vert_idx in processed_vertices:
                    continue
                    
                pos = get_vertex_world_position(obj, vert_idx)
                original_positions.append(np.array(pos))
                dithered_positions.append(np.array(pos))  # No change
                processed_vertices.add(vert_idx)
        else:
            # Non-exempt group with dithering
            unprocessed_verts = [v for v in verts_in_group if v not in processed_vertices]
            
            if len(unprocessed_verts) == 1:
                # Single vertex case
                vert_idx = unprocessed_verts[0]
                pos = get_vertex_world_position(obj, vert_idx)
                
                # Apply dithering in specified direction
                dir_vector = Vector(direction)
                # Project direction onto the plane if requested
                if group_config.get('parallel_to_plane', True):
                    dir_vector = dir_vector - dir_vector.dot(plane_normal) * plane_normal
                
                dithered = apply_dither_offset(pos, dir_vector, magnitude)
                
                original_positions.append(np.array(pos))
                dithered_positions.append(np.array(dithered))
                processed_vertices.add(vert_idx)
            elif len(unprocessed_verts) == 2:
                # Two vertex case: dither first vertex, mirror for second
                v1_idx, v2_idx = unprocessed_verts
                v1_pos = get_vertex_world_position(obj, v1_idx)
                v2_pos = get_vertex_world_position(obj, v2_idx)
                
                # Apply dithering offset to first vertex
                dir_vector = Vector(direction)
                offset = dir_vector.normalized() * magnitude
                v1_dithered = v1_pos + offset
                
                # Calculate mirror transform for second vertex
                mirrored_offset = reflect_across_plane(offset, Vector((0,0,0)), plane_normal)
                v2_dithered = v2_pos + mirrored_offset
                
                # Store positions for both vertices
                original_positions.append(np.array(v1_pos))
                original_positions.append(np.array(v2_pos))
                dithered_positions.append(np.array(v1_dithered))
                dithered_positions.append(np.array(v2_dithered))
                
                processed_vertices.add(v1_idx)
                processed_vertices.add(v2_idx)
                
            elif len(unprocessed_verts) > 2:
                print(f"Warning: Vertex group '{name}' has {len(unprocessed_verts)} vertices. Only expected 1 or 2.")
                # Skip this group or handle as needed
    
    print(f"✅ Collected {len(original_positions)} anchor pairs")
    
    # === STEP 3: Apply RBF DEFORMATION ===
    original_positions = np.array(original_positions)
    dithered_positions = np.array(dithered_positions)
    
    # Check if deformation should be applied
    apply_deformation = True
    if "_config" in dither_config:
        apply_deformation = dither_config["_config"].get("apply_deformation", True)
    
    # Skip deformation if no points were dithered or deformation is disabled
    if len(original_positions) == 0:
        print("⚠️ No vertices to process. No deformation applied.")
    elif not apply_deformation:
        print("⚠️ Deformation disabled in configuration. No deformation applied.")
    else:
        displacements = dithered_positions - original_positions
        
        # Safety check to avoid singularities
        if len(original_positions) > 1 and np.min(pdist(original_positions)) < 1e-6:
            print("⚠️ Warning: Some original anchor points are too close or identical.")
            # Add small jitter to positions
            jitter = np.random.normal(0, 1e-7, original_positions.shape)
            original_positions = original_positions + jitter
        
        # Create RBF interpolator with some smoothing to help with numerical stability
        try:
            rbf = RBFInterpolator(
                original_positions,
                displacements,
                kernel='thin_plate_spline',
                degree=0,
                smoothing=1e-4
            )
            
            # Apply deformation to all vertices in object local space
            for v in mesh.vertices:
                world_pos = obj.matrix_world @ v.co
                world_pos_array = np.array([[world_pos.x, world_pos.y, world_pos.z]])
                displacement = rbf(world_pos_array)[0]
                
                # Update the vertex position (in object space)
                world_new_pos = Vector((
                    world_pos.x + displacement[0],
                    world_pos.y + displacement[1],
                    world_pos.z + displacement[2]
                ))
                
                # Convert back to local space
                v.co = obj.matrix_world.inverted() @ world_new_pos
            
            # Update the mesh
            mesh.update()
            print("✅ RBF deformation applied.")
        except Exception as e:
            print(f"❌ Error applying RBF deformation: {e}")
    
    # === STEP 5: Save the Blender file ===
    try:
        # Apply all modifiers first if there are any
        if obj.modifiers:
            for modifier in obj.modifiers:
                bpy.ops.object.modifier_apply(modifier=modifier.name)
            print("✅ Applied all modifiers")
        
        # Save the blender file
        bpy.ops.wm.save_as_mainfile(filepath=args.output)
        print(f"✅ Saved modified blend file to {args.output}")
    except Exception as e:
        print(f"❌ Error saving file: {e}")


    # # === STEP 4: Export STL if requested ===
    # print(f"Exporting STL to {export_stl_path}")
    # # Make sure the face object is selected and active
    # bpy.ops.object.select_all(action='DESELECT')
    # obj.select_set(True)
    # bpy.context.view_layer.objects.active = obj
    # enable_addon("io_mesh_stl")
    # # Export as STL
    # try:
    #     # First, ensure the STL export addon is enabled
    #     bpy.ops.preferences.addon_enable(module="io_mesh_stl")
        
    #     # Try the export operation - Blender 2.8+ version
    #     try:
    #         bpy.ops.export_mesh.stl(
    #             filepath=export_stl_path,
    #             use_selection=True,
    #             global_scale=1.0,
    #             use_scene_unit=True,
    #             ascii=False,
    #             use_mesh_modifiers=True
    #         )
    #         print(f"✅ STL exported to {export_stl_path}")
    #     except AttributeError:
    #         # Fallback for older Blender versions
    #         bpy.ops.export_mesh.stl(
    #             filepath=export_stl_path,
    #             check_existing=True,
    #             filter_glob="*.stl",
    #             use_selection=True,
    #             global_scale=1.0,
    #             use_scene_unit=True,
    #             ascii=False
    #         )
    #         print(f"✅ STL exported to {export_stl_path} (using legacy exporter)")
    # except Exception as e:
    #     print(f"❌ Failed to export STL: {e}")
    #     print("Attempting alternative export method...")
        
    #     # Try a direct mesh export if the operator fails
    #     try:
    #         import bmesh
    #         from mathutils import Matrix
            
    #         # Create a bmesh from the object
    #         bm = bmesh.new()
    #         bm.from_mesh(obj.data)
            
    #         # Apply object transformation to vertices
    #         for v in bm.verts:
    #             v.co = obj.matrix_world @ v.co
            
    #         # Write STL file directly
    #         with open(export_stl_path, 'wb') as fp:
    #             # Write STL header (80 bytes)
    #             fp.write(b'Binary STL file generated by Blender script' + b'\0' * 48)
                
    #             # Write number of triangles (4 bytes)
    #             num_faces = len(bm.faces)
    #             fp.write(num_faces.to_bytes(4, byteorder='little'))
                
    #             # Write each triangle
    #             for f in bm.faces:
    #                 # Write normal (12 bytes)
    #                 normal = f.normal
    #                 fp.write(struct.pack('<fff', normal.x, normal.y, normal.z))
                    
    #                 # Write vertices (36 bytes)
    #                 for v in f.verts:
    #                     fp.write(struct.pack('<fff', v.co.x, v.co.y, v.co.z))
                    
    #                 # Write attribute byte count (2 bytes)
    #                 fp.write(b'\0\0')
            
    #         bm.free()
    #         print(f"✅ STL exported to {export_stl_path} (using direct method)")
    #     except Exception as manual_err:
    #         print(f"❌ All STL export methods failed: {manual_err}")

if __name__ == "__main__":
    # Add user site-packages for scipy
    if "site.USER_SITE" in os.environ:
        site.USER_SITE = os.environ["site.USER_SITE"]
    else:
        site.USER_SITE = '/Users/yitong/.local/lib/python3.11/site-packages'
    sys.path.append(site.USER_SITE)
    
    main()