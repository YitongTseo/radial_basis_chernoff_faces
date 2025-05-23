# To see print statements:
# Open terminal then
# /Applications/Blender.app/Contents/MacOS/Blender

import bpy
import numpy as np
import mathutils
import math
from collections import deque
import bmesh # For creating debug puddle mesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree

def is_point_inside_mesh(point, bvh, max_tests=12):
    # Cast rays in several directions to reduce false negatives at surface edges
    directions = [Vector((1,0,0)), Vector((0,1,0)), Vector((0,0,1)),
                  Vector((-1,0,0)), Vector((0,-1,0)), Vector((0,0,-1))]
    hits = 0
    for dir in directions:
        location, normal, index, distance = bvh.ray_cast(point, dir, 1e6)
        if location is not None:
            hits += 1
    return hits % 2 == 1


def get_enclosed_puddle_volume(target_obj_name: str, octree_depth: float, create_debug_objects: bool = False, debug: bool = True) -> float:
    """
    Estimates water volume in puddles that are enclosed by solid walls on all
    horizontal sides and have a solid floor or are supported by other enclosed water below.
    Solid voxels are determined by a flood fill from outside a remeshed shell.


    Args:
        target_obj_name (str): The name of the Blender mesh object.
        voxel_size (float): The side length of each cubic voxel.
        create_debug_objects (bool): If True, creates new mesh objects in the scene
                                     for the voxelized solid, trapped water, and bbox markers.
    Returns:
        float: The estimated total trapped_water volume in Blender units^3.
    """
    target_obj = bpy.data.objects.get(target_obj_name)
    if not target_obj or target_obj.type != 'MESH':
        print(f"Error: Object '{target_obj_name}' not found or not a mesh.")
        return 0.0

    # --- 1. Preparation: Duplicate and Remesh ---
    original_active = bpy.context.view_layer.objects.active
    original_selected = bpy.context.selected_objects[:]

    bpy.context.view_layer.objects.active = target_obj
    for ob in bpy.data.objects: ob.select_set(False)
    target_obj.select_set(True)
    
    # I got tired of the duplicate object, which seems to serve no use here...
#    bpy.ops.object.duplicate()
#    remesh_obj = bpy.context.active_object
#    remesh_obj.name = target_obj_name + "_RemeshVoxelTemp"
    remesh_obj = bpy.context.active_object
    
    # Settings
    voxel_size = 0.2  # in meters, 1 cm
    name = "Voxelized_Solid"
    mesh_collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(mesh_collection)
    
    

    # Apply transformations so the bounding box is correct
    bpy.context.view_layer.objects.active = remesh_obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Step 1: Get bounding box in world coordinates
    bbox_world = [remesh_obj.matrix_world @ Vector(corner) for corner in remesh_obj.bound_box]
    bbox_min = Vector((min(v[i] for v in bbox_world) for i in range(3)))
    bbox_max = Vector((max(v[i] for v in bbox_world) for i in range(3)))
    print(f"Bounding box: min={bbox_min}, max={bbox_max}")

    # Step 2: Define voxel grid
#    import pdb; pdb.set_trace()
    float_vec = (bbox_max - bbox_min) / voxel_size
    dims = tuple(int(x) for x in float_vec)
#    dims = ((bbox_max - bbox_min) / voxel_size).to_tuple(int)
    nx, ny, nz = dims
    print(f"Voxel grid: nx={nx}, ny={ny}, nz={nz}")

    # Prepare solid array
    is_solid = np.zeros((nx, ny, nz), dtype=bool)

    bm = bmesh.new()
    bm.from_mesh(remesh_obj.data)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bvh = BVHTree.FromBMesh(bm)
    
    # TODO: turn back on debug for later...
    if debug and False:
        bpy.ops.mesh.primitive_cube_add(size=voxel_size, location=(0, 0, 0))
        voxel_prototype = bpy.context.object
        voxel_prototype.name = "Voxel_Prototype"
#        voxel_prototype.hide_render = True
#        voxel_prototype.hide_viewport = True


    # Step 4: Fill the grid
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                point = bbox_min + Vector((ix + 0.5, iy + 0.5, iz + 0.5)) * voxel_size
                
                if bvh.find_nearest(point):  # optional culling
                    if not is_point_inside_mesh(point, bvh):
                        is_solid[ix, iy, iz] = True
                        # Turning of debug right now 
                        if debug and False:
                            voxel_center = bbox_min + Vector((ix + 0.5, iy + 0.5, iz + 0.5)) * voxel_size
                            voxel = voxel_prototype.copy()
                            voxel.data = voxel_prototype.data.copy()
                            voxel.location = voxel_center
                            voxel.hide_set(False)
                            mesh_collection.objects.link(voxel)

    
    # TRYING SOMETHING now!
    bfs_deltas_8way = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    final_trapped_water = np.zeros((nx, ny, nz), dtype=bool)
    
    for iz_slice in range(1, nz):  # Start at 1 because we check the layer below
        current_slice_solid = is_solid[:, :, iz_slice]
        below_slice = is_solid[:, :, iz_slice - 1] | final_trapped_water[:, :, iz_slice - 1]
        # Water candidates must have a "floor" and be empty
        slice_water_with_floor = ~current_slice_solid & below_slice
        
        print(f'for slice {iz_slice} we have {slice_water_with_floor.sum()} candidates')
        if not np.any(slice_water_with_floor):
            continue  # Nothing to process in this slice
        
        # TODO: REMOVE
#        final_trapped_water[:, :, iz_slice] = slice_water_with_floor
        
        # Track which water candidates have been visited
        visited = np.zeros((nx, ny), dtype=bool)
        slice_trapped_this_level = np.zeros((nx, ny), dtype=bool)
        
        # Find all connected water sections using 8-way connectivity
        for ix in range(nx):
            for iy in range(ny):
                if not slice_water_with_floor[ix, iy] or visited[ix, iy]:
                    continue
                
                # Found a new water section - use BFS to find all connected water voxels
                water_section = []
                q = deque([(ix, iy)])
                visited[ix, iy] = True
                
                while q:
                    cx, cy = q.popleft()
                    water_section.append((cx, cy))
                    
                    # Check all 8 neighbors for more water voxels
                    for dx, dy in bfs_deltas_8way:
                        nx_, ny_ = cx + dx, cy + dy
                        
                        # Skip if out of bounds
                        if not (0 <= nx_ < nx and 0 <= ny_ < ny):
                            continue
                        
                        # If neighbor is also a water candidate and not visited, add to section
                        if not visited[nx_, ny_] and slice_water_with_floor[nx_, ny_]:
                            visited[nx_, ny_] = True
                            q.append((nx_, ny_))
                
                # Now check if this water section is fully enclosed by solid walls
                is_section_trapped = True
                
                for wx, wy in water_section:
                    # Check if this water voxel has any non-solid neighbors
                    for dx, dy in bfs_deltas_8way:
                        nx_, ny_ = wx + dx, wy + dy
                        
                        # If we're at the boundary, water can escape
                        if not (0 <= nx_ < nx and 0 <= ny_ < ny):
                            is_section_trapped = False
                            break
                        
                        # If neighbor is not solid and not part of this water section, water can escape
                        if not current_slice_solid[nx_, ny_] and not slice_water_with_floor[nx_, ny_]:
                            is_section_trapped = False
                            break
                    
                    if not is_section_trapped:
                        break
                
                # If the entire section is trapped, mark all voxels in it as trapped water
                if is_section_trapped:
                    for wx, wy in water_section:
                        slice_trapped_this_level[wx, wy] = True
        
        final_trapped_water[:, :, iz_slice] = slice_trapped_this_level
#    

    if debug:
        bpy.ops.mesh.primitive_cube_add(size=voxel_size, location=(0, 0, 0))
        voxel_prototype = bpy.context.object
        voxel_prototype.name = "WATER_VOXELS"
        
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    if final_trapped_water[ix, iy, iz]:
                        print('we got in here?', ix, iy, 'THIS ONE IS Z', iz)
                        voxel_center = bbox_min + Vector((ix + 0.5, iy + 0.5, iz + 0.5)) * voxel_size
                        voxel = voxel_prototype.copy()
                        voxel.data = voxel_prototype.data.copy()
                        voxel.location = voxel_center
                        voxel.hide_set(False)
                        mesh_collection.objects.link(voxel)
                        
#    import pdb; pdb.set_trace()
    
    print('hi ya')


# --- Example Usage ---
if __name__ == "__main__":
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    object_name_to_test = "Cube" # Change to your object, e.g., "Suzanne" or a custom model
    original_obj = bpy.data.objects.get(object_name_to_test)

    if original_obj:
        bpy.context.view_layer.objects.active = original_obj
        for obj_iter in bpy.data.objects: obj_iter.select_set(False)
        original_obj.select_set(True)
        
        # It's highly recommended to apply scale to your object before running this
        # bpy.ops.object.transform_apply(location=False, rotation=False, scale=True) 

        v_size = 0.1 
        if original_obj.dimensions.length > 0:
            avg_dim = sum(original_obj.dimensions) / 3.0 
            auto_voxel_size = avg_dim / 30 # Coarser: 20-30, Finer: 50-100. Adjust for performance/detail.
            if auto_voxel_size > 0.0001: v_size = auto_voxel_size
            print(f"Auto-calculated voxel size: {v_size:.4f}")
        else:
            print(f"Object has zero dimensions. Using default voxel size: {v_size}")
        
        print(f"\n--- Calculating ENCLOSED Puddle Volume for: {original_obj.name} with Voxel Size: {v_size:.4f} ---")
        estimated_volume = get_enclosed_puddle_volume(original_obj.name, 7, create_debug_objects=True)
        #print(f"--- Finished. Final Estimated ENCLOSED Volume: {estimated_volume:.6f} ---")
    else:
        print(f"Object '{object_name_to_test}' not found.")
