# To see print statements:
# Open terminal then
# /Applications/Blender.app/Contents/MacOS/Blender

import bpy
import numpy as np
import mathutils
import math
from collections import deque
import bmesh # For creating debug puddle mesh

def get_enclosed_puddle_volume(target_obj_name: str, voxel_size: float, create_debug_objects: bool = False) -> float:
    """
    Estimates water volume in puddles that are enclosed by solid walls on all
    horizontal sides and have a solid floor or are supported by other enclosed water below.

    Args:
        target_obj_name (str): The name of the Blender mesh object.
        voxel_size (float): The side length of each cubic voxel.
        create_debug_objects (bool): If True, creates new mesh objects in the scene
                                     for the voxelized solid and the trapped water voxels.
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

    bpy.ops.object.duplicate()
    remesh_obj = bpy.context.active_object
    remesh_obj.name = target_obj_name + "_RemeshVoxelTemp"

    print(f"Remeshing '{remesh_obj.name}' with voxel size: {voxel_size}...")
    try:
        mod = remesh_obj.modifiers.new(name="VoxelRemesh", type='REMESH')
        mod.mode = 'VOXEL'
        mod.voxel_size = voxel_size
        mod.use_remove_disconnected = False
        mod.adaptivity = 0
        bpy.context.view_layer.objects.active = remesh_obj
        remesh_obj.select_set(True)
        bpy.ops.object.modifier_apply(modifier=mod.name)
    except RuntimeError as e:
        print(f"Error during remeshing: {e}. Try a larger voxel_size or check mesh integrity.")
        bpy.data.objects.remove(remesh_obj, do_unlink=True)
        for ob in original_selected:
            if ob and ob.name in bpy.data.objects: ob.select_set(True)
        if original_active and original_active.name in bpy.data.objects:
            bpy.context.view_layer.objects.active = original_active
        return 0.0
    
    if not remesh_obj.data.vertices:
        print("Error: Remeshed object has no vertices.")
        bpy.data.objects.remove(remesh_obj, do_unlink=True)
        # (Restore selection logic as above)
        return 0.0
    
    print("Calculating bounding box and grid dimensions...")
    world_verts = [remesh_obj.matrix_world @ v.co for v in remesh_obj.data.vertices]
    
    if not world_verts:
        print("Remeshed object has no world vertices.")
        bpy.data.objects.remove(remesh_obj, do_unlink=True)
        # (Restore selection logic as above)
        return 0.0

    min_coord_np = np.min(np.array([v.to_tuple() for v in world_verts]), axis=0)
    max_coord_np = np.max(np.array([v.to_tuple() for v in world_verts]), axis=0)

    padding = voxel_size 
    grid_origin_np = min_coord_np - padding
    grid_max_extent_np = max_coord_np + padding
    
    grid_dims_float = (grid_max_extent_np - grid_origin_np) / voxel_size
    nx = math.ceil(grid_dims_float[0]) if grid_dims_float[0] > 0 else 1
    ny = math.ceil(grid_dims_float[1]) if grid_dims_float[1] > 0 else 1
    nz = math.ceil(grid_dims_float[2]) if grid_dims_float[2] > 0 else 1

    if not (nx > 0 and ny > 0 and nz > 0):
        print(f"Error: Invalid grid dimensions ({nx}, {ny}, {nz}).")
        bpy.data.objects.remove(remesh_obj, do_unlink=True)
        # (Restore selection logic as above)
        return 0.0

    print(f"Grid dimensions (voxels): nx={nx}, ny={ny}, nz={nz}")

    is_solid = np.zeros((nx, ny, nz), dtype=bool)
    
    # --- 2. Populate is_solid Grid ---
    print("Populating solid voxel grid...")
    for v_obj_space in remesh_obj.data.vertices:
        v_world_np = np.array((remesh_obj.matrix_world @ v_obj_space.co).to_tuple())
        grid_coord_float = (v_world_np - grid_origin_np) / voxel_size
        ix, iy, iz = int(round(grid_coord_float[0])), int(round(grid_coord_float[1])), int(round(grid_coord_float[2]))
        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
            is_solid[ix, iy, iz] = True
    
    solid_voxel_count = np.sum(is_solid)
    if solid_voxel_count == 0:
        print("Warning: No solid voxels found.")
        # (Cleanup/restore logic as above)
        return 0.0
    print(f"Found {solid_voxel_count} solid voxels.")

    # --- 3. Identify Trapped Water ---
    print("Identifying trapped water with floor and wall enclosure...")
    final_trapped_water = np.zeros((nx, ny, nz), dtype=bool)

    for iz_slice in range(nz): # Iterate Z slices from bottom up
        print(f"  Processing slice Z={iz_slice}/{nz-1}...")
        current_slice_solid = is_solid[:, :, iz_slice]
        
        # Determine water on this slice that has floor support
        slice_water_with_floor = np.zeros((nx, ny), dtype=bool)
        for ix_w in range(nx):
            for iy_w in range(ny):
                if not current_slice_solid[ix_w, iy_w]: # Must not be solid itself
                    has_floor = False
                    if iz_slice == 0: # Grid bottom is a floor
                        has_floor = True
                    elif is_solid[ix_w, iy_w, iz_slice - 1]: # Solid floor below
                        has_floor = True
                    elif final_trapped_water[ix_w, iy_w, iz_slice - 1]: # Trapped water below acts as floor
                        has_floor = True
                    
                    if has_floor:
                        slice_water_with_floor[ix_w, iy_w] = True
        
        if not np.any(slice_water_with_floor):
            continue # No potential water with floor on this slice

        # Find laterally enclosed components on this slice
        visited_on_slice = np.zeros((nx, ny), dtype=bool)
        slice_trapped_this_level = np.zeros((nx, ny), dtype=bool)

        for ix_start in range(nx):
            for iy_start in range(ny):
                if slice_water_with_floor[ix_start, iy_start] and not visited_on_slice[ix_start, iy_start]:
                    component_coords = []
                    q = deque([(ix_start, iy_start)])
                    visited_on_slice[ix_start, iy_start] = True
                    is_component_laterally_enclosed = True # Assume enclosed until an opening is found

                    head = 0
                    while head < len(q): # Manual deque for faster appends if many small components
                        cx, cy = q[head]
                        head += 1
                        component_coords.append((cx, cy))

                        # Check 4 horizontal neighbors
                        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nb_x, nb_y = cx + dx, cy + dy

                            if not (0 <= nb_x < nx and 0 <= nb_y < ny): # Neighbor is outside grid
                                is_component_laterally_enclosed = False
                                continue # Don't break BFS, mark whole component as visited

                            if slice_water_with_floor[nb_x, nb_y]:
                                if not visited_on_slice[nb_x, nb_y]:
                                    visited_on_slice[nb_x, nb_y] = True
                                    q.append((nb_x, nb_y))
                            elif not current_slice_solid[nb_x, nb_y]: # Neighbor is not water_with_floor and not solid -> it's an opening
                                is_component_laterally_enclosed = False
                                # Don't break BFS from this component, continue to mark all its cells visited
                    
                    if is_component_laterally_enclosed:
                        for r_cx, r_cy in component_coords:
                            slice_trapped_this_level[r_cx, r_cy] = True
        
        final_trapped_water[:, :, iz_slice] = slice_trapped_this_level

    trapped_voxel_count = np.sum(final_trapped_water)
    print(f"Found {trapped_voxel_count} fully enclosed trapped water voxels.")

    # --- 4. Calculate Trapped Water Volume ---
    total_trapped_volume = trapped_voxel_count * (voxel_size ** 3)

    # --- 5. Debug Object Creation (Optional) ---
    if create_debug_objects:
        remesh_obj.name = target_obj_name + "_VoxelSolid_Debug"
        print(f"Kept solid voxel debug object: {remesh_obj.name}")

        if trapped_voxel_count > 0:
            print(f"Creating trapped water debug mesh for {trapped_voxel_count} voxels (optimized)...")
            puddle_bm = bmesh.new()
            half_vs = voxel_size / 2.0
            local_cube_verts = [
                mathutils.Vector((-half_vs, -half_vs, -half_vs)), mathutils.Vector(( half_vs, -half_vs, -half_vs)),
                mathutils.Vector(( half_vs,  half_vs, -half_vs)), mathutils.Vector((-half_vs,  half_vs, -half_vs)),
                mathutils.Vector((-half_vs, -half_vs,  half_vs)), mathutils.Vector(( half_vs, -half_vs,  half_vs)),
                mathutils.Vector(( half_vs,  half_vs,  half_vs)), mathutils.Vector((-half_vs,  half_vs,  half_vs)),
            ]
            cube_face_indices = [
                (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
                (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)
            ]
            trapped_indices = np.argwhere(final_trapped_water)
            for ix_p, iy_p, iz_p in trapped_indices:
                voxel_center_in_grid_space = (np.array([ix_p, iy_p, iz_p]) + 0.5) * voxel_size
                voxel_world_center_vec = mathutils.Vector((grid_origin_np + voxel_center_in_grid_space).tolist())
                current_cube_bmesh_verts = []
                for local_v in local_cube_verts:
                    current_cube_bmesh_verts.append(puddle_bm.verts.new(voxel_world_center_vec + local_v))
                for face_idx_tuple in cube_face_indices:
                    puddle_bm.faces.new([current_cube_bmesh_verts[i] for i in face_idx_tuple])
            
            puddle_mesh_data = bpy.data.meshes.new(name=target_obj_name + "_TrappedWater_Debug_Mesh")
            puddle_bm.to_mesh(puddle_mesh_data)
            puddle_bm.free()
            puddle_obj = bpy.data.objects.new(name=target_obj_name + "_TrappedWater_Debug", object_data=puddle_mesh_data)
            bpy.context.collection.objects.link(puddle_obj)
            print(f"Created trapped water debug object: {puddle_obj.name}")
        else:
            print("No trapped water voxels to create a debug mesh.")
    else:
        print(f"Cleaning up temporary remeshed object: {remesh_obj.name}")
        bpy.data.objects.remove(remesh_obj, do_unlink=True)

    # --- 6. Restore Original State ---
    for ob_sel in bpy.data.objects: ob_sel.select_set(False) 
    for ob_sel in original_selected: 
        if ob_sel and ob_sel.name in bpy.data.objects: ob_sel.select_set(True)
    if original_active and original_active.name in bpy.data.objects:
        bpy.context.view_layer.objects.active = original_active
    elif original_selected and len(original_selected) > 0 and \
         original_selected[0] and original_selected[0].name in bpy.data.objects:
        bpy.context.view_layer.objects.active = original_selected[0]
    elif bpy.context.selected_objects:
         bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]

    print(f"Estimated ENCLOSED puddle volume for '{target_obj_name}': {total_trapped_volume:.6f} Blender Units^3")
    return total_trapped_volume

# --- Example Usage ---
if __name__ == "__main__":
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    object_name_to_test = "tro_Special.001" # Change to your object
    original_obj = bpy.data.objects.get(object_name_to_test)

    if original_obj:
        bpy.context.view_layer.objects.active = original_obj
        for obj_iter in bpy.data.objects: obj_iter.select_set(False)
        original_obj.select_set(True)
        # bpy.ops.object.transform_apply(location=False, rotation=False, scale=True) # Good practice

        v_size = 0.1 
        if original_obj.dimensions.length > 0:
            avg_dim = sum(original_obj.dimensions) / 3.0 
            auto_voxel_size = avg_dim / 50
            if auto_voxel_size > 0.0001: v_size = auto_voxel_size
            print(f"Auto-calculated voxel size: {v_size:.4f}")
        else:
            print(f"Object has zero dimensions. Using default voxel size: {v_size}")
        
        print(f"\n--- Calculating ENCLOSED Puddle Volume for: {original_obj.name} with Voxel Size: {v_size:.4f} ---")
        estimated_volume = get_enclosed_puddle_volume(original_obj.name, v_size, create_debug_objects=True)
        print(f"--- Finished. Final Estimated ENCLOSED Volume: {estimated_volume:.6f} ---")
    else:
        print(f"Object '{object_name_to_test}' not found.")
