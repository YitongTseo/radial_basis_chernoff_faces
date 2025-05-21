# To see print statements:
# Open terminal then
# /Applications/Blender.app/Contents/MacOS/Blender


import bpy
import numpy as np
import mathutils
import math
from collections import deque
import bmesh # For creating debug puddle mesh

def get_tiered_puddle_volume(target_obj_name: str, voxel_size: float, create_debug_objects: bool = False) -> float:
    """
    Estimates trapped water volume on multi-level surfaces using a voxel fill & spill method.
    Optionally creates debug mesh objects for the solid voxels and puddle voxels.

    Args:
        target_obj_name (str): The name of the Blender mesh object.
        voxel_size (float): The side length of each cubic voxel.
        create_debug_objects (bool): If True, creates new mesh objects in the scene
                                     for the voxelized solid and the trapped water voxels.
    Returns:
        float: The estimated total trapped volume in Blender units^3.
    """
    target_obj = bpy.data.objects.get(target_obj_name)
    if not target_obj or target_obj.type != 'MESH':
        print(f"Error: Object '{target_obj_name}' not found or not a mesh.")
        return 0.0

    # --- 1. Preparation: Duplicate and Remesh ---
    original_active = bpy.context.view_layer.objects.active
    original_selected = bpy.context.selected_objects[:]

    # Ensure the target object is selected and active for ops
    bpy.context.view_layer.objects.active = target_obj
    for ob in bpy.data.objects: ob.select_set(False) # Deselect all
    target_obj.select_set(True)

    bpy.ops.object.duplicate()
    remesh_obj = bpy.context.active_object # The new duplicate
    remesh_obj.name = target_obj_name + "_RemeshVoxelTemp" # Temporary name

    print(f"Remeshing '{remesh_obj.name}' with voxel size: {voxel_size}...")
    try:
        mod = remesh_obj.modifiers.new(name="VoxelRemesh", type='REMESH')
        mod.mode = 'VOXEL'
        mod.voxel_size = voxel_size
        mod.use_remove_disconnected = False 
        mod.adaptivity = 0 
        bpy.ops.object.modifier_apply(modifier=mod.name)
    except RuntimeError as e:
        print(f"Error during remeshing: {e}. Try a larger voxel_size or check mesh.")
        bpy.data.objects.remove(remesh_obj, do_unlink=True)
        # Restore original selection and active object
        for ob in original_selected: ob.select_set(True)
        bpy.context.view_layer.objects.active = original_active
        return 0.0
    
    if not remesh_obj.data.vertices:
        print("Error: Remeshed object has no vertices. Voxel size might be too large or original mesh is empty.")
        bpy.data.objects.remove(remesh_obj, do_unlink=True)
        for ob in original_selected: ob.select_set(True)
        bpy.context.view_layer.objects.active = original_active
        return 0.0
    
    print("Calculating bounding box and grid dimensions...")
    # --- Get Bounding Box and Grid Dimensions (World Space) ---
    # Apply object transformations to the remeshed object before getting vertex coords
    # This ensures its vertices are in world space if it had any local transform.
    # However, duplication and remesh usually result in world space if original was.
    # For safety, ensure matrix_world is identity or handle it:
    # bpy.ops.object.transform_apply(location=True, rotation=True, scale=True) # On remesh_obj

    world_verts = [remesh_obj.matrix_world @ v.co for v in remesh_obj.data.vertices]
    
    if not world_verts:
        print("Remeshed object has no world vertices after matrix multiplication. Cannot proceed.")
        bpy.data.objects.remove(remesh_obj, do_unlink=True)
        for ob in original_selected: ob.select_set(True)
        bpy.context.view_layer.objects.active = original_active
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


    if not (nx > 0 and ny > 0 and nz > 0): # Check if any dimension is zero or negative
        print(f"Error: Invalid grid dimensions ({nx}, {ny}, {nz}). Check object scale or voxel size.")
        bpy.data.objects.remove(remesh_obj, do_unlink=True)
        for ob in original_selected: ob.select_set(True)
        bpy.context.view_layer.objects.active = original_active
        return 0.0

    print(f"Grid dimensions (voxels): nx={nx}, ny={ny}, nz={nz}")

    is_solid = np.zeros((nx, ny, nz), dtype=bool)
    is_water = np.zeros((nx, ny, nz), dtype=bool)
    can_escape = np.zeros((nx, ny, nz), dtype=bool)

    # --- 2. Populate is_solid Grid ---
    print("Populating solid voxel grid...")
    # The remeshed object's vertices are at the centers of the solid voxels.
    # We need to map these world coordinates to our grid indices.
    for v_obj_space in remesh_obj.data.vertices:
        v_world_np = np.array((remesh_obj.matrix_world @ v_obj_space.co).to_tuple())
        # Calculate grid index for this vertex
        grid_coord_float = (v_world_np - grid_origin_np) / voxel_size
        # It's possible due to float precision that a coord is slightly outside.
        # We should use the center of the voxel the vertex represents.
        # The remesh vertices ARE the centers of the created voxels.
        ix = int(round(grid_coord_float[0])) # Round to nearest index
        iy = int(round(grid_coord_float[1]))
        iz = int(round(grid_coord_float[2]))


        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
            is_solid[ix, iy, iz] = True
        # else:
            # print(f"Warning: Solid voxel vertex {v_world_np} maps to outside grid index ({ix},{iy},{iz}). Check padding or calculations.")
    
    solid_voxel_count = np.sum(is_solid)
    if solid_voxel_count == 0:
        print("Warning: No solid voxels found in the grid. Check remesh output or voxel_size vs object size.")
        if not create_debug_objects:
            bpy.data.objects.remove(remesh_obj, do_unlink=True)
        else:
            remesh_obj.name = target_obj_name + "_VoxelSolid_Debug_Empty"
        for ob in original_selected: ob.select_set(True)
        bpy.context.view_layer.objects.active = original_active
        return 0.0
    print(f"Found {solid_voxel_count} solid voxels.")


    # --- 3. Initial Water Fill (Bottom-Up) ---
    print("Performing initial water fill...")
    for iz_fill in range(nz):
        for ix_fill in range(nx):
            for iy_fill in range(ny):
                if is_solid[ix_fill, iy_fill, iz_fill]:
                    continue # Skip solid voxels

                supported = False
                if iz_fill == 0: # Bottom of the grid acts as ground
                    supported = True
                else: # Check voxel below
                    if is_solid[ix_fill, iy_fill, iz_fill - 1] or is_water[ix_fill, iy_fill, iz_fill - 1]:
                        supported = True
                
                if supported:
                    is_water[ix_fill, iy_fill, iz_fill] = True
    
    initial_water_count = np.sum(is_water)
    if initial_water_count == 0:
        print("No potential water voxels after initial fill. This means all empty space is unsupported from below or the object is flat on the grid bottom.")
        # Continue to see if any of this space is trapped by other means (though unlikely with this fill logic)
    print(f"Found {initial_water_count} initial water voxels.")

    # --- 4. Identify Escape Routes (BFS from Openings) ---
    print("Identifying escape routes via BFS...")
    bfs_queue = deque()

    # Seed queue from water voxels at the top layer (primary escape route)
    for ix_q in range(nx):
        for iy_q in range(ny):
            if is_water[ix_q, iy_q, nz - 1] and not can_escape[ix_q, iy_q, nz - 1]:
                bfs_queue.append((ix_q, iy_q, nz - 1))
                can_escape[ix_q, iy_q, nz - 1] = True
    
    # (Optional) Seed from sides if they are considered open
    # for iz_side in range(nz):
    #     for ix_side in range(nx):
    #         if is_water[ix_side, 0, iz_side] and not can_escape[ix_side, 0, iz_side]: # Front
    #             bfs_queue.append((ix_side, 0, iz_side)); can_escape[ix_side, 0, iz_side] = True
    #         if is_water[ix_side, ny - 1, iz_side] and not can_escape[ix_side, ny - 1, iz_side]: # Back
    #             bfs_queue.append((ix_side, ny - 1, iz_side)); can_escape[ix_side, ny - 1, iz_side] = True
    #     for iy_side in range(ny):
    #         if is_water[0, iy_side, iz_side] and not can_escape[0, iy_side, iz_side]: # Left
    #             bfs_queue.append((0, iy_side, iz_side)); can_escape[0, iy_side, iz_side] = True
    #         if is_water[nx - 1, iy_side, iz_side] and not can_escape[nx - 1, iy_side, iz_side]: # Right
    #             bfs_queue.append((nx - 1, iy_side, iz_side)); can_escape[nx - 1, iy_side, iz_side] = True

    # Define 26 neighbors for 3D BFS (allows diagonal flow)
    deltas = []
    for dx_bfs in [-1, 0, 1]:
        for dy_bfs in [-1, 0, 1]:
            for dz_bfs in [-1, 0, 1]:
                if dx_bfs == 0 and dy_bfs == 0 and dz_bfs == 0:
                    continue
                deltas.append((dx_bfs, dy_bfs, dz_bfs))

    while bfs_queue:
        cx, cy, cz = bfs_queue.popleft()
        for dx_bfs, dy_bfs, dz_bfs in deltas:
            nb_x, nb_y, nb_z = cx + dx_bfs, cy + dy_bfs, cz + dz_bfs

            if 0 <= nb_x < nx and 0 <= nb_y < ny and 0 <= nb_z < nz: # Check bounds
                # If neighbor is water and hasn't been marked as escapable yet
                if is_water[nb_x, nb_y, nb_z] and not can_escape[nb_x, nb_y, nb_z]:
                    can_escape[nb_x, nb_y, nb_z] = True
                    bfs_queue.append((nb_x, nb_y, nb_z))
    
    escaped_water_count = np.sum(can_escape & is_water) 
    print(f"{escaped_water_count} water voxels can escape.")

    # --- 5. Calculate Trapped Volume ---
    trapped_mask = is_water & (~can_escape) # Water AND NOT escapable
    trapped_voxel_count = np.sum(trapped_mask)
    total_trapped_volume = trapped_voxel_count * (voxel_size ** 3)
    print(f"Found {trapped_voxel_count} trapped water voxels.")

    # --- 6. Debug Object Creation (Optional) ---
    if create_debug_objects:
        remesh_obj.name = target_obj_name + "_VoxelSolid_Debug"
        print(f"Kept solid voxel debug object: {remesh_obj.name}")

        if trapped_voxel_count > 0:
            print("Creating puddle debug mesh...")
            puddle_bm = bmesh.new()
            
            # Get all (ix, iy, iz) indices for trapped voxels
            trapped_indices = np.argwhere(trapped_mask) 
            
            for ix_p, iy_p, iz_p in trapped_indices:
                # Calculate world position for the center of this voxel
                voxel_center_in_grid_space = (np.array([ix_p, iy_p, iz_p]) + 0.5) * voxel_size
                voxel_world_center_np = grid_origin_np + voxel_center_in_grid_space
                voxel_world_center_vec = mathutils.Vector(voxel_world_center_np.tolist())
                
                # Create a cube in the bmesh, already centered at bmesh origin
                geom = bmesh.ops.create_cube(puddle_bm, size=voxel_size) 
                
                # Translate the new cube's vertices to the calculated world position
                # geom['verts'] contains the vertices of the just-created cube
                bmesh.ops.translate(puddle_bm, verts=geom['verts'], vec=voxel_world_center_vec)

            puddle_mesh_data = bpy.data.meshes.new(name=target_obj_name + "_Puddles_Debug_Mesh")
            puddle_bm.to_mesh(puddle_mesh_data)
            puddle_bm.free()
            
            puddle_obj = bpy.data.objects.new(name=target_obj_name + "_Puddles_Debug", object_data=puddle_mesh_data)
            bpy.context.collection.objects.link(puddle_obj)
            print(f"Created puddle debug object: {puddle_obj.name} with {trapped_voxel_count} voxels.")
        else:
            print("No trapped voxels to create a puddle debug mesh.")
    else:
        # --- 7. Cleanup if not keeping debug objects ---
        print(f"Cleaning up temporary remeshed object: {remesh_obj.name}")
        bpy.data.objects.remove(remesh_obj, do_unlink=True)

    # Restore original selection and active object
    for ob_sel in bpy.data.objects: ob_sel.select_set(False) # Deselect all first
    for ob_sel in original_selected: 
        if ob_sel: # Check if object still exists
            ob_sel.select_set(True)
    if original_active and original_active.name in bpy.data.objects: # Check if active object still exists
        bpy.context.view_layer.objects.active = original_active
    elif original_selected and original_selected[0].name in bpy.data.objects: # Fallback if original active was deleted
         bpy.context.view_layer.objects.active = original_selected[0]


    print(f"Estimated tiered puddle volume for '{target_obj_name}': {total_trapped_volume:.6f} Blender Units^3")
    return total_trapped_volume

# --- Example Usage (goes at the end of your script or in a separate script) ---
if __name__ == "__main__":
    # This part runs if you execute the script directly in Blender
    
    # Ensure we're in Object Mode. This is important before many operations.
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    # Define the name of the object you want to process
    object_name_to_test = "Yitong_Face"  # <--- CHANGE THIS TO YOUR OBJECT'S NAME
    original_obj = bpy.data.objects.get(object_name_to_test)

    if original_obj:
        # Make 'original_obj' the active object and select it exclusively
        bpy.context.view_layer.objects.active = original_obj
        for obj_iter in bpy.data.objects: # Deselect all other objects
            obj_iter.select_set(False)
        original_obj.select_set(True)

        # Determine a reasonable voxel size based on original object dimensions
        v_size = 0.1 # Default voxel size
        # Ensure scale is applied on your object for dimensions to be accurate (Object > Apply > Scale)
        if original_obj.dimensions.length > 0:
            avg_dim = sum(original_obj.dimensions) / 3.0 
            auto_voxel_size = avg_dim / 50  # Adjust divisor for detail (e.g., 30 for coarser, 100 for finer)
            
            if auto_voxel_size > 0.0001: # Prevent extremely small voxel sizes
                v_size = auto_voxel_size
                print(f"Auto-calculated voxel size for '{original_obj.name}' based on its dimensions: {v_size:.4f}")
            else:
                print(f"Auto-calculated voxel size is very small or zero for '{original_obj.name}'. Using default: {v_size}. Check object scale and dimensions.")
        else:
            print(f"'{original_obj.name}' has zero dimensions (is it empty or scaled to zero?). Using default voxel size: {v_size}")
        
        print(f"\n--- Calculating Puddle Volume for: {original_obj.name} with Voxel Size: {v_size:.4f} ---")
        # Pass create_debug_objects=True to see the voxel meshes
        estimated_volume = get_tiered_puddle_volume(original_obj.name, v_size, create_debug_objects=True)
        print(f"--- Finished. Final Estimated Volume: {estimated_volume:.6f} ---")
    else:
        print(f"Object '{object_name_to_test}' not found in the scene. Please ensure it exists.")

