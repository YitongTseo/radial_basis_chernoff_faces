import subprocess
import sys
import os
import re 

def generate_face_mesh(
    blender_executable_path: str,
    analyzer_script_path: str,
    blend_file_to_open: str,
    dither_config: str,
    output_file: str,
    timeout_seconds: int = 300 # 5 minutes
    ):
    """
    Runs the Blender puddle analysis script headlessly.

    Args:
        blender_executable_path (str): Full path to the Blender executable.
        analyzer_script_path (str): Full path to the 'puddle_analyzer_headless.py'.
        blend_file_to_open (str): Path to the .blend file (e.g., 'face_blend.blend').
        object_name_in_blend (str): Name of the object to analyze.
        voxel_s (float): Voxel size.
        create_debug (bool): If True, tells Blender script to create debug meshes.
        verbose_blender_output (bool): If True, tells Blender script to print verbose logs.
        timeout_seconds (int): Timeout for the Blender process.

    Returns:
        dict or None: Parsed results (nx, ny, nz, solid_volume, trapped_water_vol, cupped_water_vol)
                      or None if an error occurred or output couldn't be parsed.
    """

    if not os.path.exists(blender_executable_path):
        print(f"Error: Blender executable not found at '{blender_executable_path}'")
        return None
    if not os.path.exists(analyzer_script_path):
        print(f"Error: Blender analyzer script not found at '{analyzer_script_path}'")
        return None
    if not os.path.exists(blend_file_to_open):
        print(f"Error: Specified .blend file not found at '{blend_file_to_open}'")
        return None
    if not os.path.exists(dither_config):
        print(f"Error: Specified config file not found at '{dither_config}'")
        return None

    command = [
        blender_executable_path,
        "-b", # Run in background (headless)
        # Removed opening .blend file from here; script handles it via its own arg
        # blend_file_to_open, # This would open it BEFORE the script runs
        "-P", analyzer_script_path,
        "--", # Separator for script arguments
        "--input", blend_file_to_open, # Pass blend file as arg to script
        "--dither_config", dither_config,
        "--output", output_file,
    ]

    try:
        # Setting PYTHONNOUSERSITE and PYTHONSAFEPATH can sometimes help avoid conflicts
        # with user's local Python packages if Blender's Python environment is sensitive.
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONSAFEPATH"] = "1"

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        stdout, stderr = process.communicate(timeout=timeout_seconds)

        if process.returncode != 0:
            print(f"Error running Blender script. Return code: {process.returncode}")
            return None
    
        return True
        
    except subprocess.TimeoutExpired:
        print(f"Error: Blender script timed out after {timeout_seconds} seconds.")
        if process: process.kill() # Ensure process is killed
        # stdout_on_timeout, stderr_on_timeout = process.communicate() # Get any remaining output
        # print("--- STDOUT (on timeout) ---\n", stdout_on_timeout)
        # print("--- STDERR (on timeout) ---\n", stderr_on_timeout)
        return None
    except Exception as e:
        print(f"An unexpected error occurred while running Blender script: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_blender_analysis(
    blender_executable_path: str,
    analyzer_script_path: str,
    blend_file_to_open: str,
    object_name_in_blend: str,
    voxel_s: float,
    create_debug: bool = False,
    verbose_blender_output: bool = False,
    timeout_seconds: int = 300 # 5 minutes
    ):
    """
    Runs the Blender puddle analysis script headlessly.

    Args:
        blender_executable_path (str): Full path to the Blender executable.
        analyzer_script_path (str): Full path to the 'puddle_analyzer_headless.py'.
        blend_file_to_open (str): Path to the .blend file (e.g., 'face_blend.blend').
        object_name_in_blend (str): Name of the object to analyze.
        voxel_s (float): Voxel size.
        create_debug (bool): If True, tells Blender script to create debug meshes.
        verbose_blender_output (bool): If True, tells Blender script to print verbose logs.
        timeout_seconds (int): Timeout for the Blender process.

    Returns:
        dict or None: Parsed results (nx, ny, nz, solid_volume, trapped_water_vol, cupped_water_vol)
                      or None if an error occurred or output couldn't be parsed.
    """

    if not os.path.exists(blender_executable_path):
        print(f"Error: Blender executable not found at '{blender_executable_path}'")
        return None
    if not os.path.exists(analyzer_script_path):
        print(f"Error: Blender analyzer script not found at '{analyzer_script_path}'")
        return None
    if not os.path.exists(blend_file_to_open):
        print(f"Error: Specified .blend file not found at '{blend_file_to_open}'")
        return None

    command = [
        blender_executable_path,
        "-b", # Run in background (headless)
        # Removed opening .blend file from here; script handles it via its own arg
        # blend_file_to_open, # This would open it BEFORE the script runs
        "-P", analyzer_script_path,
        "--", # Separator for script arguments
        "--blend_file", blend_file_to_open, # Pass blend file as arg to script
        "--object_name", object_name_in_blend,
        "--voxel_size", str(voxel_s),
    ]

    if create_debug:
        command.append("--create_debug_meshes")
    if verbose_blender_output:
        command.append("--verbose")
    
    print(f"Executing command: {' '.join(command)}")

    try:
        # Setting PYTHONNOUSERSITE and PYTHONSAFEPATH can sometimes help avoid conflicts
        # with user's local Python packages if Blender's Python environment is sensitive.
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONSAFEPATH"] = "1"

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        
        if verbose_blender_output or process.returncode != 0:
            print("--- Blender Script STDOUT ---")
            print(stdout)
            print("--- Blender Script STDERR ---")
            print(stderr)

        if process.returncode != 0:
            print(f"Error running Blender script. Return code: {process.returncode}")
            return None
        else:
            # Parse the specific output line
            # Expected format: 'nx <val> ny <val> nz <val> solid_volume <val> trapped_water_vol <val> cupped_water_vol <val>'
            output_lines = stdout.strip().split('\n')
            results_line = None
            for line in reversed(output_lines): # Find the last line that matches
                if line.startswith("nx ") and "solid_volume" in line:
                    results_line = line
                    break
        
            results_line

            # Split the string into a list
            items = results_line.split()

            # Convert to dictionary by pairing adjacent items
            result = {items[i]: int(items[i+1]) for i in range(0, len(items), 2)}

            return result

    except subprocess.TimeoutExpired:
        print(f"Error: Blender script timed out after {timeout_seconds} seconds.")
        if process: process.kill() # Ensure process is killed
        # stdout_on_timeout, stderr_on_timeout = process.communicate() # Get any remaining output
        # print("--- STDOUT (on timeout) ---\n", stdout_on_timeout)
        # print("--- STDERR (on timeout) ---\n", stderr_on_timeout)
        return None
    except Exception as e:
        print(f"An unexpected error occurred while running Blender script: {e}")
        import traceback
        traceback.print_exc()
        return None

