import json
import os
import csv
import random
import shutil
import uuid
# import functools # No longer explicitly used by functools.partial
from deap import base, creator, tools # Keep tools for ParetoFront, Statistics
# from deap import algorithms # No longer using DEAP algorithms
from utils import generate_face_mesh, run_blender_analysis
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np
import time

# --- Configuration ---
BLENDER_EXECUTABLE_PATH = '/Applications/Blender.app/Contents/MacOS/Blender'
GENERATION_HELPER_SCRIPT_PATH = 'blender_rbf_script.py'
ANALYSIS_SCRIPT_PATH = 'water_trapping.py'
BASE_INPUT_BLEND_FILE = 'face_landmark_points.blend'
OBJECT_NAME_TO_ANALYZE = "Yitong_Face"
ANALYSIS_VOXEL_SIZE = 0.1

# POPULATION_SIZE is now effectively replaced by MAX_EVALS_TPE per block
# POPULATION_SIZE = 1 # This will be ignored in the new structure for #evals per block
N_TPE_BLOCKS = 3 # Total number of TPE optimization blocks (e.g., Gen 0, Gen 1, Gen 2)
                 # If your N_GENERATIONS was 2 (meaning G0, G1, G2), then N_TPE_BLOCKS = N_GENERATIONS + 1
MAX_EVALS_PER_TPE_BLOCK = 3   # Number of evaluations hyperopt performs in each block

FITNESS_WEIGHTS = (1.0, 0.2) # (Surface Area, Cupped Water Vol)
GENE_MUTATION_MULTIPLIER = 5
GENE_MUTATION_BASE_RATE = 0.08

# --- Landmark Definition ---
try:
    with open('example_params.json', 'r') as f:
        example_params_data = json.load(f)
except FileNotFoundError:
    print("Error: 'example_params.json' not found. Please ensure it's in the current directory or update the path.")
    exit(1)

LANDMARK_NAMES = [key for key in example_params_data.keys() if key != "default"]
DEFAULT_PARAMS_SECTION = example_params_data.get("default", {"direction": [0, 0, 0], "magnitude": 0.0})
NUM_GENES_PER_LANDMARK = 3
TOTAL_NUM_GENES = len(LANDMARK_NAMES) * NUM_GENES_PER_LANDMARK
print(f'We have this many genes: {TOTAL_NUM_GENES}')
GENOME_MUTATIONS_DF = pd.read_csv('genomes/most_variable_gene_mutations.csv')
LANDMARK_TO_GENES_RANKING = pd.read_csv('landmark_ranking.csv', comment="#", header=None,)[0].to_list()

# --- Temporary Directory for EA files ---
TEMP_DIR = "INDIVIDUALS_EVOLVING"
if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)
os.makedirs(TEMP_DIR, exist_ok=True)

# --- DEAP Setup (Minimal) ---
creator.create("FitnessMultiMax", base.Fitness, weights=FITNESS_WEIGHTS)
creator.create("Individual", list, fitness=creator.FitnessMultiMax) # Still useful for HOF and stats

# toolbox = base.Toolbox() # Not much needed from toolbox anymore
# toolbox.register("attr_float", random.uniform, -1.0, 1.0)
# toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, TOTAL_NUM_GENES)
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)

regeneration_cache = {}

def individual_to_params_json(individual_genes, current_block_num): # Renamed current_generation_num
    if current_block_num == 0:
        col_of_interest = "Reference->03.2020 Mutational Load"
    elif current_block_num == 1:
        col_of_interest = "03.2020->09.2023 Mutational Load"
    elif current_block_num == 2:
        col_of_interest = "09.2023->01.2025 Mutational Load"
    else:
        print(f"Warning: TPE Block number {current_block_num} out of defined range for mutational load. Using last known or default.")
        # Fallback to the last defined column or a default behavior
        if GENOME_MUTATIONS_DF.shape[1] > 0 : # Check if DF has columns
             col_of_interest = GENOME_MUTATIONS_DF.columns[-1] if "Mutational Load" in GENOME_MUTATIONS_DF.columns[-1] else "09.2023->01.2025 Mutational Load" # GENOME_MUTATIONS_DF.columns[0]
        else: # GENOME_MUTATIONS_DF might be empty or not have expected columns
            col_of_interest = None # This will cause generational_mutation_load to be 0 later
            print("Warning: GENOME_MUTATIONS_DF seems to be empty or misconfigured for mutational load.")

    params_dict = {"default": DEFAULT_PARAMS_SECTION}
    gene_idx = 0
    for landmark_name in LANDMARK_NAMES:
        direction = [
            max(-1.0, min(1.0, individual_genes[gene_idx])),
            max(-1.0, min(1.0, individual_genes[gene_idx + 1])),
            max(-1.0, min(1.0, individual_genes[gene_idx + 2])),
        ]
        generational_mutation_load = 0.0 # Default
        if col_of_interest: # Proceed only if col_of_interest was set
            try:
                landmark_ranking = LANDMARK_TO_GENES_RANKING.index(landmark_name)
                gene_of_interest = GENOME_MUTATIONS_DF.iloc[landmark_ranking]
                generational_mutation_load = gene_of_interest[col_of_interest]
            except (ValueError, IndexError, KeyError) as e: # Added KeyError for col_of_interest
                print(f"Warning: Landmark {landmark_name} or column {col_of_interest} issue. Using default magnitude. Error: {e}")
        
        params_dict[landmark_name] = {
            "direction": direction,
            "magnitude": GENE_MUTATION_MULTIPLIER * generational_mutation_load + GENE_MUTATION_BASE_RATE,
        }
        gene_idx += 3
    return params_dict

def ensure_blend_file_exists(target_blend_filename,
                             all_individuals_log,
                             temp_dir_path,
                             blender_exec, gen_script, base_blend):
    global regeneration_cache
    if target_blend_filename == base_blend:
        return base_blend
    target_blend_full_path = os.path.join(temp_dir_path, target_blend_filename)
    if target_blend_filename in regeneration_cache and os.path.exists(target_blend_full_path):
        return target_blend_full_path
    if os.path.exists(target_blend_full_path):
        regeneration_cache[target_blend_filename] = target_blend_full_path
        return target_blend_full_path
    print(f"    File {target_blend_filename} not found. Attempting regeneration.")
    log_entry_for_target_blend = next((e for e in reversed(all_individuals_log) if e["blend_file"] == target_blend_filename), None)
    if not log_entry_for_target_blend:
        raise FileNotFoundError(f"Log entry for {target_blend_filename} not found. Cannot regenerate.")
    parent_blend_ref_for_target = log_entry_for_target_blend["parent_blend_file"]
    params_file_for_target = os.path.join(temp_dir_path, log_entry_for_target_blend["params_file"])
    if not os.path.exists(params_file_for_target):
        raise FileNotFoundError(f"Params file {params_file_for_target} for {target_blend_filename} not found.")
    print(f"    Recursively ensuring parent blend '{parent_blend_ref_for_target}' for '{target_blend_filename}'.")
    input_blend_for_regeneration = ensure_blend_file_exists(parent_blend_ref_for_target, all_individuals_log, temp_dir_path, blender_exec, gen_script, base_blend)
    if not os.path.exists(input_blend_for_regeneration) and input_blend_for_regeneration != base_blend:
        raise FileNotFoundError(f"Input blend '{input_blend_for_regeneration}' for {target_blend_filename} not found.")
    print(f"      Regenerating {target_blend_filename} using {os.path.basename(input_blend_for_regeneration)} and {log_entry_for_target_blend['params_file']}")
    if target_blend_filename in regeneration_cache: del regeneration_cache[target_blend_filename]
    success_regen = generate_face_mesh(blender_executable_path=blender_exec, analyzer_script_path=gen_script, blend_file_to_open=input_blend_for_regeneration, dither_config=params_file_for_target, output_file=target_blend_full_path)
    if not success_regen or not os.path.exists(target_blend_full_path):
        raise Exception(f"Regeneration of {target_blend_filename} failed at {target_blend_full_path}.")
    print(f"    Successfully regenerated {target_blend_filename}.")
    regeneration_cache[target_blend_filename] = target_blend_full_path
    return target_blend_full_path

def evaluate_and_log_individual(
    individual_genes, current_block_num, # Renamed
    parent_individual_id_for_log, input_blend_for_this_eval,
    detailed_log_list_in_memory, csv_writer_object, csv_file_handle,
    blender_exec, gen_script_path, analysis_script_path, base_blend_name, obj_name, voxel_size, temp_dir_path
):
    individual_id = str(uuid.uuid4()) # Unique ID for this new individual being evaluated
    param_filename = f"params_block{current_block_num}_{individual_id}.json"  # Use block in name
    blend_filename = f"output_block{current_block_num}_{individual_id}.blend" # Use block in name
    
    params_json_path = os.path.join(temp_dir_path, param_filename)
    generated_blend_path = os.path.join(temp_dir_path, blend_filename)

    # Pass current_block_num to individual_to_params_json
    params_data = individual_to_params_json(individual_genes, current_block_num)
    with open(params_json_path, 'w') as f:
        json.dump(params_data, f, indent=2)

    parent_blend_to_log_in_csv = os.path.basename(input_blend_for_this_eval) if input_blend_for_this_eval != base_blend_name else base_blend_name

    log_entry = {
        "generation": current_block_num, # Changed to generation for consistency with CSV
        "individual_id": individual_id,
        "parent_individual_id": parent_individual_id_for_log,
        "parent_blend_file": parent_blend_to_log_in_csv,
        "params_file": param_filename,
        "blend_file": None, "nx": None, "ny": None, "nz": None, "solid_volume": None,
        "trapped_water_vol": None, "surface_area": None, "cupped_water_vol": None,
        "generation_status": "pending", "analysis_status": "pending",
        "genes": str(list(individual_genes)),
    }

    success_generation = False
    try:
        success_generation = generate_face_mesh(
            blender_executable_path=blender_exec,
            analyzer_script_path=gen_script_path,
            blend_file_to_open=input_blend_for_this_eval,
            dither_config=params_json_path,
            output_file=generated_blend_path,
        )
    except Exception as e:
        print(f"  Exception during generate_face_mesh for {individual_id} (using {os.path.basename(input_blend_for_this_eval)}): {e}")
        log_entry["generation_status"] = f"error: {e}"

    current_fitness = tuple(-float('inf') for _ in FITNESS_WEIGHTS)

    if not success_generation or not os.path.exists(generated_blend_path):
        if success_generation and log_entry["generation_status"] == "pending":
             log_entry["generation_status"] = "failed_no_output"
        elif log_entry["generation_status"] == "pending":
            log_entry["generation_status"] = "failed_generation_error"
    else:
        log_entry["generation_status"] = "success"
        log_entry["blend_file"] = blend_filename 

        analysis_results = None
        try:
            analysis_results = run_blender_analysis(
                blender_executable_path=blender_exec,
                analyzer_script_path=analysis_script_path,
                blend_file_to_open=generated_blend_path,
                object_name_in_blend=obj_name,
                voxel_s=voxel_size, create_debug=False, verbose_blender_output=False
            )
        except Exception as e:
            print(f"  Exception during run_blender_analysis for {individual_id}: {e}")
            log_entry["analysis_status"] = f"error: {e}"
        
        if analysis_results is None:
            if log_entry["analysis_status"] == "pending":
                log_entry["analysis_status"] = "failed_no_results"
        else:
            log_entry["analysis_status"] = "success"
            sa = analysis_results.get('surface_area', 0)
            cwv = analysis_results.get('cupped_water_vol', 0)
            log_entry.update({k: analysis_results.get(k) for k in ['surface_area', 'cupped_water_vol', 'nx', 'ny', 'nz', 'trapped_water_vol', 'solid_volume']})
            current_fitness = (sa, cwv)

        if log_entry["generation_status"] == "success" and log_entry["analysis_status"] == "success" and os.path.exists(generated_blend_path):
            print(f"    Deleting analyzed blend file: {os.path.basename(generated_blend_path)}")
            try:
                os.remove(generated_blend_path)
            except OSError as e:
                print(f"    Warning: Could not delete {generated_blend_path}: {e}")
        elif os.path.exists(generated_blend_path):
             print(f"    Keeping blend file due to non-successful analysis: {os.path.basename(generated_blend_path)}")

    detailed_log_list_in_memory.append(log_entry)
    csv_writer_object.writerow(log_entry)
    csv_file_handle.flush()
    
    return current_fitness, log_entry # Return fitness and the log_entry dict

def format_duration(seconds):
    """Formats a duration in seconds into a human-readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def main():
    global regeneration_cache
    regeneration_cache.clear()

    print("Evolutionary Algorithm driven by TPE Optimization Blocks")
    print(f"Output and logs will be in: {os.path.abspath(TEMP_DIR)}")
    print(f"Running {N_TPE_BLOCKS} TPE blocks, each with {MAX_EVALS_PER_TPE_BLOCK} evaluations.")
    print("---")

    hof = tools.ParetoFront() # For multi-objective tracking across all evaluations
    stats = tools.Statistics(lambda ind: ind.fitness.values) # For per-block stats
    stats.register("avg", lambda x: tuple(round(sum(col)/len(col),3) for col in zip(*x)) if x and all(isinstance(i,tuple) and len(i)==len(FITNESS_WEIGHTS) for i in x) else tuple(None for _ in FITNESS_WEIGHTS))
    stats.register("min", lambda x: tuple(round(min(col),3) for col in zip(*x)) if x and all(isinstance(i,tuple) and len(i)==len(FITNESS_WEIGHTS) for i in x) else tuple(None for _ in FITNESS_WEIGHTS))
    stats.register("max", lambda x: tuple(round(max(col),3) for col in zip(*x)) if x and all(isinstance(i,tuple) and len(i)==len(FITNESS_WEIGHTS) for i in x) else tuple(None for _ in FITNESS_WEIGHTS))
    
    block_summary_log = [] # Renamed from generational_summary_log
    all_individuals_detailed_log_in_memory = []

    detailed_log_csv_path = os.path.join(TEMP_DIR, "all_individuals_evaluation_log.csv")
    csv_fieldnames = [
        "generation", "individual_id", "parent_individual_id", "parent_blend_file",
        "params_file", "blend_file", "nx", "ny", "nz", "solid_volume",
        "trapped_water_vol", "surface_area", "cupped_water_vol",
        "generation_status", "analysis_status", "genes",
    ]

    with open(detailed_log_csv_path, 'w', newline='') as csvfile_handle: # Changed to 'w' to overwrite for new runs
        csv_writer = csv.DictWriter(csvfile_handle, fieldnames=csv_fieldnames)
        csv_writer.writeheader()

        current_input_blend_for_block = BASE_INPUT_BLEND_FILE
        parent_log_entry_for_block_input = None # Stores log entry of individual providing the input blend

        for block_num in range(N_TPE_BLOCKS):
            print(f"\n--- TPE Optimization Block {block_num}/{N_TPE_BLOCKS - 1} ---")
            print(f"--- Using input blend: {os.path.basename(current_input_blend_for_block)} ---")
            if parent_log_entry_for_block_input:
                print(f"--- Input blend from individual ID: {parent_log_entry_for_block_input['individual_id']} from Block {parent_log_entry_for_block_input['generation']} ---")

            regeneration_cache.clear() # Clear for each block to be safe, or manage more carefully

            # Define the objective function for this TPE block
            # It captures block_num and current_input_blend_for_block from the outer scope.
            def tpe_block_objective(individual_genes_from_hyperopt):
                parent_id_for_current_eval = parent_log_entry_for_block_input["individual_id"] if parent_log_entry_for_block_input else "TPE_BLOCK_START"
                
                # evaluate_and_log_individual now returns (fitness_tuple, log_entry_dict)
                fitness_tuple, logged_entry = evaluate_and_log_individual(
                    individual_genes=individual_genes_from_hyperopt,
                    current_block_num=block_num,
                    parent_individual_id_for_log=parent_id_for_current_eval,
                    input_blend_for_this_eval=current_input_blend_for_block,
                    detailed_log_list_in_memory=all_individuals_detailed_log_in_memory,
                    csv_writer_object=csv_writer, csv_file_handle=csvfile_handle,
                    blender_exec=BLENDER_EXECUTABLE_PATH, gen_script_path=GENERATION_HELPER_SCRIPT_PATH,
                    analysis_script_path=ANALYSIS_SCRIPT_PATH, base_blend_name=BASE_INPUT_BLEND_FILE,
                    obj_name=OBJECT_NAME_TO_ANALYZE, voxel_size=ANALYSIS_VOXEL_SIZE, temp_dir_path=TEMP_DIR
                )

                sa, cwv = fitness_tuple
                loss = float('inf') # Default for minimization if fitness is invalid
                if sa > -float('inf') and cwv > -float('inf'):
                     loss = -(FITNESS_WEIGHTS[0] * sa + FITNESS_WEIGHTS[1] * cwv)
                
                return {'loss': loss, 'status': STATUS_OK, 
                        'genes': list(individual_genes_from_hyperopt), 
                        'true_fitness': fitness_tuple,
                        'log_entry_id': logged_entry['individual_id']} # Pass back log_entry_id

            hyperopt_space = [hp.uniform(f'g{k}', -1.0, 1.0) for k in range(TOTAL_NUM_GENES)]
            trials_this_block = Trials()
            
            print(f"  Running TPE for {MAX_EVALS_PER_TPE_BLOCK} evaluations in block {block_num}...")
            
            try:
                fmin( # best_run_params not strictly needed if using trials object
                    fn=tpe_block_objective,
                    space=hyperopt_space,
                    algo=tpe.suggest,
                    max_evals=MAX_EVALS_PER_TPE_BLOCK,
                    trials=trials_this_block,
                    rstate=np.random.default_rng(random.randint(0, 100000) + block_num),
                    verbose=1 
                )
            except Exception as e_fmin:
                print(f"  Error during fmin for TPE block {block_num}: {e_fmin}")
                if block_num < N_TPE_BLOCKS - 1: # If not the last block
                    print(f"  Reverting to BASE_INPUT_BLEND_FILE for the next block due to fmin error.")
                    current_input_blend_for_block = BASE_INPUT_BLEND_FILE
                    parent_log_entry_for_block_input = None
                continue # Skip to next block

            if not trials_this_block.results:
                print(f"  Warning: No results from TPE block {block_num}. Cannot determine next input blend.")
                if block_num < N_TPE_BLOCKS - 1:
                     current_input_blend_for_block = BASE_INPUT_BLEND_FILE
                     parent_log_entry_for_block_input = None
                continue

            # Process results from this TPE block for HOF, stats, and next input blend
            block_individuals_for_processing = []
            for trial_result in trials_this_block.results:
                # Create a DEAP individual for HOF/stats
                deap_ind = creator.Individual(trial_result['genes'])
                deap_ind.fitness.values = trial_result['true_fitness']
                block_individuals_for_processing.append(deap_ind)
            
            if block_individuals_for_processing:
                hof.update(block_individuals_for_processing) # Update HOF with all from this block
                record = stats.compile(block_individuals_for_processing)
                block_summary_log.append({"block": block_num, "type": "TPE_Block", **record})
                print(f"TPE Block {block_num} Summary: {record}")

                # Determine the best individual from this block to set up the next input blend
                best_trial_this_block = min(trials_this_block.results, key=lambda r: r['loss'])
                best_log_entry_id_this_block = best_trial_this_block['log_entry_id']
                
                # Find the full log entry for this best individual
                found_best_log_entry = next((le for le in all_individuals_detailed_log_in_memory if le['individual_id'] == best_log_entry_id_this_block), None)

                if found_best_log_entry and found_best_log_entry.get("blend_file") and found_best_log_entry.get("generation_status") == "success":
                    parent_log_entry_for_block_input = found_best_log_entry # This will be the "parent" for the next block's evaluations
                    if block_num < N_TPE_BLOCKS - 1: # If not the last block
                        next_input_blend_filename = found_best_log_entry["blend_file"]
                        print(f"  Best individual from TPE block {block_num} is {found_best_log_entry['individual_id']} (Loss: {best_trial_this_block['loss']:.4f}).")
                        try:
                            print(f"  Ensuring blend file '{next_input_blend_filename}' exists for next block.")
                            current_input_blend_for_block = ensure_blend_file_exists(
                                next_input_blend_filename, all_individuals_detailed_log_in_memory, TEMP_DIR,
                                BLENDER_EXECUTABLE_PATH, GENERATION_HELPER_SCRIPT_PATH, BASE_INPUT_BLEND_FILE
                            )
                        except Exception as e_ensure:
                            print(f"    CRITICAL: Failed to ensure/regenerate {next_input_blend_filename} for next TPE block: {e_ensure}. Reverting to base file.")
                            current_input_blend_for_block = BASE_INPUT_BLEND_FILE
                            parent_log_entry_for_block_input = None # Its blend isn't available
                elif block_num < N_TPE_BLOCKS - 1:
                    print(f"  Warning: Best trial from TPE block {block_num} (ID: {best_log_entry_id_this_block if 'best_log_entry_id_this_block' in locals() else 'N/A'}) did not result in a usable blend file. Reverting to base for next block.")
                    current_input_blend_for_block = BASE_INPUT_BLEND_FILE
                    parent_log_entry_for_block_input = None
            elif block_num < N_TPE_BLOCKS - 1 : # No individuals processed from block
                print(f"  Warning: No individuals successfully processed in TPE block {block_num}. Reverting to base file for next block.")
                current_input_blend_for_block = BASE_INPUT_BLEND_FILE
                parent_log_entry_for_block_input = None


    print("\n--- TPE Optimization Finished ---")
    print(f"Detailed log of all evaluations saved to: {os.path.abspath(detailed_log_csv_path)}")

    summary_log_path = os.path.join(TEMP_DIR, "block_summary_log.json") # Renamed
    try:
        with open(summary_log_path, 'w') as f:
            json.dump(block_summary_log, f, indent=2)
        print(f"Block summary log saved to: {os.path.abspath(summary_log_path)}")
    except Exception as e:
        print(f"Error saving block summary log: {e}")

    print(f"\nHall of Fame (Pareto Front) has {len(hof)} individuals from all evaluations.")
    for i, best_ind_hof in enumerate(hof): # Renamed best_ind to avoid conflict
        hof_ind_genes_str = str(list(best_ind_hof))
        # Match HOF individuals to their detailed log entries
        logged_entry_for_hof = next((item for item in all_individuals_detailed_log_in_memory
                                     if item["genes"] == hof_ind_genes_str and
                                        item.get("surface_area") is not None and best_ind_hof.fitness.values[0] is not None and
                                        abs(item.get("surface_area") - best_ind_hof.fitness.values[0]) < 1e-9 and
                                        item.get("cupped_water_vol") is not None and best_ind_hof.fitness.values[1] is not None and
                                        abs(item.get("cupped_water_vol") - best_ind_hof.fitness.values[1]) < 1e-9
                                     ), None)
        print(f"HOF Individual {i}: Fitness = {best_ind_hof.fitness.values}")
        if logged_entry_for_hof:
            print(f"  Individual ID: {logged_entry_for_hof['individual_id']}, From Block: {logged_entry_for_hof['generation']}") # 'generation' field now means block_num
            print(f"  Params: {logged_entry_for_hof['params_file']}, Blend Output (likely deleted): {logged_entry_for_hof['blend_file']}")
            print(f"  Input Blend Used: {logged_entry_for_hof['parent_blend_file']} (from Parent ID: {logged_entry_for_hof['parent_individual_id']})")
        else:
            print(f"  (Log entry for HOF ind {i} not precisely matched). Genes: {hof_ind_genes_str[:50]}...")

    print(f"\nAll generated parameter files and logs are in: '{os.path.abspath(TEMP_DIR)}'")
    print("Most .blend files should have been deleted after analysis.")
    return hof, all_individuals_detailed_log_in_memory # Removed pop as it's not maintained in the old sense

if __name__ == "__main__":
    script_start_time = time.perf_counter()

    final_hof, final_in_memory_log = main() # Removed final_pop
    if final_in_memory_log:
        print(f"\nTotal {len(final_in_memory_log)} individual evaluations recorded in memory and CSV.")

    script_end_time = time.perf_counter()
    total_duration_seconds = script_end_time - script_start_time
    
    print(f"Script execution finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total script execution time: {format_duration(total_duration_seconds)}")
