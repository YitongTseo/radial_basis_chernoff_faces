import json
import os
import csv
import random
import shutil
import uuid
import functools
from deap import base, creator, tools, algorithms
from utils import generate_face_mesh, run_blender_analysis # Assuming these are in utils.py
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np # For hyperopt's rstate

# --- Configuration ---
BLENDER_EXECUTABLE_PATH = '/Applications/Blender.app/Contents/MacOS/Blender'
GENERATION_HELPER_SCRIPT_PATH = 'blender_rbf_script.py'
ANALYSIS_SCRIPT_PATH = 'water_trapping.py'
BASE_INPUT_BLEND_FILE = 'face_landmark_points.blend' # Used for Gen 0
OBJECT_NAME_TO_ANALYZE = "Yitong_Face"
ANALYSIS_VOXEL_SIZE = 0.1

POPULATION_SIZE = 1
N_GENERATIONS = 2 # Keep small for testing new features
MAX_EVALS_TPE = 3   # Max evaluations for each TPE optimization run (per child mutation)

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

# --- DEAP Setup ---
creator.create("FitnessMultiMax", base.Fitness, weights=FITNESS_WEIGHTS)
creator.create("Individual", list, fitness=creator.FitnessMultiMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -1.0, 1.0)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, TOTAL_NUM_GENES)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# --- Global cache for memoizing regenerated files during a run ---
# Cleared at the start of ensure_blend_file_exists if it's the first call in a chain,
# or manage more globally if needed. For simplicity, let's make it global for the run.
# This helps avoid regenerating the same file multiple times if many children share a deleted parent.
regeneration_cache = {}


def individual_to_params_json(individual_genes, current_generation_num):
    # Determine the correct column for mutational load based on generation
    if current_generation_num == 0:
        col_of_interest = "Reference->03.2020 Mutational Load"
    elif current_generation_num == 1:
        col_of_interest = "03.2020->09.2023 Mutational Load"
    elif current_generation_num == 2: # Assuming N_GENERATIONS can go up to 2 for this logic
        col_of_interest = "09.2023->01.2025 Mutational Load"
    else: # Fallback or error for generations beyond what's defined
        # Defaulting to last known column or raise error
        print(f"Warning: Generation number {current_generation_num} out of defined range for mutational load. Using last known.")
        col_of_interest = "09.2023->01.2025 Mutational Load"
        # raise Exception(f'Generation {current_generation_num} not mapped to a mutational load column.')


    params_dict = {"default": DEFAULT_PARAMS_SECTION}
    gene_idx = 0
    for landmark_name in LANDMARK_NAMES:
        direction = [
            max(-1.0, min(1.0, individual_genes[gene_idx])),
            max(-1.0, min(1.0, individual_genes[gene_idx + 1])),
            max(-1.0, min(1.0, individual_genes[gene_idx + 2])),
        ]
        try:
            landmark_ranking = LANDMARK_TO_GENES_RANKING.index(landmark_name)
            gene_of_interest = GENOME_MUTATIONS_DF.iloc[landmark_ranking]
            generational_mutation_load = gene_of_interest[col_of_interest]
        except (ValueError, IndexError) as e:
            print(f"Warning: Landmark {landmark_name} not found in ranking or issue with GENOME_MUTATIONS_DF. Using default magnitude. Error: {e}")
            generational_mutation_load = 0.0 # Default or fallback

        params_dict[landmark_name] = {
            "direction": direction,
            "magnitude": GENE_MUTATION_MULTIPLIER * generational_mutation_load + GENE_MUTATION_BASE_RATE,
        }
        gene_idx += 3
    return params_dict

# --- Blend File Regeneration Helper ---
def ensure_blend_file_exists(target_blend_filename,
                             all_individuals_log, # Pass the main log
                             temp_dir_path,
                             blender_exec, gen_script, base_blend): # Pass necessary paths/configs
    """
    Ensures a specific .blend file exists, regenerating it if it's missing.
    Uses memoization to avoid redundant regenerations.
    """
    global regeneration_cache

    if target_blend_filename == base_blend:
        return base_blend # Base file is assumed to always exist and not be in TEMP_DIR

    target_blend_full_path = os.path.join(temp_dir_path, target_blend_filename)

    if target_blend_filename in regeneration_cache and os.path.exists(target_blend_full_path):
        return target_blend_full_path # Already handled and exists

    if os.path.exists(target_blend_full_path):
        regeneration_cache[target_blend_filename] = target_blend_full_path # Cache existing file
        return target_blend_full_path

    print(f"    File {target_blend_filename} not found at {target_blend_full_path}. Attempting regeneration.")

    # Find the log entry for the individual that produced this blend file
    log_entry_for_target_blend = None
    for entry in reversed(all_individuals_log): # Search all logs
        if entry["blend_file"] == target_blend_filename:
            log_entry_for_target_blend = entry
            break
    
    if not log_entry_for_target_blend:
        raise FileNotFoundError(f"Log entry for {target_blend_filename} not found. Cannot regenerate.")

    parent_blend_ref_for_target = log_entry_for_target_blend["parent_blend_file"]
    params_file_for_target = os.path.join(temp_dir_path, log_entry_for_target_blend["params_file"])

    if not os.path.exists(params_file_for_target):
        raise FileNotFoundError(f"Params file {log_entry_for_target_blend['params_file']} (path: {params_file_for_target}) needed for {target_blend_filename} not found.")

    # Recursively ensure the parent's blend file exists
    print(f"    Recursively ensuring parent blend '{parent_blend_ref_for_target}' exists for regenerating '{target_blend_filename}'.")
    input_blend_for_regeneration = ensure_blend_file_exists(
        parent_blend_ref_for_target,
        all_individuals_log,
        temp_dir_path,
        blender_exec, gen_script, base_blend
    )

    if not os.path.exists(input_blend_for_regeneration) and input_blend_for_regeneration != base_blend:
        raise FileNotFoundError(f"Input blend '{input_blend_for_regeneration}' for regenerating {target_blend_filename} ultimately not found.")

    print(f"      Regenerating {target_blend_filename} using input {os.path.basename(input_blend_for_regeneration)} and params {log_entry_for_target_blend['params_file']}")
    
    # Mark that we are attempting to regenerate, remove from cache if it was marked as failed/non-existent before
    if target_blend_filename in regeneration_cache:
        del regeneration_cache[target_blend_filename]

    success_regen = generate_face_mesh(
        blender_executable_path=blender_exec,
        analyzer_script_path=gen_script,
        blend_file_to_open=input_blend_for_regeneration,
        dither_config=params_file_for_target,
        output_file=target_blend_full_path, # Regenerate to its original expected path
    )

    if not success_regen or not os.path.exists(target_blend_full_path):
        raise Exception(f"Regeneration of {target_blend_filename} failed or produced no output at {target_blend_full_path}.")
    
    print(f"    Successfully regenerated {target_blend_filename} at {target_blend_full_path}.")
    regeneration_cache[target_blend_filename] = target_blend_full_path # Cache path of regenerated file
    return target_blend_full_path


def evaluate_and_log_individual(
    individual_genes, current_generation_num,
    parent_individual_id_for_log, input_blend_for_this_eval,
    detailed_log_list_in_memory, csv_writer_object, csv_file_handle,
    # Pass necessary paths/configs for utility functions if they are not global
    blender_exec, gen_script_path, analysis_script_path, base_blend_name, obj_name, voxel_size, temp_dir_path
):
    individual_id = str(uuid.uuid4())
    param_filename = f"params_gen{current_generation_num}_{individual_id}.json"
    blend_filename = f"output_gen{current_generation_num}_{individual_id}.blend"
    
    params_json_path = os.path.join(temp_dir_path, param_filename)
    generated_blend_path = os.path.join(temp_dir_path, blend_filename)

    params_data = individual_to_params_json(individual_genes, current_generation_num)
    with open(params_json_path, 'w') as f:
        json.dump(params_data, f, indent=2)

    parent_blend_to_log_in_csv = os.path.basename(input_blend_for_this_eval) if input_blend_for_this_eval != base_blend_name else base_blend_name

    log_entry = {
        "generation": current_generation_num,
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

    current_fitness = tuple(-float('inf') for _ in FITNESS_WEIGHTS) # Default for failure

    if not success_generation or not os.path.exists(generated_blend_path):
        if success_generation and log_entry["generation_status"] == "pending": # Generated but no output file
             log_entry["generation_status"] = "failed_no_output"
        elif log_entry["generation_status"] == "pending": # Did not even succeed generation
            log_entry["generation_status"] = "failed_generation_error"

    else: # Generation successful, .blend file exists
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
            log_entry.update({
                "surface_area": sa, "cupped_water_vol": cwv,
                'nx': analysis_results.get('nx'), 'ny': analysis_results.get('ny'),
                'nz': analysis_results.get('nz'), 'trapped_water_vol': analysis_results.get('trapped_water_vol'),
                'solid_volume': analysis_results.get('solid_volume')
            })
            current_fitness = (sa, cwv)

        # Delete .blend file after analysis if generation and analysis were successful
        if log_entry["generation_status"] == "success" and log_entry["analysis_status"] == "success" and os.path.exists(generated_blend_path):
            print(f"    Deleting analyzed blend file: {os.path.basename(generated_blend_path)}")
            try:
                os.remove(generated_blend_path)
            except OSError as e:
                print(f"    Warning: Could not delete {generated_blend_path}: {e}")
        elif os.path.exists(generated_blend_path): # if analysis failed but blend was created
             print(f"    Keeping blend file due to non-successful analysis: {os.path.basename(generated_blend_path)}")


    detailed_log_list_in_memory.append(log_entry)
    csv_writer_object.writerow(log_entry)
    csv_file_handle.flush()
    
    return current_fitness

# No longer using toolbox.mutate directly for primary mutation if Hyperopt is always used.
# toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=MUTATION_SIGMA, indpb=MUTATION_INDPB) # Keep for fallback/other uses
toolbox.register("select", tools.selNSGA2)

def main():
    global regeneration_cache # Allow main to clear/manage cache if needed
    regeneration_cache.clear()

    print("Evolutionary Algorithm: Accumulative Mutations, Detailed Logging, Hyperopt Mutation, Blend File Management")
    print(f"Output and logs will be in: {os.path.abspath(TEMP_DIR)}")
    print(f"Using Hyperopt for mutation with MAX_EVALS_TPE={MAX_EVALS_TPE}")
    print("---")

    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda x: tuple(round(sum(col)/len(col),3) for col in zip(*x)) if x and all(isinstance(i,tuple) and len(i)==len(FITNESS_WEIGHTS) for i in x) else tuple(None for _ in FITNESS_WEIGHTS))
    stats.register("min", lambda x: tuple(round(min(col),3) for col in zip(*x)) if x and all(isinstance(i,tuple) and len(i)==len(FITNESS_WEIGHTS) for i in x) else tuple(None for _ in FITNESS_WEIGHTS))
    stats.register("max", lambda x: tuple(round(max(col),3) for col in zip(*x)) if x and all(isinstance(i,tuple) and len(i)==len(FITNESS_WEIGHTS) for i in x) else tuple(None for _ in FITNESS_WEIGHTS))
    
    generational_summary_log = []
    all_individuals_detailed_log_in_memory = []

    detailed_log_csv_path = os.path.join(TEMP_DIR, "all_individuals_evaluation_log.csv")
    csv_fieldnames = [
        "generation", "individual_id", "parent_individual_id", "parent_blend_file",
        "params_file", "blend_file", "nx", "ny", "nz", "solid_volume",
        "trapped_water_vol", "surface_area", "cupped_water_vol",
        "generation_status", "analysis_status", "genes",
    ]

    with open(detailed_log_csv_path, 'a+', newline='') as csvfile_handle:
        csv_writer = csv.DictWriter(csvfile_handle, fieldnames=csv_fieldnames)
        if csvfile_handle.tell() == 0: # File is empty, write header
            csv_writer.writeheader()

        # Generation 0
        print("Evaluating initial population (Generation 0)...")
        for i, ind in enumerate(pop):
            if not ind.fitness.valid:
                print(f"  Gen 0, Eval Ind {i+1}/{POPULATION_SIZE}")
                ind.fitness.values = evaluate_and_log_individual(
                    individual_genes=ind, current_generation_num=0,
                    parent_individual_id_for_log="INITIAL",
                    input_blend_for_this_eval=BASE_INPUT_BLEND_FILE,
                    detailed_log_list_in_memory=all_individuals_detailed_log_in_memory,
                    csv_writer_object=csv_writer, csv_file_handle=csvfile_handle,
                    blender_exec=BLENDER_EXECUTABLE_PATH, gen_script_path=GENERATION_HELPER_SCRIPT_PATH,
                    analysis_script_path=ANALYSIS_SCRIPT_PATH, base_blend_name=BASE_INPUT_BLEND_FILE,
                    obj_name=OBJECT_NAME_TO_ANALYZE, voxel_size=ANALYSIS_VOXEL_SIZE, temp_dir_path=TEMP_DIR
                )
        
        hof.update(pop)
        record = stats.compile(pop)
        generational_summary_log.append({"gen": 0, **record})
        print(f"Generation 0 Summary: {record}")

        # Subsequent Generations
        for gen in range(1, N_GENERATIONS + 1):
            print(f"\n--- Generation {gen}/{N_GENERATIONS} ---")
            regeneration_cache.clear() # Clear cache at start of new generation to manage memory / stale entries

            selected_parents = toolbox.select(pop, POPULATION_SIZE)
            offspring_population = []

            print(f"  Generating and evaluating {POPULATION_SIZE} offspring for generation {gen}...")
            for i in range(POPULATION_SIZE):
                parent_individual = selected_parents[i % len(selected_parents)] # Cycle through parents if fewer than POP_SIZE
                
                # Determine parent log entry and input blend file for child
                parent_log_entry = next((entry for entry in reversed(all_individuals_detailed_log_in_memory)
                                         if entry["genes"] == str(list(parent_individual)) and
                                            entry["generation"] == gen - 1 and
                                            entry["blend_file"] is not None and # Parent must have a blend file associated
                                            entry["generation_status"] == "success" # Parent must have been successful
                                         ), None)

                parent_id_for_child = "UNKNOWN_PARENT"
                input_blend_for_child_eval = BASE_INPUT_BLEND_FILE # Default

                if parent_log_entry:
                    parent_id_for_child = parent_log_entry["individual_id"]
                    parent_output_blend_filename = parent_log_entry["blend_file"]
                    try:
                        print(f"    Child {i+1}, ensuring parent blend '{parent_output_blend_filename}' (Parent ID: {parent_id_for_child}) exists...")
                        input_blend_for_child_eval = ensure_blend_file_exists(
                            parent_output_blend_filename,
                            all_individuals_detailed_log_in_memory, TEMP_DIR,
                            BLENDER_EXECUTABLE_PATH, GENERATION_HELPER_SCRIPT_PATH, BASE_INPUT_BLEND_FILE
                        )
                        print(f"    Using input blend for child {i+1}: {os.path.basename(input_blend_for_child_eval)}")
                    except Exception as e:
                        print(f"      CRITICAL WARNING: Failed to ensure/regenerate parent blend {parent_output_blend_filename}. Error: {e}. Reverting to base file for child {i+1}.")
                        input_blend_for_child_eval = BASE_INPUT_BLEND_FILE
                else:
                    print(f"    WARNING: Successful parent log entry not found for child {i+1} (Parent genes hash: {hash(str(list(parent_individual)))}). Reverting to base file.")

                # Clone parent genes for the child
                child_genes_list = toolbox.clone(parent_individual) # This is a list of genes

                print(f"    Child {i+1}: Applying Hyperopt TPE mutation (max {MAX_EVALS_TPE} evals)...")
                
                # Define objective function for Hyperopt (evaluates genes, returns scalar loss)
                # This objective should NOT log to the main CSV, but manage its own temp files.
                def fmin_objective(gene_params_from_hyperopt):
                    # gene_params_from_hyperopt is a list of floats
                    temp_eval_id = str(uuid.uuid4())
                    temp_params_path = os.path.join(TEMP_DIR, f"hyperopt_trial_params_{temp_eval_id}.json")
                    temp_blend_path = os.path.join(TEMP_DIR, f"hyperopt_trial_blend_{temp_eval_id}.blend")

                    # Use the child's generation number for params
                    trial_params_data = individual_to_params_json(gene_params_from_hyperopt, gen)
                    with open(temp_params_path, 'w') as f: json.dump(trial_params_data, f)

                    trial_fitness_sa, trial_fitness_cwv = -float('inf'), -float('inf')
                    
                    # input_blend_for_child_eval is captured from the outer scope
                    trial_success_gen = False
                    try:
                        trial_success_gen = generate_face_mesh(
                            blender_executable_path=BLENDER_EXECUTABLE_PATH,
                            analyzer_script_path=GENERATION_HELPER_SCRIPT_PATH,
                            blend_file_to_open=input_blend_for_child_eval, # Use the resolved parent blend
                            dither_config=temp_params_path,
                            output_file=temp_blend_path)
                    except Exception as e_gen:
                        # print(f"      Hyperopt trial gen error: {e_gen}") # Optional: verbose
                        pass 

                    if trial_success_gen and os.path.exists(temp_blend_path):
                        try:
                            trial_analysis_results = run_blender_analysis(
                                blender_executable_path=BLENDER_EXECUTABLE_PATH,
                                analyzer_script_path=ANALYSIS_SCRIPT_PATH,
                                blend_file_to_open=temp_blend_path,
                                object_name_in_blend=OBJECT_NAME_TO_ANALYZE,
                                voxel_s=ANALYSIS_VOXEL_SIZE, create_debug=False, verbose_blender_output=False)
                            if trial_analysis_results:
                                trial_fitness_sa = trial_analysis_results.get('surface_area', 0)
                                trial_fitness_cwv = trial_analysis_results.get('cupped_water_vol', 0)
                        except Exception as e_ana:
                            # print(f"      Hyperopt trial analysis error: {e_ana}") # Optional: verbose
                            pass
                    
                    # Cleanup temporary files for this hyperopt trial
                    if os.path.exists(temp_params_path): os.remove(temp_params_path)
                    if os.path.exists(temp_blend_path): os.remove(temp_blend_path)
                    
                    # Hyperopt minimizes, so return negative of weighted fitness
                    loss = -(FITNESS_WEIGHTS[0] * trial_fitness_sa + FITNESS_WEIGHTS[1] * trial_fitness_cwv)
                    if loss == 0 and (trial_fitness_sa == -float('inf') or trial_fitness_cwv == -float('inf')): # Ensure really bad fitness is high loss
                        loss = float('inf')

                    return {'loss': loss, 'status': STATUS_OK, 'genes': list(gene_params_from_hyperopt)}

                # Define search space for hyperopt: list of hp.uniform for each gene
                # Genes are between -1 and 1.
                hyperopt_space = [hp.uniform(f'g{k}', -1.0, 1.0) for k in range(TOTAL_NUM_GENES)]
                
                trials = Trials()
                try:
                    best_run = fmin(
                        fn=fmin_objective,
                        space=hyperopt_space,
                        algo=tpe.suggest,
                        max_evals=MAX_EVALS_TPE,
                        trials=trials,
                        rstate=np.random.default_rng(random.randint(0, 100000)), # Seed for TPE
                        verbose=1 # 0 for silent, 1 for TPE progress (can be noisy)
                    )
                    # Update child_genes_list with the best genes found by TPE
                    # best_run is a dict: {'g0': val, 'g1': val, ...}
                    child_genes_list[:] = [best_run[f'g{k}'] for k in range(TOTAL_NUM_GENES)]
                    print(f"    Child {i+1}: Hyperopt mutation applied. Best loss: {trials.best_trial['result']['loss']:.4f}")
                except Exception as e_hyperopt:
                    print(f"    Child {i+1}: Hyperopt mutation failed: {e_hyperopt}. Using cloned parent genes.")
                    # child_genes_list remains a clone of the parent

                # Create the DEAP individual object
                child_individual = creator.Individual(child_genes_list)
                # Fitness will be evaluated by the main call below (del child_genes.fitness.values is implicit for new Individual)

                print(f"    Child {i+1}, Final Eval - Parent ID: {parent_id_for_child}, Input: {os.path.basename(input_blend_for_child_eval)}")
                child_individual.fitness.values = evaluate_and_log_individual(
                    individual_genes=child_individual, current_generation_num=gen,
                    parent_individual_id_for_log=parent_id_for_child,
                    input_blend_for_this_eval=input_blend_for_child_eval,
                    detailed_log_list_in_memory=all_individuals_detailed_log_in_memory,
                    csv_writer_object=csv_writer, csv_file_handle=csvfile_handle,
                    blender_exec=BLENDER_EXECUTABLE_PATH, gen_script_path=GENERATION_HELPER_SCRIPT_PATH,
                    analysis_script_path=ANALYSIS_SCRIPT_PATH, base_blend_name=BASE_INPUT_BLEND_FILE,
                    obj_name=OBJECT_NAME_TO_ANALYZE, voxel_size=ANALYSIS_VOXEL_SIZE, temp_dir_path=TEMP_DIR
                )
                offspring_population.append(child_individual)
            
            pop[:] = offspring_population
            hof.update(pop)
            record = stats.compile(pop)
            generational_summary_log.append({"gen": gen, **record})
            print(f"Generation {gen} Summary: {record}")

    print("\n--- Evolution Finished ---")
    print(f"Detailed incremental log saved to: {os.path.abspath(detailed_log_csv_path)}")

    summary_log_path = os.path.join(TEMP_DIR, "generational_summary_log.json")
    try:
        with open(summary_log_path, 'w') as f:
            json.dump(generational_summary_log, f, indent=2)
        print(f"Generational summary log saved to: {os.path.abspath(summary_log_path)}")
    except Exception as e:
        print(f"Error saving summary log: {e}")

    print(f"\nHall of Fame (Pareto Front) has {len(hof)} individuals.")
    for i, best_ind in enumerate(hof):
        hof_ind_genes_str = str(list(best_ind)) 
        logged_entry_for_hof = next((item for item in all_individuals_detailed_log_in_memory 
                                     if item["genes"] == hof_ind_genes_str and 
                                        item.get("surface_area") is not None and best_ind.fitness.values[0] is not None and
                                        abs(item.get("surface_area") - best_ind.fitness.values[0]) < 1e-9 and 
                                        item.get("cupped_water_vol") is not None and best_ind.fitness.values[1] is not None and
                                        abs(item.get("cupped_water_vol") - best_ind.fitness.values[1]) < 1e-9), None)
        
        print(f"HOF Individual {i}: Fitness = {best_ind.fitness.values}")
        if logged_entry_for_hof:
            print(f"  Individual ID: {logged_entry_for_hof['individual_id']}, Gen: {logged_entry_for_hof['generation']}")
            print(f"  Params: {logged_entry_for_hof['params_file']}, Blend Output (likely deleted): {logged_entry_for_hof['blend_file']}")
            print(f"  Parent ID: {logged_entry_for_hof['parent_individual_id']}, Parent Blend Input: {logged_entry_for_hof['parent_blend_file']}")
            # Finding grandparent info if needed:
            parent_id = logged_entry_for_hof['parent_individual_id']
            if parent_id != "INITIAL" and parent_id != "UNKNOWN_PARENT":
                grandparent_info_entry = next((entry for entry in all_individuals_detailed_log_in_memory if entry["individual_id"] == parent_id), None)
                if grandparent_info_entry:
                     print(f"  Grandparent's Blend (Input to Parent): {grandparent_info_entry['parent_blend_file']}")
        else:
            print(f"  (Log entry for HOF individual {i} not precisely matched by genes & fitness). Genes: {hof_ind_genes_str[:50]}...")


    print(f"\nAll generated parameter files and logs are in: '{os.path.abspath(TEMP_DIR)}'")
    print("Most .blend files should have been deleted after analysis to save space.")
    return pop, hof, all_individuals_detailed_log_in_memory

if __name__ == "__main__":
    final_pop, final_hof, in_memory_detailed_log = main()
    if in_memory_detailed_log:
        print(f"\nTotal {len(in_memory_detailed_log)} individual evaluations recorded in memory and CSV.")
