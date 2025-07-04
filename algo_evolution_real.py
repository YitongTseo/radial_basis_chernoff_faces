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
BASE_INPUT_BLEND_FILE = 'eyes_open_landmark.blend'
OBJECT_NAME_TO_ANALYZE = "eyes_open_mask"
ANALYSIS_VOXEL_SIZE = 0.1

POPULATION_SIZE = 300
N_GENERATIONS = 2
MUTPB = 1.0
MUTATION_SIGMA = 0.15
MUTATION_INDPB = 0.1
FITNESS_WEIGHTS = (1.0, 0.2) # (Surface Area, Cupped Water Vol)
GENE_MUTATION_MULTIPLIER = 30
GENE_MUTATION_BASE_RATE = 0.5

# --- Landmark Definition ---
try:
    with open('example_params_open_eyes.json', 'r') as f:
        example_params_data = json.load(f)
except FileNotFoundError:
    print("Error: 'example_params_open_eyes.json' not found. Please ensure it's in the current directory or update the path.")
    exit(1)

LANDMARK_NAMES = [key for key in example_params_data.keys() if key != "default"]
DEFAULT_PARAMS_SECTION = example_params_data.get("default", {"direction": [0, 0, 0], "magnitude": 0.0})
NUM_GENES_PER_LANDMARK = 3
TOTAL_NUM_GENES = len(LANDMARK_NAMES) * NUM_GENES_PER_LANDMARK
print(f'We have this many genes: {TOTAL_NUM_GENES}')
GENOME_MUTATIONS_DF = pd.read_csv('genomes/most_variable_gene_mutations.csv')
LANDMARK_TO_GENES_RANKING = pd.read_csv('landmark_ranking_open_eyes.csv', comment="#", header=None,)[0].to_list()

# --- Temporary Directory for EA files ---
TEMP_DIR = "INDIVIDUALS_EVOLVING_OPEN_EYED" # Changed by user
if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)
os.makedirs(TEMP_DIR, exist_ok=True)

# --- DEAP Setup ---
creator.create("FitnessMultiMax", base.Fitness, weights=FITNESS_WEIGHTS)
creator.create("Individual", list, fitness=creator.FitnessMultiMax) # Individuals are lists of genes

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -1.0, 1.0)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, TOTAL_NUM_GENES)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def individual_to_params_json(individual_genes, current_generation_num):
    if current_generation_num == 0:
        col_of_interest = "Reference->03.2020 Mutational Load"
    elif current_generation_num == 1:
        col_of_interest = "03.2020->09.2023 Mutational Load"
    elif current_generation_num == 2:
        col_of_interest = "09.2023->01.2025 Mutational Load"
    else:
        raise Exception('we only have 3 generations of changes worth of mutations')

    params_dict = {"default": DEFAULT_PARAMS_SECTION}
    gene_idx = 0
    for landmark_name in LANDMARK_NAMES:
        direction = [
            max(-1.0, min(1.0, individual_genes[gene_idx])),
            max(-1.0, min(1.0, individual_genes[gene_idx + 1])),
            max(-1.0, min(1.0, individual_genes[gene_idx + 2])),
        ]
        landmark_ranking = LANDMARK_TO_GENES_RANKING.index(landmark_name)
        gene_of_interest = GENOME_MUTATIONS_DF.iloc[landmark_ranking]
        generational_mutation_load = gene_of_interest[col_of_interest]
        params_dict[landmark_name] = {
            "direction": direction,
            "magnitude": GENE_MUTATION_MULTIPLIER * generational_mutation_load + GENE_MUTATION_BASE_RATE,
        }
        gene_idx += 3
    return params_dict

def evaluate_and_log_individual(
    individual_genes, current_generation_num,
    parent_individual_id_for_log, input_blend_for_this_eval, # Changed params
    detailed_log_list_in_memory, csv_writer_object, csv_file_handle
):
    individual_id = str(uuid.uuid4()) # Unique ID for this new individual being evaluated
    param_filename = f"params_gen{current_generation_num}_{individual_id}.json"
    blend_filename = f"output_gen{current_generation_num}_{individual_id}.blend" # Output of this eval
    
    params_json_path = os.path.join(TEMP_DIR, param_filename)
    generated_blend_path = os.path.join(TEMP_DIR, blend_filename)

    params_data = individual_to_params_json(individual_genes, current_generation_num)
    with open(params_json_path, 'w') as f:
        json.dump(params_data, f, indent=2)

    # Determine what parent_blend_file_to_log is. It's the input_blend_for_this_eval, but just filename part if it's not BASE
    parent_blend_to_log_in_csv = os.path.basename(input_blend_for_this_eval) if input_blend_for_this_eval != BASE_INPUT_BLEND_FILE else BASE_INPUT_BLEND_FILE

    log_entry = {
        "generation": current_generation_num,
        "individual_id": individual_id,
        "parent_individual_id": parent_individual_id_for_log,
        "parent_blend_file": parent_blend_to_log_in_csv, # Log which blend was its input
        "params_file": param_filename, # Params file for this individual
        "blend_file": None, # Output blend file of this individual
        "nx": None, "ny": None, "nz": None, "solid_volume": None,
        "trapped_water_vol": None, "surface_area": None, "cupped_water_vol": None,
        "generation_status": "pending", "analysis_status": "pending",
        "genes": str(list(individual_genes)), # Genes of this individual
    }

    success_generation = False
    try:
        # Use the passed input_blend_for_this_eval
        success_generation = generate_face_mesh(
            blender_executable_path=BLENDER_EXECUTABLE_PATH,
            analyzer_script_path=GENERATION_HELPER_SCRIPT_PATH,
            blend_file_to_open=input_blend_for_this_eval, # IMPORTANT: Use parent's output or base
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
    else:
        log_entry["generation_status"] = "success"
        log_entry["blend_file"] = blend_filename # This individual's output blend

        analysis_results = None
        try:
            analysis_results = run_blender_analysis(
                blender_executable_path=BLENDER_EXECUTABLE_PATH,
                analyzer_script_path=ANALYSIS_SCRIPT_PATH,
                blend_file_to_open=generated_blend_path, # Analyze this individual's output
                object_name_in_blend=OBJECT_NAME_TO_ANALYZE,
                voxel_s=ANALYSIS_VOXEL_SIZE, create_debug=False, verbose_blender_output=False
            )
        except Exception as e:
            print(f"  Exception during run_blender_analysis for {individual_id}: {e}")
            log_entry["analysis_status"] = f"error: {e}"
        
        # # Now get rid of the .blend file to save space
        # if os.path.exists(generated_blend_path):
        #     os.remove(generated_blend_path)
        
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

    detailed_log_list_in_memory.append(log_entry)
    csv_writer_object.writerow(log_entry)
    csv_file_handle.flush()
    
    return current_fitness

toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=MUTATION_SIGMA, indpb=MUTATION_INDPB)
toolbox.register("select", tools.selNSGA2)

def main():
    print("Evolutionary Algorithm: Accumulative Mutations & Detailed Logging")
    print(f"Output and logs will be in: {os.path.abspath(TEMP_DIR)}")
    print("---")

    pop = toolbox.population(n=POPULATION_SIZE) # Current population
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
        csvfile_handle.seek(0)
        is_empty = not csvfile_handle.read(1)
        if is_empty:
            csvfile_handle.seek(0)
            csv_writer.writeheader()

        # Generation 0
        print("Evaluating initial population (Generation 0)...")
        for i, ind in enumerate(pop): # ind is a list of genes
            if not ind.fitness.valid:
                print(f"  Gen 0, Eval Ind {i+1}/{POPULATION_SIZE}")
                ind.fitness.values = evaluate_and_log_individual(
                    individual_genes=ind, current_generation_num=0,
                    parent_individual_id_for_log="INITIAL",
                    input_blend_for_this_eval=BASE_INPUT_BLEND_FILE,
                    detailed_log_list_in_memory=all_individuals_detailed_log_in_memory,
                    csv_writer_object=csv_writer, csv_file_handle=csvfile_handle
                )
        
        hof.update(pop)
        record = stats.compile(pop)
        generational_summary_log.append({"gen": 0, **record})
        print(f"Generation 0 Summary: {record}")
        print(f"Detailed CSV log: {os.path.abspath(detailed_log_csv_path)}")

        # Subsequent Generations
        for gen in range(1, N_GENERATIONS + 1):
            print(f"\n--- Generation {gen}/{N_GENERATIONS} ---")
            
            # Select parents from the previous generation's population (`pop`)
            selected_parents = toolbox.select(pop, POPULATION_SIZE) # NSGA-II typical use
            
            # Create offspring for the new generation
            offspring_population = []

            print(f"  Generating and evaluating {POPULATION_SIZE} offspring for generation {gen}...")
            for i in range(POPULATION_SIZE):
                parent = selected_parents[i] # The parent chosen by selection
                child_genes = toolbox.clone(parent) # Clone parent's genes for the child
                
                # Mutate the child's genes
                if random.random() < MUTPB: # Assuming MUTPB is overall prob for individual
                    toolbox.mutate(child_genes)
                    del child_genes.fitness.values # Ensure re-evaluation

                # Find the log entry of the ACTUAL parent to get its output blend file
                parent_log_entry = next((entry for entry in reversed(all_individuals_detailed_log_in_memory)
                                         if entry["genes"] == str(list(parent)) and # Match parent's genes
                                            entry["generation"] == gen - 1 and  # From previous gen
                                            entry["blend_file"] is not None and # Successfully created a blend
                                            entry["generation_status"] == "success" and
                                            entry["analysis_status"] == "success" # And was fully successful
                                         ), None)

                parent_id_for_child = "UNKNOWN_PARENT"
                input_blend_for_child = BASE_INPUT_BLEND_FILE # Default fallback

                if parent_log_entry:
                    parent_id_for_child = parent_log_entry["individual_id"]
                    # Construct full path to parent's output blend file
                    parent_output_blend_filename = parent_log_entry["blend_file"]
                    path_to_parent_output_blend = os.path.join(TEMP_DIR, parent_output_blend_filename)
                    if os.path.exists(path_to_parent_output_blend):
                        input_blend_for_child = path_to_parent_output_blend
                    else:
                        # parent_vals = [vals for vals in all_individuals_detailed_log_in_memory if vals['blend_file'] in path_to_parent_output_blend][0]
                        # generate_face_mesh(
                        #     blender_executable_path=BLENDER_EXECUTABLE_PATH,
                        #     analyzer_script_path=GENERATION_HELPER_SCRIPT_PATH,
                        #     blend_file_to_open=parent_vals['parent_blend_file'], 
                        #     dither_config="INDIVIDUALS_EVOLVING/" + parent_vals['params_file'],
                        #     output_file="INDIVIDUALS_EVOLVING/" + parent_vals['blend_file'],
                        # )

                        # /Applications/Blender.app/Contents/MacOS/Blender 
                        # blender_rbf_script.py 
                        # INDIVIDUALS_EVOLVING/output_gen0_84c43045-3394-4696-ba05-8992d6dd22ef.blend 
                        # INDIVIDUALS_EVOLVING/params_gen1_11c1ac1e-f56d-4282-acab-8511c1b09ac2.json 
                        # INDIVIDUALS_EVOLVING/output_gen1_11c1ac1e-f56d-4282-acab-8511c1b09ac2.blend
                        # /Applications/Blender.app/Contents/MacOS/Blender 
                        # blender_rbf_script.py 
                        # face_landmark_points.blend 
                        # INDIVIDUALS_EVOLVING/params_gen0_84c43045-3394-4696-ba05-8992d6dd22ef.json 
                        # INDIVIDUALS_EVOLVING/output_gen0_84c43045-3394-4696-ba05-8992d6dd22ef.blend
                        # /Applications/Blender.app/Contents/MacOS/Blender 
                        # blender_rbf_script.py 
                        # face_landmark_points.blend 
                        # INDIVIDUALS_EVOLVING/params_gen0_84c43045-3394-4696-ba05-8992d6dd22ef.json 
                        # INDIVIDUALS_EVOLVING/output_gen0_84c43045-3394-4696-ba05-8992d6dd22ef.blend

                        # /Applications/Blender.app/Contents/MacOS/Blender 
                        # blender_rbf_script.py 
                        # INDIVIDUALS_EVOLVING/output_gen1_11c1ac1e-f56d-4282-acab-8511c1b09ac2.blend 
                        # INDIVIDUALS_EVOLVING/params_gen2_74923c15-768c-49e5-8290-8b06e45b0e8d.json 
                        # INDIVIDUALS_EVOLVING/output_gen2_74923c15-768c-49e5-8290-8b06e45b0e8d.blend

                        # input_blend_for_child = path_to_parent_output_blend
                        # else:
                        #     import pdb; pdb.set_trace()
                        raise Exception(f"Parent's output blend file '{parent_output_blend_filename}' not found for child {i} of parent {parent_id_for_child}. Reverting to base file.")
                else:
                    print(f"    WARNING: Successful parent log entry not found for child {i} (parent genes: {str(list(parent))[:50]}...). Reverting to base file.")
                
                print(f"    Child {i+1}, Parent ID: {parent_id_for_child}, Using Input Blend: {os.path.basename(input_blend_for_child)}")
                
                # Evaluate the child
                child_genes.fitness.values = evaluate_and_log_individual(
                    individual_genes=child_genes, current_generation_num=gen,
                    parent_individual_id_for_log=parent_id_for_child,
                    input_blend_for_this_eval=input_blend_for_child,
                    detailed_log_list_in_memory=all_individuals_detailed_log_in_memory,
                    csv_writer_object=csv_writer, csv_file_handle=csvfile_handle
                )
                offspring_population.append(child_genes)
            
            # fitness_vals = child_genes.fitness.values
            # fittest_child_index = max(enumerate(fitness_vals), key=lambda x: x[1])[0]
            # fittest_child_genes = offspring_population[fittest_child_index]
            # fittest_child_vals = [vals for vals in all_individuals_detailed_log_in_memory if vals['genes'] == fittest_child_genes][0]
            # # Recreate the fittest child's blender file
            # success_generation = generate_face_mesh(
            #     blender_executable_path=BLENDER_EXECUTABLE_PATH,
            #     analyzer_script_path=GENERATION_HELPER_SCRIPT_PATH,
            #     blend_file_to_open=fittest_child_vals['parent_blend_file'], 
            #     dither_config=fittest_child_vals['params_file'],
            #     output_file=fittest_child_vals['blend_file'],
            # )
            
            pop[:] = offspring_population # New generation replaces old
            hof.update(pop)
            record = stats.compile(pop)
            generational_summary_log.append({"gen": gen, **record})
            print(f"Generation {gen} Summary: {record}")
            print(f"Detailed CSV log: {os.path.abspath(detailed_log_csv_path)}")

    print("\n--- Evolution Finished ---")
    # ... (rest of the summary and HOF printing) ...
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
        # Search HOF entry in the log using its unique ID if possible, or by fitness + genes
        # For simplicity, we'll rely on the fitness and hope it's distinct enough along with genes
        hof_ind_genes_str = str(list(best_ind)) 
        logged_entry_for_hof = next((item for item in all_individuals_detailed_log_in_memory 
                                     if item["genes"] == hof_ind_genes_str and 
                                        item.get("surface_area") == best_ind.fitness.values[0] and 
                                        item.get("cupped_water_vol") == best_ind.fitness.values[1]), None)
        
        print(f"HOF Individual {i}: Fitness = {best_ind.fitness.values}")
        if logged_entry_for_hof:
            print(f"  Individual ID: {logged_entry_for_hof['individual_id']}, Gen: {logged_entry_for_hof['generation']}")
            print(f"  Params: {logged_entry_for_hof['params_file']}, Blend Output: {logged_entry_for_hof['blend_file']}")
            print(f"  Parent ID: {logged_entry_for_hof['parent_individual_id']}, Parent Blend Input: {logged_entry_for_hof['parent_blend_file']}")

            print('grandparent: ', [individual['parent_blend_file'] for individual in all_individuals_detailed_log_in_memory if individual['individual_id'] == logged_entry_for_hof['parent_individual_id']])
        else:
            print(f"  (Log entry for HOF individual {i} not precisely matched in in-memory list by genes & fitness).")


    print(f"\nAll generated files and logs are in: '{os.path.abspath(TEMP_DIR)}'")
    return pop, hof, all_individuals_detailed_log_in_memory

if __name__ == "__main__":
    final_pop, final_hof, in_memory_detailed_log = main()
    if in_memory_detailed_log:
        print(f"\nTotal {len(in_memory_detailed_log)} individual evaluations recorded in memory and CSV.")



