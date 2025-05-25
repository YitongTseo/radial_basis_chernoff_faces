import json
import os
import csv # NEW: For incremental CSV writing
import random
import shutil
import uuid
import functools # For functools.partial
from deap import base, creator, tools, algorithms
from utils import generate_face_mesh, run_blender_analysis # Assuming these are in utils.py


# --- Configuration ---
BLENDER_EXECUTABLE_PATH = '/Applications/Blender.app/Contents/MacOS/Blender'
GENERATION_HELPER_SCRIPT_PATH = 'blender_rbf_script.py'
ANALYSIS_SCRIPT_PATH = 'water_trapping.py'
BASE_INPUT_BLEND_FILE = 'face_landmark_points.blend'
OBJECT_NAME_TO_ANALYZE = "Yitong_Face"
ANALYSIS_VOXEL_SIZE = 0.05

POPULATION_SIZE = 3  # Number of individuals in the population
N_GENERATIONS = 2    # Number of generations to run
CXPB = 0.0            # Crossover probability (not used as per request)
MUTPB = 1.0           # Mutation probability for an individual
MUTATION_SIGMA = 0.15 # Standard deviation for Gaussian mutation of genes
MUTATION_INDPB = 0.1  # Independent probability for each gene to be mutated
N_ELITES = 2          # Number of best individuals to carry over to the next generation unchanged
FIXED_MAGNITUDE = 0.5
FITNESS_WEIGHTS = (1.0, 0.2) # (Surface Area, Cupped Water Vol)

# --- Landmark Definition ---
try:
    with open('example_params.json', 'r') as f:
        example_params_data = json.load(f)
except FileNotFoundError:
    print("Error: 'example_params.json' not found. Please ensure it's in the correct path.")
    exit(1)

LANDMARK_NAMES = [key for key in example_params_data.keys() if key != "default"]
DEFAULT_PARAMS_SECTION = example_params_data.get("default", {"direction": [0, 0, 0], "magnitude": 0.0})
NUM_GENES_PER_LANDMARK = 3
TOTAL_NUM_GENES = len(LANDMARK_NAMES) * NUM_GENES_PER_LANDMARK
print(f'We have this many genes: {TOTAL_NUM_GENES}')

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

def individual_to_params_json(individual_genes):
    params_dict = {"default": DEFAULT_PARAMS_SECTION}
    gene_idx = 0
    for landmark_name in LANDMARK_NAMES:
        direction = [
            max(-1.0, min(1.0, individual_genes[gene_idx])),
            max(-1.0, min(1.0, individual_genes[gene_idx + 1])),
            max(-1.0, min(1.0, individual_genes[gene_idx + 2])),
        ]
        params_dict[landmark_name] = {
            "direction": direction,
            "magnitude": FIXED_MAGNITUDE
        }
        gene_idx += 3
    return params_dict

# MODIFIED: Evaluation function that logs data incrementally to CSV
def evaluate_and_log_individual(individual_genes, current_generation_num, 
                                detailed_log_list_in_memory, csv_writer_object, csv_file_handle): # Added csv_writer & file_handle
    individual_id = str(uuid.uuid4())
    param_filename = f"params_gen{current_generation_num}_{individual_id}.json"
    blend_filename = f"output_gen{current_generation_num}_{individual_id}.blend"
    
    params_json_path = os.path.join(TEMP_DIR, param_filename)
    generated_blend_path = os.path.join(TEMP_DIR, blend_filename)

    params_data = individual_to_params_json(individual_genes)
    with open(params_json_path, 'w') as f:
        json.dump(params_data, f, indent=2)

    log_entry = {
        "generation": current_generation_num,
        "individual_id": individual_id,
        "params_file": param_filename,
        "blend_file": None,
        "nx": None,
        "ny": None,
        "nz": None,
        "solid_volume": None,
        "trapped_water_vol": None,
        "surface_area": None,
        "cupped_water_vol": None,
        "generation_status": "pending",
        "analysis_status": "pending"
    }

    success_generation = False
    try:
        success_generation = generate_face_mesh(
            blender_executable_path=BLENDER_EXECUTABLE_PATH,
            analyzer_script_path=GENERATION_HELPER_SCRIPT_PATH,
            blend_file_to_open=BASE_INPUT_BLEND_FILE,
            dither_config=params_json_path,
            output_file=generated_blend_path,
        )
    except Exception as e:
        print(f"  Exception during generate_face_mesh for {individual_id}: {e}")
        log_entry["generation_status"] = f"error: {e}"

    current_fitness = tuple(-float('inf') for _ in FITNESS_WEIGHTS)

    if not success_generation or not os.path.exists(generated_blend_path):
        if success_generation and "generation_status" not in log_entry: # Error was not already set
             log_entry["generation_status"] = "failed_no_output"
        # Fall-through to logging and returning bad fitness
    else:
        log_entry["generation_status"] = "success"
        log_entry["blend_file"] = blend_filename

        analysis_results = None
        try:
            analysis_results = run_blender_analysis(
                blender_executable_path=BLENDER_EXECUTABLE_PATH,
                analyzer_script_path=ANALYSIS_SCRIPT_PATH,
                blend_file_to_open=generated_blend_path,
                object_name_in_blend=OBJECT_NAME_TO_ANALYZE,
                voxel_s=ANALYSIS_VOXEL_SIZE,
                create_debug=False,
                verbose_blender_output=False
            )
        except Exception as e:
            print(f"  Exception during run_blender_analysis for {individual_id}: {e}")
            log_entry["analysis_status"] = f"error: {e}"
        
        if analysis_results is None:
            if "analysis_status" not in log_entry: # Error was not already set
                log_entry["analysis_status"] = "failed_no_results"
            # Fall-through to logging and returning bad fitness (already set)
        else:
            log_entry["analysis_status"] = "success"
            surface_area = analysis_results.get('surface_area', 0)
            cupped_water_vol = analysis_results.get('cupped_water_vol', 0)
            log_entry["surface_area"] = surface_area
            log_entry["cupped_water_vol"] = cupped_water_vol
            log_entry['nx'] = analysis_results.get('nx', 0)
            log_entry['ny'] = analysis_results.get('ny', 0)
            log_entry['nz'] = analysis_results.get('nz', 0)
            log_entry['trapped_water_vol'] = analysis_results.get('trapped_water_vol', 0)
            log_entry['solid_volume'] = analysis_results.get('solid_volume', 0)
            current_fitness = (surface_area, cupped_water_vol)

    detailed_log_list_in_memory.append(log_entry) # Append to in-memory list
    csv_writer_object.writerow(log_entry)      # Write to CSV file
    csv_file_handle.flush()                    # Ensure it's written to disk
    
    return current_fitness

toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=MUTATION_SIGMA, indpb=MUTATION_INDPB)
toolbox.register("select", tools.selNSGA2)

# --- Main Evolutionary Loop ---
def main():
    print("Evolutionary Algorithm: Detailed Incremental Logging to CSV")
    print(f"Output and logs will be in: {os.path.abspath(TEMP_DIR)}")
    print("---")

    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values) # For generational summary
    stats.register("avg", lambda x: tuple(round(sum(col) / len(col), 3) for col in zip(*x)) if x and all(isinstance(i,tuple) for i in x) else (None,None))
    stats.register("min", lambda x: tuple(round(min(col), 3) for col in zip(*x)) if x and all(isinstance(i,tuple) for i in x) else (None,None))
    stats.register("max", lambda x: tuple(round(max(col), 3) for col in zip(*x)) if x and all(isinstance(i,tuple) for i in x) else (None,None))
    
    generational_summary_log = []
    all_individuals_detailed_log_in_memory = [] # Still keep this for HOF and return

    # **NEW: Setup CSV file and writer**
    detailed_log_csv_path = os.path.join(TEMP_DIR, "all_individuals_evaluation_log.csv")
    csv_fieldnames = [
        "generation",
        "individual_id",
        "params_file",
        "blend_file",
        "nx",
        "ny",
        "nz",
        "solid_volume",
        "trapped_water_vol",
        "surface_area",
        "cupped_water_vol",
        "generation_status",
        "analysis_status",
    ]

    with open(detailed_log_csv_path, 'a+', newline='') as csvfile_handle: # Open for append, create if not exists
        csv_writer = csv.DictWriter(csvfile_handle, fieldnames=csv_fieldnames)
        csvfile_handle.seek(0) # Go to the start of the file to check if empty
        is_empty = not csvfile_handle.read(1) # Read one char to see if file is empty
        if is_empty:
            csvfile_handle.seek(0) # Go back to write header at the start
            csv_writer.writeheader()
        # csvfile_handle will be flushed by evaluate_and_log_individual after each row.

        # Evaluate the initial population (Generation 0)
        print("Evaluating initial population (Generation 0)...")
        eval_func_gen0 = functools.partial(evaluate_and_log_individual,
                                           current_generation_num=0,
                                           detailed_log_list_in_memory=all_individuals_detailed_log_in_memory,
                                           csv_writer_object=csv_writer,
                                           csv_file_handle=csvfile_handle) # Pass writer and file handle
        toolbox.register("evaluate", eval_func_gen0)

        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        hof.update(pop)
        record = stats.compile(pop)
        generational_summary_log.append({"gen": 0, **record})
        print(f"Generation 0 Summary: {record}")
        print(f"Path to detailed CSV log: {os.path.abspath(detailed_log_csv_path)}")

        # Evolution
        for gen in range(1, N_GENERATIONS + 1):
            print(f"\n--- Generation {gen}/{N_GENERATIONS} ---")
            offspring = toolbox.select(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]
            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            print(f"Evaluating {len(invalid_ind)} new individuals for generation {gen}...")
            eval_func_current_gen = functools.partial(evaluate_and_log_individual,
                                                      current_generation_num=gen,
                                                      detailed_log_list_in_memory=all_individuals_detailed_log_in_memory,
                                                      csv_writer_object=csv_writer,
                                                      csv_file_handle=csvfile_handle) # Pass writer and file handle
            toolbox.register("evaluate", eval_func_current_gen)
            
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            pop[:] = offspring
            hof.update(pop)
            record = stats.compile(pop)
            generational_summary_log.append({"gen": gen, **record})
            print(f"Generation {gen} Summary: {record}")
            print(f"Path to detailed CSV log: {os.path.abspath(detailed_log_csv_path)}")

    # File is automatically closed when 'with' block exits
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
        logged_entry_for_hof = next((item for item in all_individuals_detailed_log_in_memory 
                                     if item["surface_area"] == best_ind.fitness.values[0] and 
                                        item["cupped_water_vol"] == best_ind.fitness.values[1]), None)
        
        print(f"HOF Individual {i}: Fitness = {best_ind.fitness.values}")
        if logged_entry_for_hof:
            print(f"  Params: {logged_entry_for_hof['params_file']}, Blend: {logged_entry_for_hof['blend_file']}")
        else:
            print(f"  Log entry for HOF individual {i} not found in in-memory list (genes might differ slightly due to float precision if re-evaluated).")


    print(f"\nAll generated files and logs are in: '{os.path.abspath(TEMP_DIR)}'")
    return pop, hof, all_individuals_detailed_log_in_memory # Return in-memory log too

if __name__ == "__main__":
    final_pop, final_hof, in_memory_detailed_log = main()
    if in_memory_detailed_log:
        print(f"\nTotal {len(in_memory_detailed_log)} individual evaluations recorded in memory and CSV.")