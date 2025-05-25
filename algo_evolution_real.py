import json
import os
import subprocess
import random
import shutil
import uuid
from deap import base, creator, tools, algorithms
from utils import generate_face_mesh, run_blender_analysis

# --- Configuration ---
# Blender and script paths (UPDATE THESE)
BLENDER_EXECUTABLE_PATH = '/Applications/Blender.app/Contents/MacOS/Blender' # Your path
GENERATION_HELPER_SCRIPT_PATH = 'blender_rbf_script.py' # Script for generate_face_mesh
ANALYSIS_SCRIPT_PATH = 'water_trapping.py'           # Script for run_blender_analysis
BASE_INPUT_BLEND_FILE = 'face_landmark_points.blend' # Base .blend for generation

# Object to analyze in Blender
OBJECT_NAME_TO_ANALYZE = "Yitong_Face"
ANALYSIS_VOXEL_SIZE = 0.05

# Evolutionary Algorithm Parameters
POPULATION_SIZE = 30  # Number of individuals in the population
N_GENERATIONS = 3    # Number of generations to run
CXPB = 0.0            # Crossover probability (not used as per request)
MUTPB = 1.0           # Mutation probability for an individual
MUTATION_SIGMA = 0.15 # Standard deviation for Gaussian mutation of genes
MUTATION_INDPB = 0.1  # Independent probability for each gene to be mutated
N_ELITES = 2          # Number of best individuals to carry over to the next generation unchanged
FIXED_MAGNITUDE = 0.5

# Fitness weights: (weight for surface_area, weight for cupped_water_vol)
# Higher weight means more importance.
# Using (1.0, 0.2) means surface area is prioritized.
FITNESS_WEIGHTS = (1.0, 0.2) # (Weight for surface_area, Weight for cupped_water_vol)

# --- Landmark Definition ---
# Load landmarks from the example JSON to define the structure of our individuals
# (Assuming this script is in the same directory as 'face_individuals/example_params.json')
try:
    with open('face_individuals/example_params.json', 'r') as f:
        example_params_data = json.load(f)
except FileNotFoundError:
    print("Error: 'face_individuals/example_params.json' not found. Please ensure it's in the correct path.")
    exit(1)

LANDMARK_NAMES = [key for key in example_params_data.keys() if key != "default"]
DEFAULT_PARAMS_SECTION = example_params_data.get("default", {"direction": [0, 0, 0], "magnitude": 0.0})
NUM_GENES_PER_LANDMARK = 3  # dx, dy, dz
TOTAL_NUM_GENES = len(LANDMARK_NAMES) * NUM_GENES_PER_LANDMARK

print(f'we have this many genes: {TOTAL_NUM_GENES}')

# --- Temporary Directory for EA files ---
TEMP_DIR = "temp_ea_files"
if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR) # Clean up from previous runs
os.makedirs(TEMP_DIR, exist_ok=True)



# --- DEAP Setup ---
# Fitness: Maximize surface_area (primary) and cupped_water_vol (secondary)
creator.create("FitnessMultiMax", base.Fitness, weights=FITNESS_WEIGHTS)
creator.create("Individual", list, fitness=creator.FitnessMultiMax)

toolbox = base.Toolbox()

# Attribute generator: A gene is a float between -1 and 1 (for direction components)
toolbox.register("attr_float", random.uniform, -1.0, 1.0)

# Structure initializers
# Individual: list of floats (concatenated direction vectors)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, TOTAL_NUM_GENES)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def individual_to_params_json(individual):
    """Converts a DEAP individual (list of floats) to the params.json structure."""
    params_dict = {"default": DEFAULT_PARAMS_SECTION}
    gene_idx = 0
    for landmark_name in LANDMARK_NAMES:
        direction = [
            max(-1.0, min(1.0, individual[gene_idx])),
            max(-1.0, min(1.0, individual[gene_idx + 1])),
            max(-1.0, min(1.0, individual[gene_idx + 2])),
        ]
        params_dict[landmark_name] = {
            "direction": direction,
            "magnitude": FIXED_MAGNITUDE
        }
        gene_idx += 3
    return params_dict

# Evaluation function
def evaluate_individual(individual):
    individual_id = str(uuid.uuid4())
    params_json_path = os.path.join(TEMP_DIR, f"params_{individual_id}.json")
    generated_blend_path = os.path.join(TEMP_DIR, f"output_{individual_id}.blend")

    # 1. Convert individual to params.json
    params_data = individual_to_params_json(individual)
    with open(params_json_path, 'w') as f:
        json.dump(params_data, f, indent=2)


    success_generation = generate_face_mesh(
        blender_executable_path=BLENDER_EXECUTABLE_PATH,
        analyzer_script_path=GENERATION_HELPER_SCRIPT_PATH,
        blend_file_to_open=BASE_INPUT_BLEND_FILE,
        dither_config=params_json_path,
        output_file=generated_blend_path,
    )

    

    if not success_generation or not os.path.exists(generated_blend_path):
        print(f"  Generation failed for {individual_id}, assigning worst fitness.")
        return tuple(-float('inf') for _ in FITNESS_WEIGHTS) # Worst possible fitness for multi-objective

    analysis_results = run_blender_analysis(
        blender_executable_path=BLENDER_EXECUTABLE_PATH,
        analyzer_script_path=ANALYSIS_SCRIPT_PATH,
        blend_file_to_open=generated_blend_path, 
        object_name_in_blend=OBJECT_NAME_TO_ANALYZE,
        voxel_s=ANALYSIS_VOXEL_SIZE,
        create_debug=False,
        verbose_blender_output=False 
    )

    if analysis_results is None:
        print(f"  Analysis failed for {individual_id}, assigning worst fitness.")
        return tuple(-float('inf') for _ in FITNESS_WEIGHTS)

    surface_area = analysis_results.get('surface_area', 0)
    cupped_water_vol = analysis_results.get('cupped_water_vol', 0)

    return surface_area, cupped_water_vol


# Register genetic operators
toolbox.register("evaluate", evaluate_individual)
# No crossover as per request: toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=MUTATION_SIGMA, indpb=MUTATION_INDPB)
toolbox.register("select", tools.selNSGA2) # Good for multi-objective, handles FitnessMulti

# --- Main Evolutionary Loop ---
def main():
    print("Evolutionary Algorithm for Face Mesh Optimization")
    print(f"Landmarks: {len(LANDMARK_NAMES)}, Genes per individual: {TOTAL_NUM_GENES}")
    print(f"Population: {POPULATION_SIZE}, Generations: {N_GENERATIONS}")
    print(f"Mutation: Gaussian (sigma={MUTATION_SIGMA}, indpb={MUTATION_INDPB})")
    print(f"Fitness weights (SA, CWV): {FITNESS_WEIGHTS}")
    print(f"Fixed magnitude for all landmarks: {FIXED_MAGNITUDE}")
    print("---")

    pop = toolbox.population(n=POPULATION_SIZE)
    
    # Store hall of fame (best individuals found)
    # For multi-objective, halloffame stores non-dominated solutions
    hof = tools.ParetoFront()

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda x: tuple(round(sum(col)/len(col), 2) for col in zip(*x)))
    stats.register("min", lambda x: tuple(round(min(col), 2) for col in zip(*x)))
    stats.register("max", lambda x: tuple(round(max(col), 2) for col in zip(*x)))

    # Evaluate the first generation
    print("Evaluating initial population...")
    fitnesses = []
    for i, ind in enumerate(pop):
        print(f"Gen 0, Ind {i+1}/{len(pop)}:")
        ind.fitness.values = toolbox.evaluate(ind)
        fitnesses.append(ind.fitness.values)
    
    hof.update(pop)
    record = stats.compile(pop)
    print(f"Generation 0: {record}")
    print(f"Hall of Fame (Gen 0) size: {len(hof)}")
    for i, best_ind in enumerate(hof):
         print(f"  HOF {i}: Fitness = {best_ind.fitness.values}")


    # Evolution
    for gen in range(1, N_GENERATIONS + 1):
        print(f"\n--- Generation {gen}/{N_GENERATIONS} ---")

        # Select the next generation individuals (parents)
        # For NSGA-II, selection produces the offspring population directly of the same size
        offspring = toolbox.select(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring] # Clone selected individuals

        # Apply mutation to the offspring
        for mutant in offspring:
            if random.random() < MUTPB: # MUTPB is overall probability to mutate an individual
                toolbox.mutate(mutant)
                del mutant.fitness.values # Delete fitness values after mutation

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        print(f"Evaluating {len(invalid_ind)} new individuals for generation {gen}...")
        current_fitnesses = []
        for i, ind in enumerate(invalid_ind):
            print(f"Gen {gen}, Ind {i+1}/{len(invalid_ind)} (evaluating new):")
            ind.fitness.values = toolbox.evaluate(ind)
            current_fitnesses.append(ind.fitness.values)
        
        # The new population is the offspring
        pop[:] = offspring
        
        # Update the hall of fame with the generated individuals
        hof.update(pop)
        
        record = stats.compile(pop) # Stats from the current population
        print(f"Generation {gen}: {record}")
        print(f"Hall of Fame size: {len(hof)}")
        if hof:
            print("Best individuals in HOF:")
            for i, best_ind_in_hof in enumerate(list(hof)[:min(5, len(hof))]): # Print top 5 from HOF
                 print(f"  HOF {i}: Fitness = {best_ind_in_hof.fitness.values}")


    print("\n--- Evolution Finished ---")
    print(f"Best individuals found are in the Hall of Fame (Pareto Front) ({len(hof)} individuals):")
    
    best_overall_params = []
    for i, best_ind in enumerate(hof):
        print(f"Individual {i}: Fitness = {best_ind.fitness.values}")
        # Save the parameters of the best individuals
        params_data = individual_to_params_json(best_ind)
        best_params_path = os.path.join(TEMP_DIR, f"BEST_params_hof_{i}_sa{best_ind.fitness.values[0]:.2f}_cwv{best_ind.fitness.values[1]:.2f}.json")
        with open(best_params_path, 'w') as f:
            json.dump(params_data, f, indent=2)
        print(f"  Saved parameters to: {best_params_path}")
        best_overall_params.append({'fitness': best_ind.fitness.values, 'params_path': best_params_path, 'genes': list(best_ind)})

    # You can then inspect the JSON files in the TEMP_DIR for the best individuals.
    # The best_overall_params list also contains them.
    print(f"\nFind the parameter files for the best individuals in '{os.path.abspath(TEMP_DIR)}'")
    
    return pop, stats, hof, best_overall_params

if __name__ == "__main__":
    final_pop, final_stats, final_hof, best_params_list = main()