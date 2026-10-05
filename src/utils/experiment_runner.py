import random
import time
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union

from src.core.models import Truck, Package
from src.core.state import State
from src.core.objectives import default_objective, bonus_urgency_volume_objective
from src.algorithms.base import SearchResult
from src.algorithms.hill_climbing import (
    steepest_ascent_hill_climbing,
    hill_climbing_sideways_move,
    stochastic_hill_climbing,
    random_restart_hill_climbing,
)
from src.utils.io_handler import save_run_result
from src.visualization.metrics import plot_objective_history, plot_ga_fitness


# mapping input ke algoritma
ALGORITHM_MAP = {
    "steepest": "steepest_ascent",
    "steepest_ascent": "steepest_ascent",
    "sideways": "sideways_move",
    "sideways_move": "sideways_move",
    "stochastic": "stochastic",
    "restart": "random_restart",
    "random_restart": "random_restart",
    "sa": "simulated_annealing",
    "simulated_annealing": "simulated_annealing",
    "ga": "genetic_algorithm",
    "genetic_algorithm": "genetic_algorithm",
}

# helper impor fungsi algortima SA & GA
def _try_get_driver2_algo(algo_key: str):
    if algo_key == "simulated_annealing":
        try:
            from src.algorithms.simulated_annealing import simulated_annealing
            return simulated_annealing
        except ImportError:
            raise NotImplementedError(
                "SA not implemented"
            )
    elif algo_key == "genetic_algorithm":
        try:
            from src.algorithms.genetic_algorithm import genetic_algorithm
            return genetic_algorithm
        except ImportError:
            raise NotImplementedError(
                "GA not implemented"
            )
    return None

# fungsi untuk menjalankan satu kali pencarian algoritma
def run_single_search(
    algo_name: str,
    truck: Truck,
    packages: List[Package],
    initial_state: Optional[State] = None,
    objective_fn: Callable[[State], float] = default_objective,
    seed: Optional[int] = None,
    **algo_kwargs,
) -> SearchResult:
  
    # validasi nama alg
    key = ALGORITHM_MAP.get(algo_name.lower().strip())
    if not key:
        valid_keys = ", ".join(sorted(set(ALGORITHM_MAP.keys())))
        raise ValueError(f"Algoritma '{algo_name}' tidak dikenal. Pilihan tersedia: {valid_keys}")

    # set seed, jika gak dikasih
    if seed is not None:
        random.seed(seed)

    # kasus random restart hill-climbing, terima truck dan packages
    if key == "random_restart":
        restarts = algo_kwargs.get("max_restarts", algo_kwargs.get("restarts", 5))
        iters = algo_kwargs.get("iterations_per_restart", algo_kwargs.get("iter_per_restart", 100))
        return random_restart_hill_climbing(
            truck=truck,
            packages=packages,
            max_restarts=restarts,
            iterations_per_restart=iters,
            objective_fn=objective_fn,
        )

    # kasus GA, terima truck, packages, pop_size, generations, crossover_rate, mutation_rate, elite
    if key == "genetic_algorithm":
        ga_func = _try_get_driver2_algo(key)
        pop_size = algo_kwargs.get("pop_size", algo_kwargs.get("pop", 30))
        generations = algo_kwargs.get("generations", algo_kwargs.get("max_iterations", 50))
        crossover_rate = algo_kwargs.get("crossover_rate", 0.9)
        mutation_rate = algo_kwargs.get("mutation_rate", 0.1)
        elite = algo_kwargs.get("elite", 2)

        return ga_func(
            truck=truck,
            packages=packages,
            pop_size=pop_size,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elite=elite,
            objective_fn=objective_fn,
        )

    # alg berbasis state tunggal
    start_state = initial_state
    if start_state is None:
        start_state = State.create_random_state(truck, packages, objective_fn=objective_fn)

    # Steepest Ascent Hill-Climbing
    if key == "steepest_ascent":
        max_iter = algo_kwargs.get("max_iterations", 300)
        sample_nb = algo_kwargs.get("sample_neighbors", 25)
        return steepest_ascent_hill_climbing(
            initial_state=start_state,
            max_iterations=max_iter,
            sample_neighbors=sample_nb,
        )

    # Hill-Climbing with Sideways Move
    elif key == "sideways_move":
        max_iter = algo_kwargs.get("max_iterations", 500)
        max_side = algo_kwargs.get("max_sideways", 50)
        sample_nb = algo_kwargs.get("sample_neighbors", 25)
        return hill_climbing_sideways_move(
            initial_state=start_state,
            max_iterations=max_iter,
            max_sideways=max_side,
            sample_neighbors=sample_nb,
        )

    # Stochastic Hill-Climbing
    elif key == "stochastic":
        max_iter = algo_kwargs.get("max_iterations", 300)
        return stochastic_hill_climbing(
            initial_state=start_state,
            max_iterations=max_iter,
        )

    # Simulated Annealing
    elif key == "simulated_annealing":
        sa_func = _try_get_driver2_algo(key)
        t0 = algo_kwargs.get("t0", 200.0)
        alpha = algo_kwargs.get("alpha", 0.95)
        t_min = algo_kwargs.get("t_min", 0.01)
        iter_per_temp = algo_kwargs.get("iter_per_temp", 10)
        stuck_limit = algo_kwargs.get("stuck_limit", 50)

        return sa_func(
            initial_state=start_state,
            t0=t0,
            alpha=alpha,
            t_min=t_min,
            iter_per_temp=iter_per_temp,
            stuck_limit=stuck_limit,
        )

    raise NotImplementedError(f"Algoritma '{algo_name}' belum tersedia.")

# fungsi untuk menjalankan eksperimen berulang (default 3 run independen)
def run_experiment(
    algo_name: str,
    truck: Truck,
    packages: List[Package],
    num_runs: int = 3,
    seeds: Optional[List[int]] = None,
    objective_fn: Callable[[State], float] = default_objective,
    output_dir: Optional[Union[str, Path]] = None,
    save_plots: bool = False,
    **algo_kwargs,
) -> Dict[str, Any]:
    results: List[SearchResult] = []
    
    # buat random seed
    run_seeds = seeds if seeds is not None else [random.randint(1000, 999999) for _ in range(num_runs)]

    # loop pengujian independen
    for run_idx in range(num_runs):
        seed = run_seeds[run_idx]
        res = run_single_search(
            algo_name=algo_name,
            truck=truck,
            packages=packages,
            objective_fn=objective_fn,
            seed=seed,
            **algo_kwargs,
        )

        # add metadata pengujian ke extra_metrics
        res.extra_metrics["run_id"] = run_idx + 1
        res.extra_metrics["seed"] = seed
        results.append(res)

        # simpan file log JSON per run jika folder output ditentukan
        if output_dir:
            out_folder = Path(output_dir)
            algo_slug = algo_name.lower().replace(" ", "_")
            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{algo_slug}_run{run_idx + 1}_{ts}.json"
            save_run_result(res, output_dir=out_folder, filename=filename)

    # hitung statistik  (mean, best, worst, std dev)
    scores = [r.final_score for r in results]
    durations = [r.duration_seconds for r in results]
    iterations = [r.iterations for r in results]

    mean_score = sum(scores) / len(scores) if scores else 0.0
    var_score = sum((s - mean_score) ** 2 for s in scores) / len(scores) if scores else 0.0
    std_score = math.sqrt(var_score)

    summary = {
        "algorithm": results[0].algorithm_name if results else algo_name,
        "total_runs": num_runs,
        "seeds": run_seeds,
        "scores": scores,
        "mean_score": round(mean_score, 2),
        "best_score": round(max(scores), 2) if scores else 0.0,
        "worst_score": round(min(scores), 2) if scores else 0.0,
        "std_score": round(std_score, 2),
        "mean_duration_seconds": round(sum(durations) / len(durations), 4) if durations else 0.0,
        "mean_iterations": round(sum(iterations) / len(iterations), 1) if iterations else 0.0,
    }

    # visualisasi grafik kurva konvergensi gabungan untuk semua run
    if save_plots and output_dir and results:
        out_folder = Path(output_dir)
        algo_slug = algo_name.lower().replace(" ", "_")
        plot_path = out_folder / f"{algo_slug}_convergence.png"
        plot_objective_history(
            results=results,
            title=f"Konvergensi {results[0].algorithm_name} ({num_runs} Runs)",
            save_path=plot_path,
            show=False,
        )
        summary["plot_saved"] = str(plot_path)

    return {
        "results": results,
        "summary": summary,
    }

    # fungsi untuk studi parameter GA, menjalankan eksperimen matriks 18 run (3x iterasi x 3x populasi)
def run_ga_parameter_study(
    truck: Truck,
    packages: List[Package],
    output_dir: Optional[Union[str, Path]] = None,
    save_plots: bool = True,
    objective_fn: Callable[[State], float] = default_objective,
    study_reps: int = 3,
    iteration_settings: Optional[List[int]] = None,
    fixed_population: int = 30,
    population_settings: Optional[List[int]] = None,
    fixed_iterations: int = 50,
) -> Dict[str, Any]:
    iter_list = iteration_settings or [20, 50, 100]
    pop_list = population_settings or [15, 30, 60]

    study_results: Dict[str, Any] = {
        "study_iterations": {},
        "study_population": {},
        "all_runs_count": 0,
    }

    out_folder = Path(output_dir) if output_dir else Path("output")
    out_folder.mkdir(parents=True, exist_ok=True)

    print(f"\n========================================================")
    print(f"  MEMULAI EKSPERIMEN GA: 18-RUN PARAMETER STUDY MATRIX")
    print(f"========================================================")

    # Studi 1, populasi konstan, 9 runs
    print(f"\n>>> STUDI 1: Pengaruh Banyak Iterasi (Populasi Tetap = {fixed_population})")
    for gens in iter_list:
        print(f"  -> Menjalankan {study_reps}x run dengan {gens} generasi...")
        exp = run_experiment(
            algo_name="ga",
            truck=truck,
            packages=packages,
            num_runs=study_reps,
            objective_fn=objective_fn,
            output_dir=out_folder,
            save_plots=False,
            pop_size=fixed_population,
            generations=gens,
        )
        study_results["study_iterations"][f"gens_{gens}"] = exp["summary"]
        study_results["all_runs_count"] += study_reps

        # Buat grafik kurva fitness maksimum vs rata-rata populasi untuk run pertama
        if save_plots and exp["results"]:
            first_res = exp["results"][0]
            max_fit = first_res.extra_metrics.get("max_fitness_history", [])
            avg_fit = first_res.extra_metrics.get("avg_fitness_history", [])
            if max_fit and avg_fit:
                plot_path = out_folder / f"ga_study1_gens{gens}_fitness.png"
                plot_ga_fitness(
                    max_fitness=max_fit,
                    avg_fitness=avg_fit,
                    title=f"GA Studi 1: {gens} Generasi (Pop {fixed_population})",
                    save_path=plot_path,
                    show=False,
                )

    # Studi 2, iterasi konstan, 9 runs
    print(f"\n>>> STUDI 2: Pengaruh Ukuran Populasi (Generasi Tetap = {fixed_iterations})")
    for pop in pop_list:
        print(f"  -> Menjalankan {study_reps}x run dengan ukuran populasi {pop}...")
        exp = run_experiment(
            algo_name="ga",
            truck=truck,
            packages=packages,
            num_runs=study_reps,
            objective_fn=objective_fn,
            output_dir=out_folder,
            save_plots=False,
            pop_size=pop,
            generations=fixed_iterations,
        )
        study_results["study_population"][f"pop_{pop}"] = exp["summary"]
        study_results["all_runs_count"] += study_reps

        if save_plots and exp["results"]:
            first_res = exp["results"][0]
            max_fit = first_res.extra_metrics.get("max_fitness_history", [])
            avg_fit = first_res.extra_metrics.get("avg_fitness_history", [])
            if max_fit and avg_fit:
                plot_path = out_folder / f"ga_study2_pop{pop}_fitness.png"
                plot_ga_fitness(
                    max_fitness=max_fit,
                    avg_fitness=avg_fit,
                    title=f"GA Studi 2: Populasi {pop} ({fixed_iterations} Generasi)",
                    save_path=plot_path,
                    show=False,
                )

    print(f"\n[Selesai] Berhasil mengeksekusi {study_results['all_runs_count']} runs eksperimen GA.")
    return study_results
