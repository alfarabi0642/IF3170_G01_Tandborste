import random
from typing import List, Optional, Callable

from src.core.models import Truck, Package
from src.core.state import State
from src.core.objectives import default_objective
from src.algorithms.base import SearchResult
from src.algorithms.hill_climbing import (
    steepest_ascent_hill_climbing,
    hill_climbing_sideways_move,
    stochastic_hill_climbing,
    random_restart_hill_climbing,
)


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

    # set seed kalau dikasih
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
