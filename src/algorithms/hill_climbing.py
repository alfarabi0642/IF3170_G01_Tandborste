import time
from typing import List, Callable, Optional
from src.core.models import Package, Truck
from src.core.state import State, get_all_neighbors, get_random_neighbor
from src.core.objectives import default_objective
from .base import SearchResult


def steepest_ascent_hill_climbing(
    initial_state: State,
    max_iterations: int = 300,
    sample_neighbors: int = 25,
) -> SearchResult:
    """
    Steepest Ascent Hill-Climbing Search.
    - Dimulai dari initial state acak.
    - Loop bergerak secara kontinu ke arah tetangga dengan nilai objektif tertinggi.
    - Berhenti ketika mencapai puncak (local optima/flat) di mana tidak ada tetangga
      yang memiliki nilai lebih tinggi dari state saat ini.
    """
    start_time = time.time()
    current = initial_state.clone()
    initial_score = current.get_score()

    score_history = [initial_score]
    iteration_history = [0]

    iteration = 0
    while iteration < max_iterations:
        iteration += 1

        # Bangkitkan kandidat himpunan tetangga
        neighbors = get_all_neighbors(current, sample_limit=sample_neighbors)
        if not neighbors:
            break

        # Cari tetangga terbaik
        best_neighbor = max(neighbors, key=lambda s: s.get_score())
        best_score = best_neighbor.get_score()
        current_score = current.get_score()

        # Berhenti jika tidak ada tetangga yang memberikan peningkatan nilai
        if best_score <= current_score:
            score_history.append(current_score)
            iteration_history.append(iteration)
            break

        current = best_neighbor
        score_history.append(best_score)
        iteration_history.append(iteration)

    duration = time.time() - start_time

    return SearchResult(
        algorithm_name="Steepest Ascent Hill-Climbing",
        initial_state=initial_state,
        final_state=current,
        initial_score=initial_score,
        final_score=current.get_score(),
        iterations=iteration,
        duration_seconds=duration,
        score_history=score_history,
        iteration_history=iteration_history,
        extra_metrics={"termination_reason": "local_optimum" if iteration < max_iterations else "max_iterations"},
    )


def hill_climbing_sideways_move(
    initial_state: State,
    max_iterations: int = 500,
    max_sideways: int = 50,
    sample_neighbors: int = 25,
) -> SearchResult:
    """
    Hill-Climbing with Sideways Move.
    - Mengizinkan pergerakan datar (delta E = 0) untuk melintasi dataran (shoulder/flat).
    - Membatasi pergerakan sideways berturut-turut hingga parameter `max_sideways`.
    - Berhenti ketika mencapai puncak atau sideways move berturut-turut melebihi limit.
    """
    start_time = time.time()
    current = initial_state.clone()
    initial_score = current.get_score()

    score_history = [initial_score]
    iteration_history = [0]

    consecutive_sideways = 0
    total_sideways_count = 0
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        neighbors = get_all_neighbors(current, sample_limit=sample_neighbors)
        if not neighbors:
            break

        best_neighbor = max(neighbors, key=lambda s: s.get_score())
        best_score = best_neighbor.get_score()
        current_score = current.get_score()
        delta = best_score - current_score

        if delta < 0:
            # Tetangga terbaik pun lebih buruk (sudah di puncak murni)
            score_history.append(current_score)
            iteration_history.append(iteration)
            break
        elif delta == 0:
            consecutive_sideways += 1
            total_sideways_count += 1
            if consecutive_sideways > max_sideways:
                # Batas maksimum sideways move tercapai
                score_history.append(current_score)
                iteration_history.append(iteration)
                break
            current = best_neighbor
        else:
            # Terjadi peningkatan nilai: reset counter sideways berturut-turut
            consecutive_sideways = 0
            current = best_neighbor

        score_history.append(current.get_score())
        iteration_history.append(iteration)

    duration = time.time() - start_time

    return SearchResult(
        algorithm_name="Hill-Climbing with Sideways Move",
        initial_state=initial_state,
        final_state=current,
        initial_score=initial_score,
        final_score=current.get_score(),
        iterations=iteration,
        duration_seconds=duration,
        score_history=score_history,
        iteration_history=iteration_history,
        extra_metrics={
            "max_sideways_parameter": max_sideways,
            "total_sideways_performed": total_sideways_count,
            "consecutive_sideways_at_stop": consecutive_sideways,
        },
    )


def stochastic_hill_climbing(
    initial_state: State,
    max_iterations: int = 300,
) -> SearchResult:
    """
    Stochastic Hill-Climbing Search.
    - Di setiap iterasi, membangkitkan 1 tetangga secara acak (bukan mengevaluasi semua tetangga).
    - Berpindah ke tetangga tersebut jika nilainya lebih baik daripada state saat ini.
    - Berhenti ketika mencapai batas iterasi n_max.
    """
    start_time = time.time()
    current = initial_state.clone()
    initial_score = current.get_score()

    score_history = [initial_score]
    iteration_history = [0]

    accepted_moves = 0

    for iteration in range(1, max_iterations + 1):
        # Membangkitkan 1 tetangga acak tunggal
        neighbor = get_random_neighbor(current, max_tries=10)
        if neighbor is not None:
            if neighbor.get_score() > current.get_score():
                current = neighbor
                accepted_moves += 1

        score_history.append(current.get_score())
        iteration_history.append(iteration)

    duration = time.time() - start_time

    return SearchResult(
        algorithm_name="Stochastic Hill-Climbing",
        initial_state=initial_state,
        final_state=current,
        initial_score=initial_score,
        final_score=current.get_score(),
        iterations=max_iterations,
        duration_seconds=duration,
        score_history=score_history,
        iteration_history=iteration_history,
        extra_metrics={
            "accepted_moves_count": accepted_moves,
        },
    )


def random_restart_hill_climbing(
    truck: Truck,
    packages: List[Package],
    max_restarts: int = 5,
    iterations_per_restart: int = 100,
    objective_fn: Callable[[State], float] = default_objective,
) -> SearchResult:
    """
    Random Restart Hill-Climbing Search.
    - Melakukan serangkaian pencarian hill-climbing dari state awal acak yang berbeda.
    - Menyimpan dan mengembalikan solusi terbaik global dari seluruh iterasi restart.
    - Berhenti ketika jumlah restart mencapai parameter `max_restart`.
    """
    start_time = time.time()

    best_global_state: Optional[State] = None
    best_global_score = -1.0
    first_initial_state: Optional[State] = None

    combined_score_history: List[float] = []
    combined_iteration_history: List[int] = []

    cumulative_iterations = 0
    restart_scores = []
    iterations_per_run = []

    for r in range(max_restarts):
        # Bangkitkan initial state acak baru
        init_state = State.create_random_state(truck, packages, objective_fn)
        if first_initial_state is None:
            first_initial_state = init_state.clone()

        # Jalankan salah satu pencarian lokal (Steepest Ascent) per restart
        run_res = steepest_ascent_hill_climbing(
            init_state,
            max_iterations=iterations_per_restart,
            sample_neighbors=15,
        )

        restart_scores.append(round(run_res.final_score, 2))
        iterations_per_run.append(run_res.iterations)

        # Gabungkan kurva iterasi
        for score in run_res.score_history:
            cumulative_iterations += 1
            combined_iteration_history.append(cumulative_iterations)
            combined_score_history.append(score)

        if best_global_state is None or run_res.final_score > best_global_score:
            best_global_score = run_res.final_score
            best_global_state = run_res.final_state.clone()

    duration = time.time() - start_time

    assert first_initial_state is not None
    assert best_global_state is not None

    return SearchResult(
        algorithm_name="Random Restart Hill-Climbing",
        initial_state=first_initial_state,
        final_state=best_global_state,
        initial_score=first_initial_state.get_score(),
        final_score=best_global_state.get_score(),
        iterations=cumulative_iterations,
        duration_seconds=duration,
        score_history=combined_score_history,
        iteration_history=combined_iteration_history,
        extra_metrics={
            "max_restart_parameter": max_restarts,
            "restarts_performed": max_restarts,
            "scores_per_restart": restart_scores,
            "iterations_per_restart_actual": iterations_per_run,
        },
    )
