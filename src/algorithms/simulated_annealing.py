import math
import random
import time
from typing import List

from src.core.state import State, get_random_neighbor
from .base import SearchResult


def simulated_annealing(
    initial_state: State,
    t0: float = 200.0,
    alpha: float = 0.95,
    t_min: float = 0.01,
    iter_per_temp: int = 10,
    stuck_limit: int = 50,
) -> SearchResult:
    """
    Simulated Annealing (Salindia Kuliah IF3170 - Beyond Classical Search, hal. 34-35).
    - Suhu awal t0, diturunkan secara geometrik T(k+1) = alpha * T(k) (cooling schedule).
    - Pada setiap suhu dibangkitkan iter_per_temp tetangga acak.
    - Tetangga lebih baik (dE > 0) selalu diterima, tetangga lebih buruk diterima
      dengan probabilitas exp(dE / T).
    - Berhenti ketika T <= t_min.
    - Solusi akhir adalah state terbaik yang pernah dikunjungi.

    Jumlah iterasi total kira-kira ln(t_min / t0) / ln(alpha) * iter_per_temp.
    """
    if t0 <= 0 or t_min <= 0:
        raise ValueError("t0 dan t_min harus bernilai positif")
    if not (0 < alpha < 1):
        raise ValueError("alpha harus berada di antara 0 dan 1")

    start_time = time.time()
    current = initial_state.clone()
    current_score = current.get_score()
    initial_score = current_score

    best = current
    best_score = current_score

    score_history: List[float] = [current_score]
    iteration_history: List[int] = [0]
    temp_history: List[float] = []

    # exp(dE/T) dicatat setiap kali tetangga yang tidak lebih baik dievaluasi
    prob_history: List[float] = []
    prob_iterations: List[int] = []

    # Dianggap "stuck" di local optima jika tetangga ditolak stuck_limit kali berturut-turut
    rejected_streak = 0
    stuck_count = 0
    accepted_worse = 0

    T = t0
    iteration = 0
    while T > t_min:
        for _ in range(iter_per_temp):
            iteration += 1
            neighbor = get_random_neighbor(current)

            accepted = False
            if neighbor is not None:
                neighbor_score = neighbor.get_score()
                delta = neighbor_score - current_score

                if delta > 0:
                    accepted = True
                else:
                    prob = math.exp(delta / T)
                    prob_history.append(prob)
                    prob_iterations.append(iteration)
                    if random.random() < prob:
                        accepted = True
                        if delta < 0:
                            accepted_worse += 1

                if accepted:
                    current = neighbor
                    current_score = neighbor_score

            if accepted:
                rejected_streak = 0
            else:
                rejected_streak += 1
                if rejected_streak == stuck_limit:
                    stuck_count += 1
                    rejected_streak = 0

            if current_score > best_score:
                best = current
                best_score = current_score

            score_history.append(current_score)
            iteration_history.append(iteration)
            temp_history.append(T)

        T *= alpha

    duration = time.time() - start_time

    return SearchResult(
        algorithm_name="Simulated Annealing",
        initial_state=initial_state,
        final_state=best,
        initial_score=initial_score,
        final_score=best_score,
        iterations=iteration,
        duration_seconds=duration,
        score_history=score_history,
        iteration_history=iteration_history,
        extra_metrics={
            "t0": t0,
            "alpha": alpha,
            "t_min": t_min,
            "iter_per_temp": iter_per_temp,
            "stuck_limit": stuck_limit,
            "stuck_count": stuck_count,
            "accepted_worse_moves": accepted_worse,
            "last_score": current_score,
            "temperature_history": temp_history,
            "prob_history": prob_history,
            "prob_iterations": prob_iterations,
        },
    )
