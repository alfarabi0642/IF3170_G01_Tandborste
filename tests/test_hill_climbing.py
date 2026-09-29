import pytest
from src.core.models import Package, Truck
from src.core.state import State
from src.algorithms.hill_climbing import (
    steepest_ascent_hill_climbing,
    hill_climbing_sideways_move,
    stochastic_hill_climbing,
    random_restart_hill_climbing,
)


@pytest.fixture
def problem_setup():
    truck = Truck(id="Truck-HC", w=10, l=15, h=5, max_capacity=100.0)
    packages = [
        Package(id="P1", w=3, l=4, h=1, value=60.0, weight=10.0, is_fragile=False, eta=3),
        Package(id="P2", w=4, l=5, h=2, value=110.0, weight=25.0, is_fragile=True, eta=2),
        Package(id="P3", w=2, l=3, h=1, value=45.0, weight=8.0, is_fragile=False, eta=5),
        Package(id="P4", w=5, l=5, h=2, value=150.0, weight=35.0, is_fragile=False, eta=1),
        Package(id="P5", w=3, l=3, h=2, value=80.0, weight=18.0, is_fragile=False, eta=4),
    ]
    return truck, packages


def test_steepest_ascent(problem_setup):
    truck, packages = problem_setup
    initial_state = State.create_random_state(truck, packages)

    res = steepest_ascent_hill_climbing(initial_state, max_iterations=20, sample_neighbors=10)

    assert res.algorithm_name == "Steepest Ascent Hill-Climbing"
    assert res.final_score >= res.initial_score
    assert res.final_state.is_valid() is True
    assert len(res.score_history) > 0
    assert len(res.iteration_history) == len(res.score_history)

    # Uji serialisasi dictionary untuk folder output/
    d = res.to_dict()
    assert "algorithm" in d
    assert "duration_seconds" in d
    assert "score_history" in d


def test_hill_climbing_sideways(problem_setup):
    truck, packages = problem_setup
    initial_state = State.create_random_state(truck, packages)

    res = hill_climbing_sideways_move(initial_state, max_iterations=30, max_sideways=10, sample_neighbors=10)

    assert res.algorithm_name == "Hill-Climbing with Sideways Move"
    assert res.final_score >= res.initial_score
    assert res.final_state.is_valid() is True
    assert "max_sideways_parameter" in res.extra_metrics
    assert res.extra_metrics["max_sideways_parameter"] == 10


def test_stochastic_hill_climbing(problem_setup):
    truck, packages = problem_setup
    initial_state = State.create_random_state(truck, packages)

    res = stochastic_hill_climbing(initial_state, max_iterations=40)

    assert res.algorithm_name == "Stochastic Hill-Climbing"
    assert res.iterations == 40
    assert res.final_score >= res.initial_score
    assert res.final_state.is_valid() is True
    assert "accepted_moves_count" in res.extra_metrics


def test_random_restart_hill_climbing(problem_setup):
    truck, packages = problem_setup

    res = random_restart_hill_climbing(
        truck=truck,
        packages=packages,
        max_restarts=3,
        iterations_per_restart=10,
    )

    assert res.algorithm_name == "Random Restart Hill-Climbing"
    assert res.final_state.is_valid() is True
    assert res.extra_metrics["restarts_performed"] == 3
    assert len(res.extra_metrics["scores_per_restart"]) == 3
    assert res.final_score >= 0.0
