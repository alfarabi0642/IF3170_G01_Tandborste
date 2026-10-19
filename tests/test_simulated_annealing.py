import random

import pytest
from src.core.models import Package, Truck
from src.core.state import State
from src.core.objectives import bonus_urgency_volume_objective
from src.algorithms.simulated_annealing import simulated_annealing


@pytest.fixture
def problem_setup():
    truck = Truck(id="Truck-SA", w=10, l=15, h=5, max_capacity=100.0)
    packages = [
        Package(id="P1", w=3, l=4, h=1, value=60.0, weight=10.0, is_fragile=False, eta=3),
        Package(id="P2", w=4, l=5, h=2, value=110.0, weight=25.0, is_fragile=True, eta=2),
        Package(id="P3", w=2, l=3, h=1, value=45.0, weight=8.0, is_fragile=False, eta=5),
        Package(id="P4", w=5, l=5, h=2, value=150.0, weight=35.0, is_fragile=False, eta=1),
        Package(id="P5", w=3, l=3, h=2, value=80.0, weight=18.0, is_fragile=False, eta=4),
        Package(id="P6", w=4, l=4, h=3, value=90.0, weight=30.0, is_fragile=False, eta=2),
    ]
    return truck, packages


def test_sa_result_valid(problem_setup):
    random.seed(1)
    truck, packages = problem_setup
    initial_state = State.create_random_state(truck, packages)

    res = simulated_annealing(initial_state, t0=100.0, alpha=0.9, t_min=0.1, iter_per_temp=5)

    assert res.algorithm_name == "Simulated Annealing"
    assert res.final_state.is_valid() is True
    # State akhir adalah state terbaik yang dikunjungi, jadi tidak mungkin lebih buruk dari awal
    assert res.final_score >= res.initial_score
    assert res.final_score == res.final_state.get_score()
    assert len(res.iteration_history) == len(res.score_history) == res.iterations + 1

    d = res.to_dict()
    assert d["algorithm"] == "Simulated Annealing"


def test_sa_does_not_modify_initial_state(problem_setup):
    random.seed(2)
    truck, packages = problem_setup
    initial_state = State.create_random_state(truck, packages)
    before = [(p.id, p.is_loaded, p.x, p.y, p.z, p.orientation_idx) for p in initial_state.packages]

    simulated_annealing(initial_state, t0=50.0, alpha=0.9, t_min=1.0, iter_per_temp=5)

    after = [(p.id, p.is_loaded, p.x, p.y, p.z, p.orientation_idx) for p in initial_state.packages]
    assert before == after


def test_sa_metrics(problem_setup):
    random.seed(3)
    truck, packages = problem_setup
    initial_state = State.create_random_state(truck, packages)

    res = simulated_annealing(initial_state, t0=100.0, alpha=0.9, t_min=0.1, iter_per_temp=5, stuck_limit=5)
    m = res.extra_metrics

    # exp(dE/T) dengan dE <= 0 selalu berada di (0, 1]
    assert len(m["prob_history"]) == len(m["prob_iterations"])
    assert all(0.0 <= p <= 1.0 for p in m["prob_history"])

    # Suhu tidak pernah naik dan selalu di atas t_min saat iterasi berjalan
    temps = m["temperature_history"]
    assert len(temps) == res.iterations
    assert all(t1 >= t2 for t1, t2 in zip(temps, temps[1:]))
    assert all(t > 0.1 for t in temps)

    assert m["stuck_count"] >= 0


def test_sa_iteration_count(problem_setup):
    random.seed(4)
    truck, packages = problem_setup
    initial_state = State.create_random_state(truck, packages)

    # Suhu 100, 50, 25, 12.5 dijalankan, lalu 6.25 <= 10 berhenti: 4 level suhu x 4 iterasi
    res = simulated_annealing(initial_state, t0=100.0, alpha=0.5, t_min=10.0, iter_per_temp=4)
    assert res.iterations == 16


def test_sa_bonus_objective(problem_setup):
    random.seed(5)
    truck, packages = problem_setup
    initial_state = State.create_random_state(truck, packages, bonus_urgency_volume_objective)

    res = simulated_annealing(initial_state, t0=20.0, alpha=0.9, t_min=0.5, iter_per_temp=5)

    assert res.final_state.is_valid() is True
    assert res.final_score == pytest.approx(bonus_urgency_volume_objective(res.final_state))


def test_sa_invalid_params(problem_setup):
    truck, packages = problem_setup
    initial_state = State.create_random_state(truck, packages)
    with pytest.raises(ValueError):
        simulated_annealing(initial_state, alpha=1.5)
    with pytest.raises(ValueError):
        simulated_annealing(initial_state, t0=0)
