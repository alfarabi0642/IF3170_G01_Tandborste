import random

import pytest
from src.core.models import Package, Truck, ORIENTATIONS
from src.core.objectives import default_objective, bonus_urgency_volume_objective
from src.algorithms.genetic_algorithm import (
    decode,
    random_chromosome,
    order_crossover,
    mutate,
    genetic_algorithm,
)


@pytest.fixture
def problem_setup():
    truck = Truck(id="Truck-GA", w=10, l=15, h=5, max_capacity=100.0)
    packages = [
        Package(id="P1", w=3, l=4, h=1, value=60.0, weight=10.0, is_fragile=False, eta=3),
        Package(id="P2", w=4, l=5, h=2, value=110.0, weight=25.0, is_fragile=True, eta=2),
        Package(id="P3", w=2, l=3, h=1, value=45.0, weight=8.0, is_fragile=False, eta=5),
        Package(id="P4", w=5, l=5, h=2, value=150.0, weight=35.0, is_fragile=False, eta=1),
        Package(id="P5", w=3, l=3, h=2, value=80.0, weight=18.0, is_fragile=False, eta=4),
        Package(id="P6", w=4, l=4, h=3, value=90.0, weight=30.0, is_fragile=False, eta=2),
    ]
    return truck, packages


def _is_permutation(chrom, n):
    return sorted(idx for idx, _ in chrom) == list(range(n))


def test_decode_always_valid(problem_setup):
    random.seed(1)
    truck, packages = problem_setup
    for _ in range(50):
        chrom = random_chromosome(len(packages))
        state = decode(chrom, truck, packages)
        assert state.is_valid() is True
        assert state.total_loaded_weight() <= truck.max_capacity


def test_decode_is_deterministic(problem_setup):
    random.seed(2)
    truck, packages = problem_setup
    chrom = random_chromosome(len(packages))
    s1 = decode(chrom, truck, packages)
    s2 = decode(chrom, truck, packages)
    assert s1.get_score() == s2.get_score()
    assert [(p.x, p.y, p.z, p.is_loaded) for p in s1.packages] == [(p.x, p.y, p.z, p.is_loaded) for p in s2.packages]


def test_decode_does_not_modify_input(problem_setup):
    truck, packages = problem_setup
    decode(random_chromosome(len(packages)), truck, packages)
    assert all(not p.is_loaded for p in packages)


def test_decode_fragile_not_stacked():
    # Paket pecah-belah dimuat pertama; paket kedua tidak boleh ditaruh di atasnya
    truck = Truck(id="T", w=2, l=2, h=4, max_capacity=100.0)
    packages = [
        Package(id="F", w=2, l=2, h=1, value=10, weight=1, is_fragile=True),
        Package(id="B", w=2, l=2, h=1, value=10, weight=1),
    ]
    state = decode([(0, 0), (1, 0)], truck, packages)
    assert state.is_valid() is True
    assert [p.id for p in state.loaded_packages] == ["F"]


def test_decode_respects_capacity():
    truck = Truck(id="T", w=10, l=10, h=10, max_capacity=15.0)
    packages = [
        Package(id="A", w=1, l=1, h=1, value=10, weight=10),
        Package(id="B", w=1, l=1, h=1, value=10, weight=10),
    ]
    state = decode([(0, 0), (1, 0)], truck, packages)
    assert len(state.loaded_packages) == 1


def test_order_crossover_keeps_permutation():
    random.seed(3)
    n = 8
    for _ in range(100):
        p1 = random_chromosome(n)
        p2 = random_chromosome(n)
        child = order_crossover(p1, p2)
        assert len(child) == n
        assert _is_permutation(child, n)
        # Orientasi ikut terbawa bersama paketnya dari salah satu induk
        for idx, ori in child:
            assert (idx, ori) in p1 or (idx, ori) in p2


def test_mutate_keeps_permutation():
    random.seed(4)
    n = 8
    chrom = random_chromosome(n)
    mutated = mutate(chrom, mutation_rate=1.0)
    assert _is_permutation(mutated, n)
    assert all(0 <= ori < len(ORIENTATIONS) for _, ori in mutated)
    # Kromosom asal tidak ikut berubah
    assert mutate(chrom, mutation_rate=0.0) == chrom


def test_genetic_algorithm_run(problem_setup):
    random.seed(5)
    truck, packages = problem_setup
    res = genetic_algorithm(truck, packages, pop_size=10, generations=8)

    assert res.algorithm_name == "Genetic Algorithm"
    assert res.final_state.is_valid() is True
    assert res.final_score >= res.initial_score
    assert res.iterations == 8

    m = res.extra_metrics
    assert len(m["max_fitness_history"]) == len(m["avg_fitness_history"]) == 9
    assert len(res.iteration_history) == len(res.score_history) == 9
    # Dengan elitism, fitness maksimum tidak pernah turun
    maxes = m["max_fitness_history"]
    assert all(a <= b for a, b in zip(maxes, maxes[1:]))
    assert all(avg <= mx for avg, mx in zip(m["avg_fitness_history"], maxes))
    assert res.final_score == pytest.approx(maxes[-1])

    d = res.to_dict()
    assert d["algorithm"] == "Genetic Algorithm"


def test_genetic_algorithm_bonus_objective(problem_setup):
    random.seed(6)
    truck, packages = problem_setup
    res = genetic_algorithm(truck, packages, pop_size=8, generations=5,
                            objective_fn=bonus_urgency_volume_objective)
    assert res.final_state.is_valid() is True
    assert res.final_score == pytest.approx(bonus_urgency_volume_objective(res.final_state))
    assert res.final_score != pytest.approx(default_objective(res.final_state))
