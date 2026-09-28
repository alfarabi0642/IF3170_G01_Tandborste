import pytest
from src.core.models import Package, Truck
from src.core.state import (
    State,
    neighbor_move,
    neighbor_swap,
    neighbor_rotate,
    get_random_neighbor,
    get_all_neighbors,
)
from src.core.objectives import default_objective, bonus_urgency_volume_objective


@pytest.fixture
def sample_problem():
    truck = Truck(id="Truck-01", w=10, l=10, h=5, max_capacity=100.0)
    packages = [
        Package(id="P1", w=3, l=4, h=1, value=60.0, weight=10.0, is_fragile=False, eta=2),
        Package(id="P2", w=4, l=5, h=2, value=100.0, weight=20.0, is_fragile=True, eta=1),
        Package(id="P3", w=2, l=3, h=1, value=40.0, weight=8.0, is_fragile=False, eta=4),
        Package(id="P4", w=3, l=3, h=2, value=80.0, weight=15.0, is_fragile=False, eta=3),
    ]
    return truck, packages


def test_state_basic_properties(sample_problem):
    truck, packages = sample_problem
    state = State(truck=truck, packages=packages)

    # Awalnya semua belum dimuat
    assert len(state.loaded_packages) == 0
    assert len(state.unloaded_packages) == 4
    assert state.get_score() == 0.0

    # Muat P1 di lantai
    packages[0].is_loaded = True
    packages[0].set_position(0, 0, 0)

    assert len(state.loaded_packages) == 1
    assert len(state.unloaded_packages) == 3
    assert state.total_loaded_weight() == 10.0
    assert state.total_loaded_volume() == 12  # 3 * 4 * 1
    assert state.is_valid() is True
    assert state.get_score() == 60.0


def test_state_cloning(sample_problem):
    truck, packages = sample_problem
    packages[0].is_loaded = True
    packages[0].set_position(0, 0, 0)

    state = State(truck, packages)
    clone_state = state.clone()

    assert clone_state.get_score() == state.get_score()
    assert len(clone_state.loaded_packages) == 1

    # Modifikasi clone tidak boleh merusak state asal
    clone_state.loaded_packages[0].set_position(5, 5, 0)
    assert state.loaded_packages[0].x == 0
    assert clone_state.loaded_packages[0].x == 5


def test_bonus_objective_function(sample_problem):
    truck, packages = sample_problem
    # P1: value=60, eta=2 -> 60 / (2+1) = 20
    # P3: value=40, eta=4 -> 40 / (4+1) = 8
    packages[0].is_loaded = True
    packages[0].set_position(0, 0, 0)

    packages[2].is_loaded = True
    packages[2].set_position(4, 0, 0)

    state = State(truck, packages, objective_fn=bonus_urgency_volume_objective)
    assert state.is_valid() is True

    score = state.get_score()
    expected_weighted_val = (60.0 / 3.0) + (40.0 / 5.0)  # 20 + 8 = 28
    # Volume total = (3*4*1) + (2*3*1) = 12 + 6 = 18. Truck vol = 500.
    vol_ratio = 18.0 / 500.0
    expected_score = expected_weighted_val + (20.0 * vol_ratio)

    assert pytest.approx(score, 0.01) == expected_score


def test_create_random_state_validity(sample_problem):
    truck, packages = sample_problem
    # Uji 5 kali untuk memastikan initial state random selalu valid secara fisika
    for _ in range(5):
        random_state = State.create_random_state(truck, packages)
        assert random_state.is_valid() is True
        assert len(random_state.get_errors()) == 0


def test_neighbor_generation(sample_problem):
    truck, packages = sample_problem
    # Buat initial state
    initial_state = State.create_random_state(truck, packages)

    # Uji pembangkitan tetangga acak
    neighbor = get_random_neighbor(initial_state, max_tries=30)
    if neighbor is not None:
        assert neighbor.is_valid() is True
        assert isinstance(neighbor.get_score(), float)

    # Uji himpunan tetangga untuk Steepest Ascent
    all_neighbors = get_all_neighbors(initial_state, sample_limit=5)
    for nb in all_neighbors:
        assert nb.is_valid() is True
