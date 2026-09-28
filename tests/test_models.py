import pytest
from src.core.models import Package, Truck, ORIENTATIONS


def test_truck_basics():
    truck = Truck(id="Truck-01", w=10, l=20, h=5, max_capacity=500.0)
    assert truck.id == "Truck-01"
    assert truck.w == 10
    assert truck.l == 20
    assert truck.h == 5
    assert truck.max_capacity == 500.0
    assert truck.volume() == 10 * 20 * 5

    data = truck.to_dict()
    restored = Truck.from_dict(data)
    assert restored.id == truck.id
    assert restored.volume() == truck.volume()


def test_package_all_orientations():
    # Paket dengan dimensi unik untuk membedakan rotasi
    pkg = Package(id="Box-1", w=3, l=4, h=1, value=100, weight=10)
    assert len(ORIENTATIONS) == 6

    expected_rotations = [
        (3, 4, 1),
        (3, 1, 4),
        (4, 3, 1),
        (4, 1, 3),
        (1, 3, 4),
        (1, 4, 3),
    ]

    for idx, expected in enumerate(expected_rotations):
        pkg.set_orientation(idx)
        assert pkg.current_dims() == expected

    with pytest.raises(ValueError):
        pkg.set_orientation(6)


def test_package_bounding_box():
    pkg = Package(id="Box-1", w=3, l=4, h=2, value=50, weight=5)
    pkg.set_position(2, 3, 1)
    pkg.set_orientation(0)  # (3, 4, 2)
    pkg.is_loaded = True

    min_pt, max_pt = pkg.bounding_box()
    assert min_pt == (2, 3, 1)
    assert max_pt == (5, 7, 3)


def test_package_cloning():
    pkg = Package(id="Box-1", w=2, l=3, h=4, value=80, weight=15, is_fragile=True, eta=2)
    pkg.set_position(1, 1, 0)
    pkg.is_loaded = True
    pkg.set_orientation(2)

    clone_pkg = pkg.clone()
    assert clone_pkg.id == pkg.id
    assert clone_pkg.current_dims() == pkg.current_dims()
    assert clone_pkg.x == 1 and clone_pkg.y == 1 and clone_pkg.z == 0
    assert clone_pkg.is_fragile is True

    # Mutasi clone tidak boleh mengubah objek awal
    clone_pkg.set_position(5, 5, 2)
    assert pkg.x == 1
    assert clone_pkg.x == 5
