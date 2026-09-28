import pytest
from src.core.models import Package, Truck
from src.core.constraints import (
    is_within_bounds,
    is_overlapping_3d,
    has_horizontal_overlap,
    is_supported,
    is_fragile_violated,
    is_capacity_exceeded,
    check_all_constraints,
)


@pytest.fixture
def truck():
    return Truck(id="Truck-Test", w=10, l=10, h=10, max_capacity=100.0)


def test_boundary_constraints(truck):
    # Paket valid di dalam batas truk
    pkg_valid = Package("P1", w=3, l=4, h=2, value=10, weight=5)
    pkg_valid.set_position(0, 0, 0)
    assert is_within_bounds(pkg_valid, truck) is True

    # Paket menyentuh tepat batas pojok
    pkg_corner = Package("P2", w=2, l=3, h=1, value=10, weight=5)
    pkg_corner.set_position(8, 7, 9)
    assert is_within_bounds(pkg_corner, truck) is True

    # Paket menembus sumbu x
    pkg_out_x = Package("P3", w=4, l=2, h=1, value=10, weight=5)
    pkg_out_x.set_position(7, 0, 0)  # 7 + 4 = 11 > 10
    assert is_within_bounds(pkg_out_x, truck) is False

    # Paket dengan koordinat negatif
    pkg_neg = Package("P4", w=2, l=2, h=2, value=10, weight=5)
    pkg_neg.set_position(-1, 0, 0)
    assert is_within_bounds(pkg_neg, truck) is False


def test_overlap_3d():
    # Box 1: [0, 3] x [0, 4] x [0, 2]
    b1 = Package("B1", w=3, l=4, h=2, value=10, weight=5)
    b1.set_position(0, 0, 0)

    # Box 2: Bersebelahan menempel di sumbu x (x = 3..6) -> TIDAK overlap
    b2_touching = Package("B2", w=3, l=4, h=2, value=10, weight=5)
    b2_touching.set_position(3, 0, 0)
    assert is_overlapping_3d(b1, b2_touching) is False

    # Box 3: Beririsan volume dengan Box 1 (x = 2..5) -> OVERLAP
    b3_intersect = Package("B3", w=3, l=4, h=2, value=10, weight=5)
    b3_intersect.set_position(2, 1, 0)
    assert is_overlapping_3d(b1, b3_intersect) is True

    # Box 4: Bertumpuk di atas Box 1 (z = 2..4) -> TIDAK overlap (hanya menempel di atap)
    b4_stacked = Package("B4", w=2, l=2, h=2, value=10, weight=5)
    b4_stacked.set_position(0, 0, 2)
    assert is_overlapping_3d(b1, b4_stacked) is False


def test_support_and_floating():
    # Paket di lantai z = 0 selalu punya support
    p_floor = Package("Floor", w=4, l=4, h=2, value=20, weight=10)
    p_floor.set_position(0, 0, 0)
    assert is_supported(p_floor, [p_floor]) is True

    # Paket di atas paket lantai (z = 2) dengan alas tertumpu penuh
    p_top = Package("Top", w=2, l=2, h=2, value=15, weight=5)
    p_top.set_position(1, 1, 2)
    assert is_supported(p_top, [p_floor, p_top]) is True

    # Paket sebagian menggantung tapi masih ada irisan tumpuan -> VALID menurut spek
    p_overhang = Package("Overhang", w=4, l=4, h=2, value=15, weight=5)
    p_overhang.set_position(2, 2, 2)  # irisan dengan p_floor di [2,4] x [2,4]
    assert is_supported(p_overhang, [p_floor, p_overhang]) is True

    # Paket melayang di udara (z = 2 tanpa paket di bawah)
    p_float = Package("Float", w=2, l=2, h=2, value=10, weight=5)
    p_float.set_position(6, 6, 2)
    assert is_supported(p_float, [p_floor, p_float]) is False

    # Paket dengan celah vertikal (z_floor=0..2, p_gap di z=3) -> melayang
    p_gap = Package("Gap", w=2, l=2, h=2, value=10, weight=5)
    p_gap.set_position(0, 0, 3)
    assert is_supported(p_gap, [p_floor, p_gap]) is False


def test_fragile_constraint():
    # Paket pecah-belah di lantai
    p_fragile = Package("Fragile", w=4, l=4, h=2, value=50, weight=10, is_fragile=True)
    p_fragile.set_position(0, 0, 0)

    # Paket non-pecah-belah
    p_heavy = Package("Heavy", w=2, l=2, h=2, value=30, weight=20, is_fragile=False)

    # Kasus 1: Paket diletakkan di atas barang pecah-belah -> VIOLATION
    p_heavy.set_position(1, 1, 2)
    loaded_violated = [p_fragile, p_heavy]
    assert is_fragile_violated(loaded_violated) is True

    # Kasus 2: Barang pecah-belah diletakkan di atas barang biasa -> VALID (Boleh)
    p_heavy.set_position(0, 0, 0)
    p_fragile.set_position(1, 1, 2)
    loaded_allowed = [p_heavy, p_fragile]
    assert is_fragile_violated(loaded_allowed) is False

    # Kasus 3: Bersebelahan tanpa bertumpuk -> VALID (Boleh)
    p_fragile.set_position(0, 0, 0)
    p_heavy.set_position(5, 0, 0)
    assert is_fragile_violated([p_fragile, p_heavy]) is False


def test_capacity_constraint(truck):
    # Kapasitas truk = 100
    p1 = Package("P1", w=2, l=2, h=2, value=10, weight=60)
    p2 = Package("P2", w=2, l=2, h=2, value=10, weight=35)
    p3 = Package("P3", w=2, l=2, h=2, value=10, weight=15)

    assert is_capacity_exceeded([p1, p2], truck) is False  # 60 + 35 = 95 <= 100
    assert is_capacity_exceeded([p1, p2, p3], truck) is True  # 95 + 15 = 110 > 100


def test_check_all_constraints_integration(truck):
    # Konfigurasi valid:
    # B1 di lantai [0,4] x [0,4] x [0,2] (non-fragile)
    # B2 bertumpu di atas B1 [0,2] x [0,2] x [2,4] (fragile)
    b1 = Package("B1", w=4, l=4, h=2, value=100, weight=20, is_fragile=False)
    b1.set_position(0, 0, 0)

    b2 = Package("B2", w=2, l=2, h=2, value=80, weight=15, is_fragile=True)
    b2.set_position(0, 0, 2)

    valid, errors = check_all_constraints([b1, b2], truck)
    assert valid is True
    assert len(errors) == 0

    # Jadikan b1 melayang di z=1
    b1.set_position(0, 0, 1)
    valid_float, errors_float = check_all_constraints([b1, b2], truck)
    assert valid_float is False
    assert any("melayang" in err for err in errors_float)
