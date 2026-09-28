from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import State


def default_objective(state: "State") -> float:
    """
    Fungsi objektif default (Spesifikasi Bagian Objective Function):
    Memaksimalkan total value dari paket-paket yang berhasil dimuat ke dalam truk.
    """
    return sum(p.value for p in state.loaded_packages)


def bonus_urgency_volume_objective(state: "State") -> float:
    """
    Fungsi objektif alternatif (Bonus 1 - 3 Poin):
    Mengoptimalkan nilai muatan berdasarkan:
    1. Urgensi pengiriman (ETA sisa hari lebih kecil -> bobot nilai lebih tinggi).
    2. Efisiensi utilisasi ruang (rasio volume paket terhadap volume truk)
       untuk meminimalkan ruang kosong tanpa menambah variabel baru.
    """
    if not state.loaded_packages:
        return 0.0

    # Komponen 1: Nilai paket terbobot urgensi ETA
    weighted_value = sum(p.value / (p.eta + 1.0) for p in state.loaded_packages)

    # Komponen 2: Rasio kepadatan ruang (Volume Utilization)
    truck_vol = state.truck.volume()
    loaded_vol = sum(p.volume() for p in state.loaded_packages)
    volume_ratio = loaded_vol / truck_vol if truck_vol > 0 else 0.0

    # Bobot pengali volume_ratio disesuaikan agar seimbang dengan rentang nilai
    return weighted_value + (20.0 * volume_ratio)
