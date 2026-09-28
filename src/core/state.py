import random
from typing import List, Tuple, Callable, Optional, Dict, Any
from .models import Package, Truck, ORIENTATIONS
from .constraints import check_all_constraints
from .objectives import default_objective


class State:
    """
    Representasi status (state) konfigurasi pemuatan seluruh paket ke dalam truk.
    Menyimpan pemetaan paket di dalam peti kemas (loaded) dan di luar (unloaded).
    """

    def __init__(
        self,
        truck: Truck,
        packages: List[Package],
        objective_fn: Callable[["State"], float] = default_objective,
    ):
        self.truck = truck
        self.packages = packages
        self.objective_fn = objective_fn

    @property
    def loaded_packages(self) -> List[Package]:
        """Daftar paket yang berada di dalam peti kemas truk."""
        return [p for p in self.packages if p.is_loaded]

    @property
    def unloaded_packages(self) -> List[Package]:
        """Daftar paket yang berada di luar truk."""
        return [p for p in self.packages if not p.is_loaded]

    def is_valid(self) -> bool:
        """Memeriksa apakah seluruh kendala fisika terpenuhi."""
        valid, _ = check_all_constraints(self.loaded_packages, self.truck)
        return valid

    def get_errors(self) -> List[str]:
        """Mengembalikan daftar pesan pelanggaran kendala fisika jika ada."""
        _, errors = check_all_constraints(self.loaded_packages, self.truck)
        return errors

    def get_score(self) -> float:
        """
        Menghitung nilai fungsi objektif pada state saat ini.
        Jika state tidak valid secara fisika, diberikan penalti nilai 0.
        """
        if not self.is_valid():
            return 0.0
        return float(self.objective_fn(self))

    def total_loaded_weight(self) -> float:
        return sum(p.weight for p in self.loaded_packages)

    def total_loaded_volume(self) -> int:
        return sum(p.volume() for p in self.loaded_packages)

    def clone(self) -> "State":
        """Membuat salinan mendalam (deep copy) dari state."""
        cloned_packages = [p.clone() for p in self.packages]
        return State(
            truck=self.truck,
            packages=cloned_packages,
            objective_fn=self.objective_fn,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialisasi state ke format dictionary JSON."""
        return {
            "truck": self.truck.to_dict(),
            "score": self.get_score(),
            "is_valid": self.is_valid(),
            "total_weight": self.total_loaded_weight(),
            "loaded_count": len(self.loaded_packages),
            "unloaded_count": len(self.unloaded_packages),
            "packages": [p.to_dict() for p in self.packages],
        }

    # =========================================================================
    # GENERATOR INITIAL STATE RANDOM
    # =========================================================================
    @classmethod
    def create_random_state(
        cls,
        truck: Truck,
        packages: List[Package],
        objective_fn: Callable[["State"], float] = default_objective,
    ) -> "State":
        """
        Membangkitkan initial state acak yang valid secara fisika.
        Sesuai spesifikasi: 'Inisialisasi (initial state) random.'
        """
        cloned_packages = [p.clone() for p in packages]
        # Inisialisasi awal: semua paket di luar truk
        for p in cloned_packages:
            p.is_loaded = False
            p.set_position(0, 0, 0)
            p.set_orientation(random.randint(0, len(ORIENTATIONS) - 1))

        state = cls(truck, cloned_packages, objective_fn)

        # Coba muat paket secara acak satu per satu
        pkg_pool = list(cloned_packages)
        random.shuffle(pkg_pool)

        for pkg in pkg_pool:
            if random.random() < 0.8:  # 80% peluang mencoba muat
                candidate_positions = _get_candidate_anchors(state)
                random.shuffle(candidate_positions)

                placed = False
                # Coba beberapa orientasi acak
                orientations = list(range(len(ORIENTATIONS)))
                random.shuffle(orientations)

                for ori in orientations:
                    pkg.set_orientation(ori)
                    for x, y, z in candidate_positions:
                        pkg.set_position(x, y, z)
                        pkg.is_loaded = True

                        if state.is_valid():
                            placed = True
                            break
                        else:
                            pkg.is_loaded = False

                    if placed:
                        break

        return state


# =============================================================================
# HELPER: CANDIDATE ANCHOR POSITIONS (HEURISTIK TITIK TUMPUAN)
# =============================================================================
def _get_candidate_anchors(state: State) -> List[Tuple[int, int, int]]:
    """
    Menghasilkan kandidat koordinat penempatan yang berpotensi valid:
    1. Titik asal lantai truk (0, 0, 0)
    2. Menempel pada sisi samping paket lantai lain
    3. Di atas paket lain yang bukan pecah-belah (is_fragile=False)
    """
    anchors = [(0, 0, 0)]
    truck = state.truck

    for p in state.loaded_packages:
        w, l, h = p.current_dims()

        # Posisi di samping paket (hanya jika di lantai z=0)
        if p.z == 0:
            if p.x + w < truck.w:
                anchors.append((p.x + w, p.y, 0))
            if p.y + l < truck.l:
                anchors.append((p.x, p.y + l, 0))

        # Posisi di atas paket (hanya jika paket di bawah bukan barang pecah-belah)
        if not p.is_fragile:
            if p.z + h < truck.h:
                anchors.append((p.x, p.y, p.z + h))

    # Tambahkan beberapa titik acak di lantai untuk variasi stokastik
    for _ in range(3):
        rx = random.randint(0, max(0, truck.w - 1))
        ry = random.randint(0, max(0, truck.l - 1))
        anchors.append((rx, ry, 0))

    return list(set(anchors))


# =============================================================================
# OPERATOR TETANGGA (NEIGHBOR OPERATORS)
# Sesuai spesifikasi:
# 1. Menukar dua paket, baik di dalam maupun di luar truk (Swap)
# 2. Memindahkan sebuah paket ke koordinat yang berbeda (Move)
# 3. Merotasi paket 90 derajat terhadap sumbu x, y, atau z (Rotate)
# =============================================================================

def neighbor_move(state: State) -> Optional[State]:
    """
    Operator Move:
    - Memindahkan paket di dalam truk ke koordinat berbeda, ATAU
    - Memasukkan paket dari luar ke dalam truk, ATAU
    - Mengeluarkan paket dari truk ke luar.
    """
    new_state = state.clone()
    if not new_state.packages:
        return None

    # Tentukan aksi move: memuat paket baru, memindahkan posisi, atau membongkar
    loaded = new_state.loaded_packages
    unloaded = new_state.unloaded_packages

    candidates = _get_candidate_anchors(new_state)
    random.shuffle(candidates)

    choice = random.choice(["move_loaded", "load_new", "unload"])

    if choice == "load_new" and unloaded:
        pkg = random.choice(unloaded)
        for ori in random.sample(range(len(ORIENTATIONS)), len(ORIENTATIONS)):
            pkg.set_orientation(ori)
            for x, y, z in candidates:
                pkg.set_position(x, y, z)
                pkg.is_loaded = True
                if new_state.is_valid():
                    return new_state
                pkg.is_loaded = False

    elif choice == "move_loaded" and loaded:
        pkg = random.choice(loaded)
        orig_pos = (pkg.x, pkg.y, pkg.z)
        pkg.is_loaded = False  # Lepas sementara agar tidak tabrakan dengan posisi lama
        for x, y, z in candidates:
            if (x, y, z) == orig_pos:
                continue
            pkg.set_position(x, y, z)
            pkg.is_loaded = True
            if new_state.is_valid():
                return new_state
            pkg.is_loaded = False
        # Kembalikan jika gagal
        pkg.set_position(*orig_pos)
        pkg.is_loaded = True

    elif choice == "unload" and loaded:
        pkg = random.choice(loaded)
        pkg.is_loaded = False
        # Membongkar kardus mungkin membuat kardus di atasnya melayang, pastikan valid
        if new_state.is_valid():
            return new_state

    return None


def neighbor_swap(state: State) -> Optional[State]:
    """
    Operator Swap:
    - Menukar posisi dan status dua paket (dalam-dalam, atau dalam-luar).
    """
    new_state = state.clone()
    if len(new_state.packages) < 2:
        return None

    p1, p2 = random.sample(new_state.packages, 2)

    # Kasus 1: Keduanya di dalam truk -> Tukar posisi koordinat
    if p1.is_loaded and p2.is_loaded:
        pos1 = (p1.x, p1.y, p1.z)
        pos2 = (p2.x, p2.y, p2.z)
        p1.set_position(*pos2)
        p2.set_position(*pos1)
        if new_state.is_valid():
            return new_state

    # Kasus 2: Satu di dalam, satu di luar -> Tukar posisi dan status loaded
    elif p1.is_loaded != p2.is_loaded:
        inside = p1 if p1.is_loaded else p2
        outside = p2 if p1.is_loaded else p1

        # Coba letakkan outside di posisi inside
        target_pos = (inside.x, inside.y, inside.z)
        inside.is_loaded = False
        outside.is_loaded = True
        outside.set_position(*target_pos)

        # Cek beberapa orientasi untuk outside
        for ori in random.sample(range(len(ORIENTATIONS)), len(ORIENTATIONS)):
            outside.set_orientation(ori)
            if new_state.is_valid():
                return new_state

    return None


def neighbor_rotate(state: State) -> Optional[State]:
    """
    Operator Rotate:
    - Merotasi orientasi salah satu paket di dalam truk sebesar 90 derajat.
    """
    new_state = state.clone()
    loaded = new_state.loaded_packages
    if not loaded:
        return None

    pkg = random.choice(loaded)
    curr_ori = pkg.orientation_idx
    other_orientations = [i for i in range(len(ORIENTATIONS)) if i != curr_ori]
    random.shuffle(other_orientations)

    for ori in other_orientations:
        pkg.set_orientation(ori)
        if new_state.is_valid():
            return new_state

    return None


def get_random_neighbor(state: State, max_tries: int = 25) -> Optional[State]:
    """
    Membangkitkan 1 tetangga acak yang valid secara fisika.
    Memilih secara acak antara operator Move, Swap, atau Rotate.
    """
    operators = [neighbor_move, neighbor_swap, neighbor_rotate]
    for _ in range(max_tries):
        op = random.choice(operators)
        neighbor = op(state)
        if neighbor is not None and neighbor.is_valid():
            return neighbor
    return None


def get_all_neighbors(state: State, sample_limit: int = 30) -> List[State]:
    """
    Menghasilkan himpunan tetangga (sample neighbors) untuk evaluasi Steepest Ascent.
    """
    neighbors: List[State] = []
    seen_signatures = set()

    for _ in range(sample_limit * 2):
        nb = get_random_neighbor(state, max_tries=5)
        if nb is not None:
            # Gunakan representasi konfigurasi posisi sebagai identifikasi unik
            sig = tuple(
                (p.id, p.is_loaded, p.x, p.y, p.z, p.orientation_idx)
                for p in sorted(nb.packages, key=lambda x: x.id)
            )
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                neighbors.append(nb)
                if len(neighbors) >= sample_limit:
                    break

    return neighbors
