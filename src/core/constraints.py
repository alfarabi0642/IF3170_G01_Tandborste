from typing import List, Tuple
from .models import Package, Truck


def is_within_bounds(pkg: Package, truck: Truck) -> bool:
    """
    Memeriksa apakah seluruh bagian paket berada di dalam peti kemas truk.
    Koordinat berupa bilangan bulat dan tidak boleh menembus dinding peti kemas.
    """
    if pkg.x < 0 or pkg.y < 0 or pkg.z < 0:
        return False
    w, l, h = pkg.current_dims()
    if (pkg.x + w > truck.w) or (pkg.y + l > truck.l) or (pkg.z + h > truck.h):
        return False
    return True


def is_overlapping_3d(pkg1: Package, pkg2: Package) -> bool:
    """
    Memeriksa tabrakan 3D (AABB intersection) antara dua paket.
    Jika paket hanya saling bersentuhan di permukaan (touching faces), tidak dianggap overlap.
    """
    w1, l1, h1 = pkg1.current_dims()
    w2, l2, h2 = pkg2.current_dims()

    # Jika terpisah pada salah satu sumbu, maka tidak bertabrakan
    if pkg1.x + w1 <= pkg2.x or pkg2.x + w2 <= pkg1.x:
        return False
    if pkg1.y + l1 <= pkg2.y or pkg2.y + l2 <= pkg1.y:
        return False
    if pkg1.z + h1 <= pkg2.z or pkg2.z + h2 <= pkg1.z:
        return False

    return True


def has_horizontal_overlap(pkg1: Package, pkg2: Package) -> bool:
    """
    Memeriksa apakah terdapat irisan proyeksi horizontal (bidang x-y) antara dua paket.
    """
    w1, l1, _ = pkg1.current_dims()
    w2, l2, _ = pkg2.current_dims()

    overlap_x = max(0, min(pkg1.x + w1, pkg2.x + w2) - max(pkg1.x, pkg2.x))
    overlap_y = max(0, min(pkg1.y + l1, pkg2.y + l2) - max(pkg1.y, pkg2.y))

    return overlap_x > 0 and overlap_y > 0


def is_supported(pkg: Package, loaded_packages: List[Package]) -> bool:
    """
    Memeriksa apakah alas paket memiliki tumpuan (tidak melayang).
    Sesuai spesifikasi:
    - Sebuah titik di alas memiliki support jika:
      1. z = 0 (menempel langsung di alas truk), ATAU
      2. Terdapat paket lain j yang memiliki bagian atas tepat di bawah titik tersebut.
    - Melayang didefinisikan sebagai kondisi tidak ada satupun titik alas yang memiliki support.
    Artinya, jika minimal ada 1 titik integer yang memiliki tumpuan, paket dianggap valid.
    """
    if pkg.z == 0:
        return True  # Bertumpu langsung pada lantai peti kemas truk

    _, _, _ = pkg.current_dims()

    for other in loaded_packages:
        if other.id == pkg.id:
            continue
        _, _, other_h = other.current_dims()
        # Permukaan atas 'other' harus tepat di bawah elevasi alas 'pkg'
        if other.z + other_h == pkg.z:
            if has_horizontal_overlap(pkg, other):
                return True

    return False


def is_fragile_violated(loaded_packages: List[Package]) -> bool:
    """
    Memeriksa kendala barang pecah-belah (is_fragile).
    Sesuai spesifikasi: Barang pecah-belah tidak boleh berada di bawah paket lain
    (tidak boleh menjadi support bagi paket lain).
    """
    for fragile_pkg in loaded_packages:
        if not fragile_pkg.is_fragile:
            continue
        _, _, fh = fragile_pkg.current_dims()
        fragile_top = fragile_pkg.z + fh

        for other in loaded_packages:
            if other.id == fragile_pkg.id:
                continue
            # Jika ada paket lain yang menumpuk tepat di atas barang pecah belah
            if other.z == fragile_top:
                if has_horizontal_overlap(fragile_pkg, other):
                    return True  # Pelanggaran: ada paket di atas barang pecah belah

    return False


def is_capacity_exceeded(loaded_packages: List[Package], truck: Truck) -> bool:
    """
    Memeriksa apakah total berat paket melebihi kapasitas angkut maksimum truk.
    """
    total_weight = sum(p.weight for p in loaded_packages)
    return total_weight > truck.max_capacity


def check_all_constraints(loaded_packages: List[Package], truck: Truck) -> Tuple[bool, List[str]]:
    """
    Memeriksa keseluruhan batasan fisika dari konfigurasi paket di dalam truk.
    Mengembalikan (is_valid, list_error_messages).
    """
    errors: List[str] = []

    # 1. Batas kapasitas muatan
    total_weight = sum(p.weight for p in loaded_packages)
    if total_weight > truck.max_capacity:
        errors.append(f"Muatan melebihi kapasitas: {total_weight} > {truck.max_capacity}")

    # 2. Batas dinding peti kemas
    for p in loaded_packages:
        if not is_within_bounds(p, truck):
            errors.append(f"Paket {p.id} menembus dinding truk di posisi ({p.x}, {p.y}, {p.z})")

    # 3. Tabrakan antarpaket (overlap 3D)
    n = len(loaded_packages)
    for i in range(n):
        for j in range(i + 1, n):
            if is_overlapping_3d(loaded_packages[i], loaded_packages[j]):
                errors.append(f"Tabrakan antara paket {loaded_packages[i].id} dan {loaded_packages[j].id}")

    # 4. Kardus melayang (support alas)
    for p in loaded_packages:
        if not is_supported(p, loaded_packages):
            errors.append(f"Paket {p.id} melayang di ketinggian z={p.z}")

    # 5. Barang pecah-belah ditumpuk
    if is_fragile_violated(loaded_packages):
        errors.append("Barang pecah-belah menjadi tumpuan paket lain")

    is_valid = len(errors) == 0
    return is_valid, errors
