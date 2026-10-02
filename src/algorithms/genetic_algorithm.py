import random
import time
from typing import Callable, Dict, List, Tuple

from src.core.models import Package, Truck, ORIENTATIONS
from src.core.constraints import (
    is_within_bounds,
    is_overlapping_3d,
    has_horizontal_overlap,
    is_supported,
)
from src.core.objectives import default_objective
from src.core.state import State
from .base import SearchResult

# Satu gen = (indeks paket, indeks orientasi).
# Kromosom = urutan pemuatan seluruh paket beserta orientasinya.
Gene = Tuple[int, int]
Chromosome = List[Gene]


# =============================================================================
# DECODING KROMOSOM -> STATE
# Paket dimuat satu per satu sesuai urutan kromosom ke titik kandidat pertama
# yang valid (urut z, y, x -> heuristik bottom-left-back). Paket yang tidak muat
# dibiarkan di luar truk, sehingga hasil decode selalu valid secara fisika.
# =============================================================================
def _can_place(pkg: Package, loaded: List[Package], truck: Truck, loaded_weight: float) -> bool:
    """Cek kendala hanya untuk paket baru terhadap paket yang sudah dimuat (lebih cepat dari cek penuh)."""
    if loaded_weight + pkg.weight > truck.max_capacity:
        return False
    if not is_within_bounds(pkg, truck):
        return False
    for other in loaded:
        if is_overlapping_3d(pkg, other):
            return False
    if not is_supported(pkg, loaded):
        return False

    _, _, h = pkg.current_dims()
    for other in loaded:
        _, _, oh = other.current_dims()
        # Paket baru tidak boleh bertumpu di atas barang pecah-belah
        if other.is_fragile and other.z + oh == pkg.z and has_horizontal_overlap(pkg, other):
            return False
        # Jika paket baru pecah-belah, tidak boleh ada paket yang sudah bertumpu di atasnya
        if pkg.is_fragile and other.z == pkg.z + h and has_horizontal_overlap(pkg, other):
            return False
    return True


def _new_anchors(pkg: Package, truck: Truck) -> List[Tuple[int, int, int]]:
    """Titik kandidat baru di sisi kanan, sisi belakang, dan atas paket yang baru dimuat."""
    w, l, h = pkg.current_dims()
    points = [(pkg.x + w, pkg.y, pkg.z), (pkg.x, pkg.y + l, pkg.z)]
    if not pkg.is_fragile:
        points.append((pkg.x, pkg.y, pkg.z + h))
    return [(x, y, z) for x, y, z in points if x < truck.w and y < truck.l and z < truck.h]


def decode(
    chromosome: Chromosome,
    truck: Truck,
    packages: List[Package],
    objective_fn: Callable[[State], float] = default_objective,
) -> State:
    """Mengubah kromosom menjadi State yang valid secara fisika."""
    pkgs = [p.clone() for p in packages]
    for p in pkgs:
        p.is_loaded = False
        p.set_position(0, 0, 0)

    loaded: List[Package] = []
    loaded_weight = 0.0
    anchors = {(0, 0, 0)}

    for idx, ori in chromosome:
        pkg = pkgs[idx]
        pkg.set_orientation(ori)
        for x, y, z in sorted(anchors, key=lambda a: (a[2], a[1], a[0])):
            pkg.set_position(x, y, z)
            if _can_place(pkg, loaded, truck, loaded_weight):
                pkg.is_loaded = True
                loaded.append(pkg)
                loaded_weight += pkg.weight
                anchors.discard((x, y, z))
                anchors.update(_new_anchors(pkg, truck))
                break
        if not pkg.is_loaded:
            pkg.set_position(0, 0, 0)

    return State(truck, pkgs, objective_fn)


# =============================================================================
# OPERATOR GENETIKA (Salindia Kuliah IF3170 - Beyond Classical Search, hal. 42-49)
# =============================================================================
def random_chromosome(n: int) -> Chromosome:
    order = list(range(n))
    random.shuffle(order)
    return [(i, random.randrange(len(ORIENTATIONS))) for i in order]


def roulette_select(population: List[Chromosome], fitness: List[float]) -> Chromosome:
    """Seleksi dengan peluang terpilih sebanding dengan nilai fitness."""
    if sum(fitness) <= 0:
        return random.choice(population)
    return random.choices(population, weights=fitness, k=1)[0]


def order_crossover(parent1: Chromosome, parent2: Chromosome) -> Chromosome:
    """
    Order Crossover (OX): salin satu segmen dari parent1, sisanya diisi sesuai urutan
    paket di parent2. Setiap paket tetap muncul tepat satu kali pada anak.
    """
    n = len(parent1)
    if n < 2:
        return list(parent1)

    a, b = sorted(random.sample(range(n), 2))
    child: List[Gene] = [None] * n  # type: ignore
    child[a:b + 1] = parent1[a:b + 1]
    taken = {idx for idx, _ in parent1[a:b + 1]}

    rest = [g for g in parent2 if g[0] not in taken]
    empty_slots = [i for i in range(n) if i < a or i > b]
    for i, gene in zip(empty_slots, rest):
        child[i] = gene
    return child


def mutate(chromosome: Chromosome, mutation_rate: float) -> Chromosome:
    """Setiap gen bermutasi dengan peluang mutation_rate: tukar urutan atau ganti orientasi."""
    child = list(chromosome)
    n = len(child)
    for i in range(n):
        if random.random() >= mutation_rate:
            continue
        if random.random() < 0.5 and n > 1:
            j = random.randrange(n)
            child[i], child[j] = child[j], child[i]
        else:
            idx, _ = child[i]
            child[i] = (idx, random.randrange(len(ORIENTATIONS)))
    return child


# =============================================================================
# GENETIC ALGORITHM
# =============================================================================
def genetic_algorithm(
    truck: Truck,
    packages: List[Package],
    pop_size: int = 30,
    generations: int = 50,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.1,
    elite: int = 2,
    objective_fn: Callable[[State], float] = default_objective,
) -> SearchResult:
    """
    Genetic Algorithm (Salindia Kuliah IF3170 - Beyond Classical Search, hal. 42-49).
    - Populasi awal: kromosom acak (urutan dan orientasi acak).
    - Fitness: nilai fungsi objektif dari state hasil decode kromosom.
    - Seleksi roulette wheel, order crossover, mutasi, dan elitism.
    - Berhenti setelah jumlah generasi (iterasi) mencapai `generations`.
    """
    if pop_size < 2:
        raise ValueError("pop_size minimal 2")
    elite = max(0, min(elite, pop_size))

    start_time = time.time()
    n = len(packages)

    # Cache fitness agar kromosom yang sama (misalnya elite) tidak di-decode ulang
    cache: Dict[Tuple[Gene, ...], float] = {}

    def fitness_of(chrom: Chromosome) -> float:
        key = tuple(chrom)
        if key not in cache:
            cache[key] = float(objective_fn(decode(chrom, truck, packages, objective_fn)))
        return cache[key]

    population = [random_chromosome(n) for _ in range(pop_size)]
    fitness = [fitness_of(c) for c in population]

    best_idx = max(range(pop_size), key=lambda i: fitness[i])
    initial_chrom = population[best_idx]
    best_chrom = initial_chrom
    best_fit = fitness[best_idx]

    max_history = [max(fitness)]
    avg_history = [sum(fitness) / pop_size]
    iteration_history = [0]

    for gen in range(1, generations + 1):
        ranked = sorted(range(pop_size), key=lambda i: fitness[i], reverse=True)
        new_pop = [population[i] for i in ranked[:elite]]

        while len(new_pop) < pop_size:
            p1 = roulette_select(population, fitness)
            p2 = roulette_select(population, fitness)
            if random.random() < crossover_rate:
                child = order_crossover(p1, p2)
            else:
                child = list(p1)
            new_pop.append(mutate(child, mutation_rate))

        population = new_pop
        fitness = [fitness_of(c) for c in population]

        gen_best = max(range(pop_size), key=lambda i: fitness[i])
        if fitness[gen_best] > best_fit:
            best_fit = fitness[gen_best]
            best_chrom = population[gen_best]

        max_history.append(max(fitness))
        avg_history.append(sum(fitness) / pop_size)
        iteration_history.append(gen)

    duration = time.time() - start_time
    initial_state = decode(initial_chrom, truck, packages, objective_fn)
    final_state = decode(best_chrom, truck, packages, objective_fn)

    return SearchResult(
        algorithm_name="Genetic Algorithm",
        initial_state=initial_state,
        final_state=final_state,
        initial_score=initial_state.get_score(),
        final_score=final_state.get_score(),
        iterations=generations,
        duration_seconds=duration,
        score_history=list(max_history),
        iteration_history=iteration_history,
        extra_metrics={
            "pop_size": pop_size,
            "generations": generations,
            "crossover_rate": crossover_rate,
            "mutation_rate": mutation_rate,
            "elite": elite,
            "max_fitness_history": max_history,
            "avg_fitness_history": avg_history,
            "best_chromosome": [list(g) for g in best_chrom],
        },
    )
