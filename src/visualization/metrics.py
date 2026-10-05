from pathlib import Path
from typing import List, Union, Dict, Any, Optional
import matplotlib
# pakai backend non-GUI 'Agg'
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.algorithms.base import SearchResult

# warna buat matplotlib
RUN_COLORS = ["royalblue", "crimson", "forestgreen", "darkorange", "purple", "teal"]

# fungsi untuk memvisualisasikan kurva konvergensi nilai objektif vs iterasi
# x = iterasi, y = nilai objektif (total value muatan atau bonus objective)
def plot_objective_history(
    results: Union[SearchResult, List[SearchResult]],
    title: str = "Konvergensi Nilai Objektif vs Banyak Iterasi",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:

    # buat ke dalam list jika input berupa objek SearchResult tunggal
    if isinstance(results, SearchResult):
        results = [results]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Loop setiap hasil pengujian dan plot kurvanya
    for idx, res in enumerate(results):
        scores = res.score_history
        
        # Sinkronisasi sumbu X, gunakan iteration_history jika ada, atau buat range indeks
        if res.iteration_history and len(res.iteration_history) == len(scores):
            iters = res.iteration_history
        else:
            iters = list(range(len(scores)))

        # color picker
        color = RUN_COLORS[idx % len(RUN_COLORS)]
        label = (
            f"Run {idx + 1} (Akhir: {res.final_score:.1f}, {res.iterations} iter)"
            if len(results) > 1
            else f"{res.algorithm_name} (Akhir: {res.final_score:.1f})"
        )
        ax.plot(iters, scores, label=label, color=color, linewidth=1.8)

    # konfig tata letak dan anotasi grafik
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Banyak Iterasi", fontsize=10)
    ax.set_ylabel("Nilai Fungsi Objektif", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()

    # Simpan ke disk jika path ditentukan
    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300)

    # Tampilkan jika diminta, atau tutup figure untuk menghemat memori
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

# fungsi untuk memvisualisasikan kurva probabilitas penerimaan solusi buruk exp(delta E / T) vs iterasi
# x = iterasi, y = probabilitas penerimaan solusi buruk
def plot_sa_acceptance(
    extra_metrics: Dict[str, Any],
    title: str = "Simulated Annealing: Probabilitas exp(ΔE/T) vs Iterasi",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> Optional[plt.Figure]:
  
    # ambil riwayat probabilitas
    probs = extra_metrics.get("prob_history", extra_metrics.get("acceptance_probabilities", extra_metrics.get("exp_delta_history", [])))
    if not probs:
        return None

    # ambil nomor iterasi saat probabilitas dievaluasi
    iters = extra_metrics.get("prob_iterations")
    if not iters or len(iters) != len(probs):
        iters = list(range(1, len(probs) + 1))

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Plot nilai probabilitas exp(dE/T)
    ax.plot(iters, probs, color="darkorange", linewidth=1.5, alpha=0.85, label="exp(ΔE / T)")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Iterasi", fontsize=10)
    ax.set_ylabel("Probabilitas Penerimaan", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    fig.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

# fungsi untuk memvisualisasikan kurva fitness maksimum vs rata-rata populasi per generasi (khusus GA)
# x = generasi, y1 = fitness maksimum, y2 = fitness rata-rata populasi
# garis biru navy = fitness individu terbaik dalam populasi pada generasi ke-g
# garis teal putus-putus = rata-rata fitness seluruh individu dalam populasi
def plot_ga_fitness(
    max_fitness: List[float],
    avg_fitness: List[float],
    title: str = "Genetic Algorithm: Fitness Maksimum vs Rata-rata Populasi",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    
    fig, ax = plt.subplots(figsize=(8, 5))
    generations = list(range(len(max_fitness)))

    # garis fitness maksimum
    ax.plot(generations, max_fitness, color="navy", linewidth=2.0, label="Fitness Maksimum")

    # garis fitness rata-rata
    ax.plot(generations, avg_fitness, color="teal", linewidth=1.8, linestyle="--", label="Fitness Rata-rata")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Generasi (Iterasi)", fontsize=10)
    ax.set_ylabel("Nilai Fitness", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")
    fig.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
