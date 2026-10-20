import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from src.core.state import State


@dataclass
class SearchResult:
    """
    Struktur data untuk menyimpan hasil eksekusi algoritma local search.
    - State awal dan akhir
    - Nilai objective function akhir
    - Plot/riwayat nilai objective terhadap banyak iterasi
    - Durasi proses pencarian
    """
    algorithm_name: str
    initial_state: State
    final_state: State
    initial_score: float
    final_score: float
    iterations: int
    duration_seconds: float
    score_history: List[float] = field(default_factory=list)
    iteration_history: List[int] = field(default_factory=list)
    extra_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Konversi hasil pencarian ke dictionary JSON untuk disimpan ke folder output/."""
        return {
            "algorithm": self.algorithm_name,
            "duration_seconds": round(self.duration_seconds, 4),
            "iterations": self.iterations,
            "initial_score": round(self.initial_score, 2),
            "final_score": round(self.final_score, 2),
            "loaded_packages_count": len(self.final_state.loaded_packages),
            "unloaded_packages_count": len(self.final_state.unloaded_packages),
            "total_loaded_weight": round(self.final_state.total_loaded_weight(), 2),
            "truck_capacity": self.final_state.truck.max_capacity,
            "score_history": [round(s, 2) for s in self.score_history],
            "iteration_history": self.iteration_history,
            "extra_metrics": self.extra_metrics,
            "initial_state": self.initial_state.to_dict(),
            "final_state": self.final_state.to_dict(),
        }
