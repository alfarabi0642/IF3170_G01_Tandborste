import argparse
import sys
import time
from pathlib import Path

from src.utils.io_handler import load_input, save_run_result
from src.utils.experiment_runner import (
    run_single_search,
    run_experiment,
    run_ga_parameter_study,
    ALGORITHM_MAP,
)
from src.core.objectives import default_objective, bonus_urgency_volume_objective
from src.visualization.plot3d import visualize_comparison_3d, visualize_state_3d
from src.visualization.metrics import plot_objective_history, plot_sa_acceptance


def parse_args(args=None):
    """
    Menyusun dan mem-parsing parameter Command Line Interface (CLI) aplikasi.
    Mendukung konfigurasi kasus uji, algoritma, parameter pencarian,
    mode eksperimen otomatis, serta opsi visualisasi.
    """
    parser = argparse.ArgumentParser(
        description="Mejiro Express: 3D Container Loading Problem Solver (Local Search Suite)"
    )

    # -------------------------------------------------------------------------
    # 1. Konfigurasi Input Data & Algoritma
    # -------------------------------------------------------------------------
    parser.add_argument(
        "-i", "--input",
        type=str,
        default="test_cases/small.json",
        help="Path ke file input JSON kasus uji (default: test_cases/small.json)",
    )
    parser.add_argument(
        "-a", "--algo",
        type=str,
        default="steepest",
        choices=sorted(list(set(ALGORITHM_MAP.keys()))),
        help="Pilihan algoritma local search (default: steepest)",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="default",
        choices=["default", "bonus"],
        help="Fungsi objektif: 'default' (total nilai) atau 'bonus' (urgensi ETA + rasio volume)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed bilangan acak untuk menjamin keterulangan (reproducibility) eksperimen",
    )

    # -------------------------------------------------------------------------
    # 2. Parameter Hill-Climbing Suite (Driver 1)
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--max-iterations", "--max-iter",
        dest="max_iterations",
        type=int,
        default=300,
        help="Batas maksimum iterasi pencarian (default: 300)",
    )
    parser.add_argument(
        "--sample-neighbors",
        type=int,
        default=25,
        help="Jumlah tetangga yang disampel per iterasi pada Steepest/Sideways (default: 25)",
    )
    parser.add_argument(
        "--max-sideways",
        type=int,
        default=50,
        help="Batas maksimum pergerakan sideways berturut-turut pada Sideways Move (default: 50)",
    )
    parser.add_argument(
        "--restarts", "--max-restarts",
        dest="restarts",
        type=int,
        default=5,
        help="Batas restart untuk Random Restart Hill-Climbing (default: 5)",
    )
    parser.add_argument(
        "--iter-per-restart",
        type=int,
        default=100,
        help="Iterasi per restart untuk Random Restart HC (default: 100)",
    )

    # -------------------------------------------------------------------------
    # 3. Parameter Simulated Annealing & Genetic Algorithm (Driver 2)
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--t0",
        type=float,
        default=200.0,
        help="Suhu awal untuk Simulated Annealing (default: 200.0)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.95,
        help="Koefisien penurunan suhu geometrik T(k+1) = alpha * T(k) (default: 0.95)",
    )
    parser.add_argument(
        "--t-min",
        type=float,
        default=0.01,
        help="Suhu akhir terminasi Simulated Annealing (default: 0.01)",
    )
    parser.add_argument(
        "--stuck-limit",
        type=int,
        default=50,
        help="Ambang penolakan berturut-turut untuk counter stuck local optima SA (default: 50)",
    )
    parser.add_argument(
        "--iter-per-temp",
        type=int,
        default=10,
        help="Jumlah iterasi sampling per tingkat temperatur pada SA (default: 10)",
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=30,
        help="Ukuran populasi untuk Genetic Algorithm (default: 30)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Jumlah generasi iterasi untuk Genetic Algorithm (default: 50)",
    )

    # -------------------------------------------------------------------------
    # 4. Mode Eksperimen & Otomatisasi Batch
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--experiment",
        action="store_true",
        help="Menjalankan mode batch eksperimen berulang (default 3 kali run independen)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Jumlah pengujian independen dalam mode eksperimen (default: 3)",
    )
    parser.add_argument(
        "--ga-study",
        action="store_true",
        help="Menjalankan matriks pengujian lengkap GA (variasi populasi & iterasi)",
    )
    parser.add_argument(
        "--ga-iters",
        type=int,
        nargs="+",
        default=[20, 50, 100],
        help="Daftar variasi jumlah generasi/iterasi untuk GA study (default: 20 50 100)",
    )
    parser.add_argument(
        "--ga-pops",
        type=int,
        nargs="+",
        default=[15, 30, 60],
        help="Daftar variasi ukuran populasi untuk GA study (default: 15 30 60)",
    )
    parser.add_argument(
        "--ga-reps",
        type=int,
        default=3,
        help="Jumlah pengulangan/repetisi per variasi parameter pada GA study (default: 3)",
    )

    # -------------------------------------------------------------------------
    # 5. Konfigurasi Output & Visualisasi
    # -------------------------------------------------------------------------
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="output",
        help="Direktori penyimpanan file JSON hasil eksekusi dan grafik (default: output)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Membuka visualisasi 3D interaktif di browser setelah pencarian selesai",
    )
    parser.add_argument(
        "--save-plot",
        action="store_true",
        help="Menyimpan visualisasi 3D ke file HTML di folder output",
    )
    parser.add_argument(
        "--save-metrics",
        action="store_true",
        help="Menyimpan grafik konvergensi (PNG) di folder output",
    )

    return parser.parse_args(args)


def main(cli_args=None):
    """
    Fungsi utama program (CLI Entry Point).
    Mengorkestrasi input file, eksekusi search engine, pelaporan hasil terminal,
    penyimpanan file log JSON, serta pembuatan visualisasi 3D dan grafik metrik.
    """
    # 1. Parsing argumen baris perintah
    args = parse_args(cli_args)

    # 2. Baca dan validasi data kasus uji dari file JSON
    try:
        truck, packages = load_input(args.input)
    except Exception as e:
        print(f"[Error] Gagal membaca input: {e}", file=sys.stderr)
        return 1

    # Tentukan fungsi objektif (default sum of values, atau bonus urgency & volume ratio)
    obj_fn = bonus_urgency_volume_objective if args.objective == "bonus" else default_objective

    # Kumpulkan dictionary parameter algoritma
    algo_params = {
        "max_iterations": args.max_iterations,
        "sample_neighbors": args.sample_neighbors,
        "max_sideways": args.max_sideways,
        "max_restarts": args.restarts,
        "iterations_per_restart": args.iter_per_restart,
        "t0": args.t0,
        "alpha": args.alpha,
        "t_min": args.t_min,
        "stuck_limit": args.stuck_limit,
        "iter_per_temp": args.iter_per_temp,
        "pop_size": args.pop_size,
        "generations": args.generations,
    }

    # Buat direktori output jika belum ada
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    # eksekusi langsung GA 18 kali, untuk spek
    if args.ga_study:
        try:
            run_ga_parameter_study(
                truck=truck,
                packages=packages,
                output_dir=out_dir,
                save_plots=args.save_metrics,
                objective_fn=obj_fn,
                study_reps=args.ga_reps,
                iteration_settings=args.ga_iters,
                fixed_population=args.pop_size,
                population_settings=args.ga_pops,
                fixed_iterations=args.generations,
            )
            return 0
        except NotImplementedError as e:
            print(f"[Info] {e}", file=sys.stderr)
            return 1

    # Mode 3x run independen untuk eksperimen
    if args.experiment:
        print(f"=== Menjalankan Eksperimen: {args.algo} ({args.runs} runs) ===")
        print(f"Kasus Uji: {args.input} | Objektif: {args.objective}")

        try:
            exp_data = run_experiment(
                algo_name=args.algo,
                truck=truck,
                packages=packages,
                num_runs=args.runs,
                objective_fn=obj_fn,
                output_dir=out_dir,
                save_plots=args.save_metrics,
                **algo_params,
            )
        except NotImplementedError as e:
            print(f"[Info] {e}", file=sys.stderr)
            return 1

        # output tabel ringkasan performa 
        summary = exp_data["summary"]
        print("\n--- Ringkasan Hasil Eksperimen ---")
        print(f"Algoritma          : {summary['algorithm']}")
        print(f"Total Pengujian    : {summary['total_runs']} runs")
        print(f"Skor per Run       : {summary['scores']}")
        print(f"Skor Rata-rata     : {summary['mean_score']:.2f}")
        print(f"Skor Terbaik       : {summary['best_score']:.2f}")
        print(f"Standar Deviasi    : {summary['std_score']:.2f}")
        print(f"Durasi Rata-rata   : {summary['mean_duration_seconds']:.4f} detik")
        print(f"Iterasi Rata-rata  : {summary['mean_iterations']:.1f}")
        if "plot_saved" in summary:
            print(f"Grafik Konvergensi : {summary['plot_saved']}")
        print(f"Hasil JSON lengkap tersimpan di folder: {out_dir}")
        return 0

    # mode single run
    print(f"=== Menjalankan Pencarian: {args.algo} ===")
    print(f"Kasus Uji: {args.input} | Objektif: {args.objective} | Seed: {args.seed}")

    try:
        result = run_single_search(
            algo_name=args.algo,
            truck=truck,
            packages=packages,
            objective_fn=obj_fn,
            seed=args.seed,
            **algo_params,
        )
    except NotImplementedError as e:
        print(f"[Info] {e}", file=sys.stderr)
        return 1

    # Tampilkan ringkasan hasil pencarian ke terminal
    final_state = result.final_state
    print("\n--- Hasil Pencarian ---")
    print(f"Algoritma        : {result.algorithm_name}")
    print(f"Skor Awal        : {result.initial_score:.2f}")
    print(f"Skor Akhir       : {result.final_score:.2f}")
    print(f"Total Iterasi    : {result.iterations}")
    print(f"Durasi Komputasi : {result.duration_seconds:.4f} detik")
    print(f"Status Fisika    : {'Valid' if final_state.is_valid() else 'Tidak Valid'}")
    print(f"Paket Dimuat     : {len(final_state.loaded_packages)} dari {len(final_state.packages)} paket")
    print(f"Kapasitas Truk   : {final_state.total_loaded_weight():.1f} / {final_state.truck.max_capacity:.1f} kg")

    # Simpan hasil eksekusi dalam format JSON di folder output
    json_path = save_run_result(
        result,
        output_dir=out_dir,
        extra_info={"input_file": str(args.input), "objective": args.objective, "seed": args.seed},
    )
    print(f"\nHasil pencarian disimpan ke: {json_path}")

    # buat hasil grafik konvergensi jika diminta
    if args.save_metrics:
        plot_path = out_dir / f"{args.algo}_{ts}_convergence.png"
        plot_objective_history(
            results=result,
            title=f"Konvergensi {result.algorithm_name}",
            save_path=plot_path,
            show=False,
        )
        print(f"Grafik konvergensi disimpan ke: {plot_path}")

        # untuk SA, buat juga grafik probabilitas exp(delta E / T)
        if "prob_history" in result.extra_metrics:
            sa_plot_path = out_dir / f"{args.algo}_{ts}_sa_acceptance.png"
            plot_sa_acceptance(
                extra_metrics=result.extra_metrics,
                title=f"{result.algorithm_name}: Probabilitas Penerimaan exp(ΔE/T)",
                save_path=sa_plot_path,
                show=False,
            )
            print(f"Grafik probabilitas SA disimpan ke: {sa_plot_path}")

    # visualisasi 
    if args.save_plot or args.plot:
        plot_3d_path = out_dir / f"{args.algo}_{ts}_3d.html" if args.save_plot else None
        visualize_comparison_3d(
            initial_state=result.initial_state,
            final_state=result.final_state,
            title=f"{result.algorithm_name}: Perbandingan Solusi Awal vs Akhir",
            show=args.plot,
            save_path=plot_3d_path,
        )
        if plot_3d_path:
            print(f"Visualisasi 3D interaktif disimpan ke: {plot_3d_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
