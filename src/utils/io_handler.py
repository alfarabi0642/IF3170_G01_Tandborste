import json
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

from src.core.models import Package, Truck
from src.algorithms.base import SearchResult

# helper function untuk validasi dimensi dictionary (w, l, h)
def _validate_dims(dims: Any, context: str) -> Tuple[int, int, int]:
   
    # cek data type, harus berupa dictionary
    if not isinstance(dims, dict):
        raise ValueError(f"{context}: 'dimensions' harus berupa dictionary dengan w, l, h.")
    
    # cek nilai untuk masing-masing sumbu lebar (w), panjang (l), dan tinggi (h)
    for key in ("w", "l", "h"):
        if key not in dims:
            raise ValueError(f"{context}: dimensi kurang '{key}'.")
        val = dims[key]
        
        # cek nilai adalah integer sejati dan lebih besar dari nol
        if type(val) is not int or val <= 0:
            raise ValueError(f"{context}: dimensi '{key}' harus integer positif, dapat: {val!r}")
            
    return dims["w"], dims["l"], dims["h"]

# fungsi untuk membaca file JSON input dan memvalidasi isinya
def load_input(file_path: str | Path) -> Tuple[Truck, List[Package]]:
  
    path = Path(file_path)

    # sanity cek
    if not path.is_file():
        raise FileNotFoundError(f"File masukan tidak ditemukan: {path}")

    # baca file
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # validasi struktur root data JSON
    if not isinstance(data, dict):
        raise ValueError("Format file JSON harus berupa dictionary.")

    if "truck" not in data or "packages" not in data:
        raise ValueError("JSON wajib memuat 'truck' dan 'packages'.")

    # validasi dan instansiasi objek Truck
    truck_data = data["truck"]
    if not isinstance(truck_data, dict):
        raise ValueError("'truck' harus berupa dictionary.")

    # ambil dan periksa ID truk agar tidak kosong
    truck_id = str(truck_data.get("id", "")).strip()
    if not truck_id:
        raise ValueError("Truk harus memiliki 'id' berupa teks.")

    # Validasi ukuran kontainer dan kapasitas angkut maksimum
    tw, tl, th = _validate_dims(truck_data.get("dimensions"), f"Truk '{truck_id}'")
    cap = truck_data.get("max_capacity")
    if (type(cap) not in (int, float)) or cap <= 0:
        raise ValueError(f"Truk '{truck_id}': 'max_capacity' harus angka positif > 0, dapat: {cap!r}")

    # Bentuk objek Truck model inti
    truck = Truck(id=truck_id, w=tw, l=tl, h=th, max_capacity=float(cap))

    # validasi dan instansiasi daftar objek Package
    pkg_list = data["packages"]
    if not isinstance(pkg_list, list) or len(pkg_list) == 0:
        raise ValueError("Daftar 'packages' tidak boleh kosong.")

    seen_ids = set()
    packages: List[Package] = []

    # Iterasi setiap paket dan periksa integritas datanya
    for idx, item in enumerate(pkg_list):
        if not isinstance(item, dict):
            raise ValueError(f"Paket indeks {idx} bukan dictionary.")

        # Cek keunikan ID paket (harus unik)
        pkg_id = str(item.get("id", "")).strip()
        if not pkg_id:
            raise ValueError(f"Paket indeks {idx} tidak memiliki 'id'.")
        if pkg_id in seen_ids:
            raise ValueError(f"ID paket duplikat: '{pkg_id}'.")
        seen_ids.add(pkg_id)

        # Cek dimensi paket
        pw, pl, ph = _validate_dims(item.get("dimensions"), f"Paket '{pkg_id}'")

        # Cek nilai ekonomis barang (harus > 0)
        val = item.get("value", 0)
        if (type(val) not in (int, float)) or val < 0:
            raise ValueError(f"Paket '{pkg_id}': 'value' harus non-negatif, dapat: {val!r}")

        # cek bobot berat barang (harus strictly positif)
        wt = item.get("weight", 0)
        if (type(wt) not in (int, float)) or wt <= 0:
            raise ValueError(f"Paket '{pkg_id}': 'weight' harus bernilai positif > 0, dapat: {wt!r}")

        # cek atribut barang pecah belah (is_fragile harus tipe boolean sejati)
        fragile = item.get("is_fragile", False)
        if type(fragile) is not bool:
            raise ValueError(f"Paket '{pkg_id}': 'is_fragile' harus boolean (true/false).")

        # cek sisa hari pengiriman (ETA harus bilangan bulat >= 1)
        eta = item.get("eta", 1)
        if type(eta) is not int or eta < 1:
            raise ValueError(f"Paket '{pkg_id}': 'eta' harus integer >= 1, dapat: {eta!r}")

        # buat objek Package dan masukkan ke dalam daftar
        pkg = Package(
            id=pkg_id,
            w=pw,
            l=pl,
            h=ph,
            value=float(val),
            weight=float(wt),
            is_fragile=fragile,
            eta=eta,
        )
        packages.append(pkg)

    return truck, packages

# funsi untuk menyimpan hasil pencarian algoritma ke file JSON
def save_run_result(
    result: SearchResult,
    output_dir: str | Path = "output",
    filename: Optional[str] = None,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Path:
   
    out_dir = Path(output_dir)

    # Buat direktori output secara rekursif jika belum ada
    out_dir.mkdir(parents=True, exist_ok=True)

    # penamaan file otomatis jika tidak ditentukan: <nama_algo>_<timestamp>.json
    if not filename:
        slug = result.algorithm_name.lower().replace(" ", "_").replace("-", "_")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{slug}_{timestamp}.json"

    out_path = out_dir / filename

    # Serialisasi data SearchResult ke format dictionary
    data = result.to_dict()
    # add informasi tambahan (misal path file input, jenis objective function, seed)
    if extra_info:
        data.update(extra_info)

    # Tulis file JSON ke disk
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return out_path
