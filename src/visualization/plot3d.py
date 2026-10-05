from pathlib import Path
from typing import Optional, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.core.state import State
from src.core.models import Truck, Package


# color palette
COLOR_PALETTE = [
    "royalblue",
    "forestgreen",
    "darkorange",
    "mediumpurple",
    "teal",
    "goldenrod",
    "deepskyblue",
    "orchid",
]

# untuk is_fragile = True, pakei merah
FRAGILE_COLOR = "crimson"
# untuk paket yang belum dimuat (unloaded), pakei abu-abu
UNLOADED_COLOR = "slategray"

# fungsi untuk menghitung data mesh dan wireframe balok 3D (paket atau kontainer truk)
def _box_mesh_data(x: float, y: float, z: float, w: float, l: float, h: float):
    
    # koordinat X, Y, Z untuk kedelapan titik sudut balok
    vx = [x, x + w, x + w, x, x, x + w, x + w, x]
    vy = [y, y, y + l, y + l, y, y, y + l, y + l]
    vz = [z, z, z, z, z + h, z + h, z + h, z + h]

    # index segitiga pembentuk 6 bidang sisi balok (i, j, k)
    i = [0, 0, 4, 4, 0, 0, 2, 2, 0, 0, 1, 1]
    j = [1, 2, 5, 6, 1, 5, 3, 7, 3, 7, 2, 6]
    k = [2, 3, 6, 7, 5, 4, 7, 6, 7, 4, 6, 5]

    return vx, vy, vz, i, j, k

# fungsi untuk menghasilkan data garis kerangka (wireframe) balok 3D
# pemisah segmennya pakeki None
def _box_wireframe_data(x: float, y: float, z: float, w: float, l: float, h: float):

    # Garis sisi alas (4 rusuk bawah) 
    # Garis sisi atap (4 rusuk atas)
    # 4 Rusuk tegak vertikal
    wx = [
        x, x + w, x + w, x, x, None,
        x, x + w, x + w, x, x, None,
        x, x, None,
        x + w, x + w, None,
        x + w, x + w, None,
        x, x,
    ]
    wy = [
        y, y, y + l, y + l, y, None,
        y, y, y + l, y + l, y, None,
        y, y, None,
        y, y, None,
        y + l, y + l, None,
        y + l, y + l,
    ]
    wz = [
        z, z, z, z, z, None,
        z + h, z + h, z + h, z + h, z + h, None,
        z, z + h, None,
        z, z + h, None,
        z, z + h, None,
        z, z + h,
    ]
    return wx, wy, wz

# helper untuk membuat trace garis tepi truk (wireframe) dan paket (mesh + outline + label) 
def _create_truck_traces(truck: Truck) -> List[go.Scatter3d]:

    wx, wy, wz = _box_wireframe_data(0, 0, 0, truck.w, truck.l, truck.h)
    trace = go.Scatter3d(
        x=wx,
        y=wy,
        z=wz,
        mode="lines",
        line=dict(color="black", width=4),
        name=f"Truk ({truck.id})",
        hoverinfo="name",
    )
    return [trace]

# helper untuk membangun trace visualisasi paket (Mesh3d + Outline + Label) di dalam truk atau di luar truk
def _build_package_traces(
    pkg: Package,
    disp_x: float,
    disp_y: float,
    disp_z: float,
    color: str,
    opacity: float = 0.8,
    is_outside: bool = False,
) -> List[go.Trace]:
    
    w, l, h = pkg.current_dims()
    tag = "[LUAR] " if is_outside else ""
    hover = (
        f"<b>{tag}Paket {pkg.id}</b><br>"
        f"Dimensi: {w} x {l} x {h}<br>"
        f"Posisi: ({disp_x:.1f}, {disp_y:.1f}, {disp_z:.1f})<br>"
        f"Nilai: {pkg.value}<br>"
        f"Berat: {pkg.weight}<br>"
        f"Fragile: {'Ya' if pkg.is_fragile else 'Tidak'}<br>"
        f"ETA: {pkg.eta}"
    )

    # mesh permukaan balok 3D
    vx, vy, vz, i, j, k = _box_mesh_data(disp_x, disp_y, disp_z, w, l, h)
    mesh = go.Mesh3d(
        x=vx,
        y=vy,
        z=vz,
        i=i,
        j=j,
        k=k,
        color=color,
        opacity=opacity,
        name=f"{tag}{pkg.id}",
        hoverinfo="text",
        text=hover,
        showlegend=True,
    )

    # outline garis tepi (kerangka rusuk balok)
    wx, wy, wz = _box_wireframe_data(disp_x, disp_y, disp_z, w, l, h)
    outline = go.Scatter3d(
        x=wx,
        y=wy,
        z=wz,
        mode="lines",
        line=dict(color="darkslategray" if is_outside else "black", width=2),
        hoverinfo="skip",
        showlegend=False,
    )

    # label teks ID tepat di titik tengah balok
    center_text = go.Scatter3d(
        x=[disp_x + w / 2],
        y=[disp_y + l / 2],
        z=[disp_z + h / 2],
        mode="text",
        text=[pkg.id],
        textfont=dict(color="white" if not is_outside else "gray", size=10),
        hoverinfo="skip",
        showlegend=False,
    )

    return [mesh, outline, center_text]

# fungsi untuk membangun semua trace visualisasi 3D dari suatu konfigurasi State
def build_state_traces(state: State) -> List[go.Trace]:
    
    traces: List[go.Trace] = []
    truck = state.truck

    # Wireframe pembatas kotak truk
    traces.extend(_create_truck_traces(truck))

    # paket di dalam kontainer truk
    for idx, pkg in enumerate(state.loaded_packages):
        # Paket pecah belah diberi warna merah khusus, paket biasa diberi warna dari palet
        color = FRAGILE_COLOR if pkg.is_fragile else COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        pkg_traces = _build_package_traces(
            pkg=pkg,
            disp_x=pkg.x,
            disp_y=pkg.y,
            disp_z=pkg.z,
            color=color,
            opacity=0.85,
            is_outside=False,
        )
        traces.extend(pkg_traces)

    # Paket di luar truk, ditata disamping kiri truk (ditata di area staging ilustratif samping kiri truk)
    staging_y = 0.0
    for pkg in state.unloaded_packages:
        w, l, h = pkg.current_dims()
        
        # Geser posisi tampilan ke area luar kiri kontainer
        disp_x = -(w + 2.0)
        disp_y = staging_y
        disp_z = 0.0

        pkg_traces = _build_package_traces(
            pkg=pkg,
            disp_x=disp_x,
            disp_y=disp_y,
            disp_z=disp_z,
            color=UNLOADED_COLOR,
            opacity=0.45,
            is_outside=True,
        )
        traces.extend(pkg_traces)
        
        # Geser offset y untuk paket luar berikutnya agar tidak saling tumpuk
        staging_y += l + 2.0

    return traces

# fungsi utama untuk memvisualisasikan konfigurasi muatan truk dalam 3D interaktif
def visualize_state_3d(
    state: State,
    title: str = "3D Container Loading",
    show: bool = False,
    save_path: Optional[str | Path] = None,
) -> go.Figure:
  
    fig = go.Figure()

    # add semua trace ke plotly
    for trace in build_state_traces(state):
        fig.add_trace(trace)

    # teks ringkasan untuk subtitle 
    truck = state.truck
    loaded_wt = state.total_loaded_weight()
    score = state.get_score()
    valid_str = "Valid" if state.is_valid() else "Tidak Valid"

    subtitle = (
        f"Status: {valid_str} | Nilai Objektif: {score:.2f} | "
        f"Muatan: {loaded_wt:.1f}/{truck.max_capacity:.1f} kg | "
        f"Dimuat: {len(state.loaded_packages)}/{len(state.packages)} paket"
    )

    # konfig tata letak scene 3D Plotly 
    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br><sup>{subtitle}</sup>", x=0.5),
        scene=dict(
            xaxis_title="Lebar (W - x)",
            yaxis_title="Panjang (L - y)",
            zaxis_title="Tinggi (H - z)",
            aspectmode="data",
        ),
        template="plotly_white",
        margin=dict(l=20, r=20, b=20, t=60),
    )

    # Simpan file HTML interaktif jika path ditentukan
    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out))

    # show figure di browser
    if show:
        fig.show()

    return fig

# fungsi untuk memvisualisasikan perbandingan 3D side-by-side antara State Awal dan State Akhir
def visualize_comparison_3d(
    initial_state: State,
    final_state: State,
    title: str = "Perbandingan Solusi Awal vs Akhir",
    show: bool = False,
    save_path: Optional[str | Path] = None,
) -> go.Figure:
    
    init_score = initial_state.get_score()
    final_score = final_state.get_score()

    # Buat subplot 1 baris 2 kolom dengan scene 3D pada masing-masing kolom
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[
            f"State Awal (Nilai: {init_score:.1f})",
            f"State Akhir (Nilai: {final_score:.1f})",
        ],
    )

    # render State Awal pada kolom kiri
    for trace in build_state_traces(initial_state):
        fig.add_trace(trace, row=1, col=1)

    # render State Akhir pada kolom kanan
    for trace in build_state_traces(final_state):
        fig.add_trace(trace, row=1, col=2)

    # Konfigurasi tata letak gabungan
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5),
        template="plotly_white",
        margin=dict(l=20, r=20, b=20, t=60),
    )
    fig.update_scenes(aspectmode="data")

    # Simpan file HTML 
    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out))

    # show figure di browser
    if show:
        fig.show()

    return fig
