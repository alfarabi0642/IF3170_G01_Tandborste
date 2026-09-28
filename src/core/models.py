from typing import Tuple, Dict, Any

# 6 orientasi standar rotasi 90 derajat terhadap sumbu x, y, dan z (sejajar sumbu kardinal)
# Format indeks dimensi asal: 0 -> w, 1 -> l, 2 -> h
ORIENTATIONS: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2),  # (w, l, h) - orientasi asli
    (0, 2, 1),  # (w, h, l)
    (1, 0, 2),  # (l, w, h)
    (1, 2, 0),  # (l, h, w)
    (2, 0, 1),  # (h, w, l)
    (2, 1, 0),  # (h, l, w)
)


class Package:
    """Representasi paket yang akan dimuat ke dalam peti kemas."""

    def __init__(
        self,
        id: str,
        w: int,
        l: int,
        h: int,
        value: float,
        weight: float,
        is_fragile: bool = False,
        eta: int = 1,
    ):
        self.id = str(id)
        self.w = int(w)
        self.l = int(l)
        self.h = int(h)
        self.value = float(value)
        self.weight = float(weight)
        self.is_fragile = bool(is_fragile)
        self.eta = int(eta)

        # Status penempatan dalam state
        self.is_loaded = False
        self.x = 0
        self.y = 0
        self.z = 0
        self.orientation_idx = 0

    def current_dims(self) -> Tuple[int, int, int]:
        """Mengembalikan dimensi efektif (w', l', h') berdasarkan orientasi rotasi saat ini."""
        dims = (self.w, self.l, self.h)
        i, j, k = ORIENTATIONS[self.orientation_idx]
        return dims[i], dims[j], dims[k]

    def bounding_box(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """Mengembalikan koordinat pojok ((x_min, y_min, z_min), (x_max, y_max, z_max))."""
        cw, cl, ch = self.current_dims()
        return (self.x, self.y, self.z), (self.x + cw, self.y + cl, self.z + ch)

    def volume(self) -> int:
        return self.w * self.l * self.h

    def set_position(self, x: int, y: int, z: int) -> None:
        self.x = int(x)
        self.y = int(y)
        self.z = int(z)

    def set_orientation(self, orientation_idx: int) -> None:
        if not (0 <= orientation_idx < len(ORIENTATIONS)):
            raise ValueError(f"Indeks orientasi harus antara 0 dan {len(ORIENTATIONS) - 1}")
        self.orientation_idx = int(orientation_idx)

    def clone(self) -> "Package":
        """Membuat salinan paket baru dengan atribut dan status yang sama."""
        copy_pkg = Package(
            id=self.id,
            w=self.w,
            l=self.l,
            h=self.h,
            value=self.value,
            weight=self.weight,
            is_fragile=self.is_fragile,
            eta=self.eta,
        )
        copy_pkg.is_loaded = self.is_loaded
        copy_pkg.x = self.x
        copy_pkg.y = self.y
        copy_pkg.z = self.z
        copy_pkg.orientation_idx = self.orientation_idx
        return copy_pkg

    def to_dict(self) -> Dict[str, Any]:
        cw, cl, ch = self.current_dims()
        return {
            "id": self.id,
            "dimensions": {"w": self.w, "l": self.l, "h": self.h},
            "current_dimensions": {"w": cw, "l": cl, "h": ch},
            "value": self.value,
            "weight": self.weight,
            "is_fragile": self.is_fragile,
            "eta": self.eta,
            "is_loaded": self.is_loaded,
            "position": {"x": self.x, "y": self.y, "z": self.z} if self.is_loaded else None,
            "orientation_idx": self.orientation_idx,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Package":
        dims = data["dimensions"]
        pkg = cls(
            id=data["id"],
            w=dims["w"],
            l=dims["l"],
            h=dims["h"],
            value=data.get("value", 0.0),
            weight=data.get("weight", 0.0),
            is_fragile=data.get("is_fragile", False),
            eta=data.get("eta", 1),
        )
        if "is_loaded" in data:
            pkg.is_loaded = bool(data["is_loaded"])
        if data.get("position"):
            pos = data["position"]
            pkg.set_position(pos["x"], pos["y"], pos["z"])
        if "orientation_idx" in data:
            pkg.set_orientation(data["orientation_idx"])
        return pkg

    def __repr__(self) -> str:
        cw, cl, ch = self.current_dims()
        status = f"pos=({self.x},{self.y},{self.z})" if self.is_loaded else "outside"
        return f"Package({self.id}, dims=({cw}x{cl}x{ch}), {status}, val={self.value})"


class Truck:
    """Representasi truk peti kemas pengangkut paket."""

    def __init__(self, id: str, w: int, l: int, h: int, max_capacity: float):
        self.id = str(id)
        self.w = int(w)
        self.l = int(l)
        self.h = int(h)
        self.max_capacity = float(max_capacity)

    def volume(self) -> int:
        return self.w * self.l * self.h

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "dimensions": {"w": self.w, "l": self.l, "h": self.h},
            "max_capacity": self.max_capacity,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Truck":
        dims = data["dimensions"]
        return cls(
            id=data["id"],
            w=dims["w"],
            l=dims["l"],
            h=dims["h"],
            max_capacity=data["max_capacity"],
        )

    def __repr__(self) -> str:
        return f"Truck({self.id}, dims=({self.w}x{self.l}x{self.h}), max_cap={self.max_capacity})"
