from dataclasses import dataclass


@dataclass
class ModelConfig:
    image_inputsize: tuple[int, int]
    kernel_shape: tuple[int, int]
    widths: list[int]
    hidden_layerwidths: list[int]
    pool_size: tuple[int, int]
    stride: tuple[int, int]
    output_size: tuple[int, int]
    parameter_path: str


@dataclass
class AppConfig:
    canvas_size: int = 280
    brush_radius: int = 12
    window_size: tuple[int, int] = (1920, 1080)
    left_panel_min_width: int = 420
    right_panel_min_width: int = 1000


@dataclass
class FeatureMapViewConfig:
    panel_width: int = 1200
    panel_height: int = 950
    map_scale: int = 10
    map_padding: int = 12
    section_padding: int = 24
    max_columns: int | None = None
