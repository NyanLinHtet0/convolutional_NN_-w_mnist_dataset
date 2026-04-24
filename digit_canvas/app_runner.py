from .config import AppConfig, FeatureMapViewConfig
from .ui import DigitCanvasApp


def create_app(
    model,
    image_inputsize,
    app_config: AppConfig,
    feature_map_config: FeatureMapViewConfig | None = None,
):
    if feature_map_config is None:
        feature_map_config = FeatureMapViewConfig()

    return DigitCanvasApp(
        model=model,
        image_inputsize=image_inputsize,
        app_config=app_config,
        feature_map_config=feature_map_config,
    )


def parse_draw_map_args(argv: list[str]):
    """
    Usage:
        python draw.py
            -> maps disabled

        python draw.py maps
            -> maps enabled, show every map

        python draw.py maps 50
            -> maps enabled, randomly show floor(50% of maps) per conv layer
    """
    args = [str(arg).strip() for arg in argv]

    if not args or args[0].lower() not in {"map", "maps", "--map", "--maps"}:
        return False, None

    if len(args) == 1:
        return True, None

    try:
        map_percent = float(args[1])
    except ValueError as exc:
        raise ValueError(
            f"Expected map percentage to be a number, got: {args[1]!r}\n"
            "Example: python draw.py maps 50"
        ) from exc

    map_percent = max(0.0, min(100.0, map_percent))
    return True, map_percent


def run_digit_canvas_app(app: DigitCanvasApp):
    app.run()
