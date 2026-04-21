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


def run_digit_canvas_app(app: DigitCanvasApp):
    app.run()
