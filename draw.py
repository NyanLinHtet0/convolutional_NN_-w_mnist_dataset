from digit_canvas import (
    AppConfig,
    FeatureMapViewConfig,
    build_model,
    create_app,
    infer_model_config,
    run_digit_canvas_app,
    validate_parameter_file,
)


DEFAULT_PARAMETER_PATH = "trained_parameters_widths=[32, 64]_fc=[128, 64]_input(20, 20)_ysize=10.npz"
DEFAULT_IMAGE_INPUTSIZE = (20, 20)
DEFAULT_POOL_SIZE = (2, 2)
DEFAULT_STRIDE = (2, 2)


def main(
    parameter_path: str = DEFAULT_PARAMETER_PATH,
    image_inputsize: tuple[int, int] = DEFAULT_IMAGE_INPUTSIZE,
    pool_size: tuple[int, int] = DEFAULT_POOL_SIZE,
    stride: tuple[int, int] = DEFAULT_STRIDE,
    app_config: AppConfig | None = None,
    feature_map_config: FeatureMapViewConfig | None = None,
):
    if app_config is None:
        app_config = AppConfig()

    if feature_map_config is None:
        feature_map_config = FeatureMapViewConfig()

    validate_parameter_file(parameter_path)

    model_config = infer_model_config(
        parameter_path=parameter_path,
        image_inputsize=image_inputsize,
        pool_size=pool_size,
        stride=stride,
    )
    model = build_model(model_config)

    app = create_app(
        model=model,
        image_inputsize=model_config.image_inputsize,
        app_config=app_config,
        feature_map_config=feature_map_config,
    )
    run_digit_canvas_app(app)


if __name__ == "__main__":
    main()
