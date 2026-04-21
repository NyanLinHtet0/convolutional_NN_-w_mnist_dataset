import os
import numpy as np

try:
    from mycnn.core.cnn import CNN
except ImportError:
    from mycnn import CNN

from .config import ModelConfig


def to_int_list(value):
    if value is None:
        return []

    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return [int(x) for x in value.tolist()]
        return [int(x) for x in value.flatten().tolist()]

    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]

    return [int(value)]


def load_parameter_data(parameter_path: str):
    return np.load(parameter_path, allow_pickle=True)


def unwrap_saved_layers(raw_value):
    if getattr(raw_value, "dtype", None) == object:
        return list(raw_value)
    return [raw_value]


def infer_model_config(
    parameter_path: str,
    image_inputsize: tuple[int, int],
    pool_size: tuple[int, int],
    stride: tuple[int, int],
) -> ModelConfig:
    data = load_parameter_data(parameter_path)

    conv_kernels = unwrap_saved_layers(data["conv_kernels"])
    dense_biases = unwrap_saved_layers(data["dense_biases"])

    if not conv_kernels:
        raise ValueError("No convolution kernels found in the parameter file.")

    if not dense_biases:
        raise ValueError("No dense biases found in the parameter file.")

    if "widths" in data:
        widths = to_int_list(data["widths"])
    else:
        widths = [int(kernel.shape[0]) for kernel in conv_kernels]

    if "hidden_layerwidths" in data:
        hidden_layerwidths = to_int_list(data["hidden_layerwidths"])
    else:
        hidden_layerwidths = [int(bias.shape[0]) for bias in dense_biases[:-1]]

    first_kernel = conv_kernels[0]
    kernel_shape = (int(first_kernel.shape[2]), int(first_kernel.shape[3]))

    last_bias = dense_biases[-1]
    output_size = (int(last_bias.shape[0]), int(last_bias.shape[1]))

    return ModelConfig(
        image_inputsize=tuple(int(x) for x in image_inputsize),
        kernel_shape=kernel_shape,
        widths=widths,
        hidden_layerwidths=hidden_layerwidths,
        pool_size=tuple(int(x) for x in pool_size),
        stride=tuple(int(x) for x in stride),
        output_size=output_size,
        parameter_path=parameter_path,
    )


def validate_parameter_file(parameter_path: str):
    if not os.path.exists(parameter_path):
        raise FileNotFoundError(
            f"Could not find parameter file: {parameter_path}\n"
            "Pass the correct path into main() or change the default in this script."
        )


def build_model(config: ModelConfig):
    model = CNN(
        image_inputsize=config.image_inputsize,
        kernel_shape=config.kernel_shape,
        widths=config.widths,
        hidden_layerwidths=config.hidden_layerwidths,
        pool_size=config.pool_size,
        stride=config.stride,
        output_size=config.output_size,
    )
    model.load_parameters(config.parameter_path)
    return model
