from .shape_utils import normalize_pair
from .convolution import Convolution


def _spatial_shape(shape_like):
    if len(shape_like) == 2:
        return int(shape_like[0]), int(shape_like[1])

    if len(shape_like) == 3:
        return int(shape_like[1]), int(shape_like[2])

    raise ValueError("image_inputsize must be (height, width) or (channels, height, width).")


def normalize_widths(widths):
    if isinstance(widths, int):
        widths = [widths]
    elif not isinstance(widths, (list, tuple)):
        raise TypeError("widths must be an int, list, or tuple")

    widths = [int(w) for w in widths]

    if len(widths) == 0:
        raise ValueError("widths must contain at least one value")

    return widths


def infer_input_depth(image_inputsize):
    if len(image_inputsize) == 2:
        return 1

    if len(image_inputsize) == 3:
        return int(image_inputsize[0])

    raise ValueError(
        "image_inputsize must be (height, width) or (channels, height, width)"
    )


def build_convolution_layers(kernel_shape, widths, input_depth):
    convolution_layers = []
    current_input_depth = int(input_depth)

    for width in widths:
        conv_layer = Convolution(
            kernel_shape=kernel_shape,
            width=int(width),
            input_depth=current_input_depth,
        )
        convolution_layers.append(conv_layer)
        current_input_depth = int(width)

    return convolution_layers


def compute_flatten_shape(
    image_inputsize,
    kernel_shape=3,
    widths=3,
    pool_size=(2, 2),
    stride=(2, 2),
    pad=1,
):
    kh, kw = normalize_pair(kernel_shape, "kernel_shape")
    ph, pw = normalize_pair(pool_size, "pool_size")
    sh, sw = normalize_pair(stride, "stride")
    image_height, image_width = _spatial_shape(image_inputsize)

    widths = normalize_widths(widths)

    current_height = image_height
    current_width = image_width

    # all convs first
    for _ in widths:
        conv_height = current_height + 2 * pad - kh + 1
        conv_width = current_width + 2 * pad - kw + 1

        if conv_height <= 0 or conv_width <= 0:
            raise ValueError(
                f"Invalid convolution output shape: ({conv_height}, {conv_width}). "
                "Check kernel_shape/pad against image_inputsize."
            )

        current_height = conv_height
        current_width = conv_width

    # one final pool
    pool_height = (current_height - ph) // sh + 1
    pool_width = (current_width - pw) // sw + 1

    if pool_height <= 0 or pool_width <= 0:
        raise ValueError(
            f"Invalid final pooling output shape: ({pool_height}, {pool_width}). "
            "Check pool_size/stride against the final convolution output."
        )

    current_height = pool_height
    current_width = pool_width

    final_width = widths[-1]
    f_height = final_width * current_height * current_width
    f_width = 1

    return (f_height, f_width)