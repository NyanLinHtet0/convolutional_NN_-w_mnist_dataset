import numpy as np


def predict_and_collect_feature_maps(model, small_image: np.ndarray):
    """
    Returns:
        pred: int
        feature_maps: list[np.ndarray]
            Each item is expected to be shaped like (channels, height, width).

    Notes:
    - Final prediction always uses model.predict(...) so your normal inference path stays unchanged.
    - Feature maps are collected through a best-effort pass over the model's convolution layers.
    - If the model structure is different than expected, prediction still works and feature_maps falls back to [].
    """
    pred = int(model.predict(small_image))

    try:
        feature_maps = collect_feature_maps(model, small_image)
    except Exception:
        feature_maps = []

    return pred, feature_maps


def collect_feature_maps(model, small_image: np.ndarray):
    conv_layers = _get_conv_layers(model)
    if not conv_layers:
        return []

    x = np.asarray(small_image, dtype=np.float32)
    feature_maps = []

    total_layers = len(conv_layers)

    for layer_index, layer in enumerate(conv_layers):
        x = _forward_layer(layer, x)
        x = _relu(x)

        maps = _ensure_channel_first_3d(x)
        feature_maps.append(np.array(maps, copy=True))

        if _should_pool_after_layer(model, layer_index, total_layers):
            x = _maxpool_channelwise(
                x,
                pool_size=_get_pool_size(model),
                stride=_get_stride(model),
            )

    return feature_maps


def _get_conv_layers(model):
    for attr_name in ("convolution_layers", "conv_layers", "layers"):
        layers = getattr(model, attr_name, None)
        if isinstance(layers, list) and layers:
            return layers
    return []


def _forward_layer(layer, x):
    if hasattr(layer, "forward"):
        return layer.forward(x)
    raise AttributeError(f"Layer {type(layer).__name__} does not have a forward(...) method.")


def _relu(x):
    return np.maximum(np.asarray(x, dtype=np.float32), 0.0)


def _ensure_channel_first_3d(x):
    arr = np.asarray(x, dtype=np.float32)

    if arr.ndim == 2:
        return arr[np.newaxis, :, :]

    if arr.ndim == 3:
        return arr

    if arr.ndim == 4 and arr.shape[0] == 1:
        return arr[0]

    raise ValueError(f"Unsupported feature map shape: {arr.shape}")


def _get_pool_size(model):
    pool_size = getattr(model, "pool_size", (2, 2))
    return tuple(int(v) for v in pool_size)


def _get_stride(model):
    stride = getattr(model, "stride", (2, 2))
    return tuple(int(v) for v in stride)


def _should_pool_after_layer(model, layer_index: int, total_layers: int) -> bool:
    if getattr(model, "pool_every_layer", False):
        return True

    if getattr(model, "pool_only_at_end", False):
        return layer_index == total_layers - 1

    if getattr(model, "pool_between_convs", False):
        return True

    # Default matches your newer setup more closely: pool only after the last conv layer.
    return layer_index == total_layers - 1


def _maxpool_channelwise(x, pool_size=(2, 2), stride=(2, 2)):
    arr = _ensure_channel_first_3d(x)
    channels, height, width = arr.shape

    pool_h, pool_w = pool_size
    stride_h, stride_w = stride

    if height < pool_h or width < pool_w:
        return arr

    out_h = 1 + (height - pool_h) // stride_h
    out_w = 1 + (width - pool_w) // stride_w

    pooled = np.zeros((channels, out_h, out_w), dtype=np.float32)

    for channel in range(channels):
        for row in range(out_h):
            y0 = row * stride_h
            y1 = y0 + pool_h

            for col in range(out_w):
                x0 = col * stride_w
                x1 = x0 + pool_w

                pooled[channel, row, col] = float(np.max(arr[channel, y0:y1, x0:x1]))

    return pooled
