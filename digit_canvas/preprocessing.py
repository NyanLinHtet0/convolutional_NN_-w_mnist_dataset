import numpy as np


def center_and_square(image: np.ndarray) -> np.ndarray:
    active = np.argwhere(image > 0.05)
    if active.size == 0:
        return image

    y_min, x_min = active.min(axis=0)
    y_max, x_max = active.max(axis=0)

    crop = image[y_min : y_max + 1, x_min : x_max + 1]
    height, width = crop.shape
    side = max(height, width)
    pad = max(2, side // 8)

    square = np.zeros((side + 2 * pad, side + 2 * pad), dtype=np.float32)
    y_offset = (square.shape[0] - height) // 2
    x_offset = (square.shape[1] - width) // 2
    square[y_offset : y_offset + height, x_offset : x_offset + width] = crop
    return square


def resize_average(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    in_h, in_w = image.shape
    y_edges = np.linspace(0, in_h, out_h + 1).astype(int)
    x_edges = np.linspace(0, in_w, out_w + 1).astype(int)

    resized = np.zeros((out_h, out_w), dtype=np.float32)

    for row in range(out_h):
        y0 = y_edges[row]
        y1 = max(y_edges[row + 1], y0 + 1)
        for col in range(out_w):
            x0 = x_edges[col]
            x1 = max(x_edges[col + 1], x0 + 1)
            block = image[y0:y1, x0:x1]
            resized[row, col] = float(np.mean(block)) if block.size else 0.0

    return resized


def prepare_model_input(buffer: np.ndarray, image_inputsize: tuple[int, int]) -> np.ndarray:
    if np.max(buffer) <= 0:
        return np.zeros(image_inputsize, dtype=np.float32)

    normalized = buffer.astype(np.float32)
    cropped = center_and_square(normalized)
    small = resize_average(cropped, image_inputsize[0], image_inputsize[1])

    max_val = float(np.max(small))
    if max_val > 0:
        small = small / max_val

    return small.astype(np.float32)
