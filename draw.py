import argparse
import os
import tkinter as tk
from dataclasses import dataclass

import numpy as np

try:
    from mycnn.core.cnn import CNN
except ImportError:
    # Fallback in case the user exposes CNN at package level differently.
    from mycnn import CNN


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


def _to_int_list(value):
    if value is None:
        return []

    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return [int(x) for x in value.tolist()]
        return [int(x) for x in value.flatten().tolist()]

    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]

    return [int(value)]


def infer_config_from_parameters(parameter_path, image_inputsize=(20, 20), pool_size=(2, 2), stride=(2, 2)):
    data = np.load(parameter_path, allow_pickle=True)

    conv_kernels_raw = data["conv_kernels"]
    dense_biases_raw = data["dense_biases"]

    conv_kernels = list(conv_kernels_raw) if getattr(conv_kernels_raw, "dtype", None) == object else [conv_kernels_raw]
    dense_biases = list(dense_biases_raw) if getattr(dense_biases_raw, "dtype", None) == object else [dense_biases_raw]

    if len(conv_kernels) == 0:
        raise ValueError("No convolution kernels found in the parameter file.")

    if len(dense_biases) == 0:
        raise ValueError("No dense biases found in the parameter file.")

    widths = _to_int_list(data["widths"]) if "widths" in data else [int(k.shape[0]) for k in conv_kernels]

    if "hidden_layerwidths" in data:
        hidden_layerwidths = _to_int_list(data["hidden_layerwidths"])
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


class DigitCanvasApp:
    def __init__(self, model, image_inputsize=(20, 20), canvas_size=280, brush_radius=15):
        self.model = model
        self.image_inputsize = tuple(image_inputsize)
        self.canvas_size = int(canvas_size)
        self.brush_radius = int(brush_radius)

        self.root = tk.Tk()
        self.root.title("CNN Digit Canvas Demo")
        self.root.resizable(False, False)

        self.sequence_var = tk.StringVar(value="")
        self.prediction_var = tk.StringVar(value="Current prediction: -")
        self.status_var = tk.StringVar(
            value="Draw with the mouse. Enter/Space = append prediction | C = clear | Backspace = remove last digit | Esc = quit"
        )

        self.canvas = tk.Canvas(
            self.root,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="black",
            highlightthickness=1,
            highlightbackground="#555555",
        )
        self.canvas.pack(padx=12, pady=(12, 8))

        output_frame = tk.Frame(self.root)
        output_frame.pack(fill="x", padx=12, pady=(0, 8))

        tk.Label(output_frame, text="Predicted sequence:", anchor="w", font=("Arial", 11, "bold")).pack(fill="x")
        tk.Label(
            output_frame,
            textvariable=self.sequence_var,
            anchor="w",
            justify="left",
            bg="#f1f1f1",
            relief="sunken",
            padx=8,
            pady=8,
            font=("Consolas", 16),
        ).pack(fill="x", pady=(4, 8))

        tk.Label(output_frame, textvariable=self.prediction_var, anchor="w", font=("Arial", 11)).pack(fill="x")
        tk.Label(
            output_frame,
            textvariable=self.status_var,
            anchor="w",
            justify="left",
            wraplength=self.canvas_size,
            fg="#333333",
            font=("Arial", 10),
        ).pack(fill="x", pady=(4, 4))

        self.preview_label = tk.Label(self.root, text="10x10 input preview", font=("Arial", 10, "bold"))
        self.preview_label.pack(anchor="w", padx=12)

        self.preview = tk.Canvas(
            self.root,
            width=self.image_inputsize[1] * 16,
            height=self.image_inputsize[0] * 16,
            bg="white",
            highlightthickness=1,
            highlightbackground="#555555",
        )
        self.preview.pack(padx=12, pady=(4, 12), anchor="w")

        self.preview_rects = []
        for r in range(self.image_inputsize[0]):
            row = []
            for c in range(self.image_inputsize[1]):
                rect = self.preview.create_rectangle(
                    c * 16,
                    r * 16,
                    (c + 1) * 16,
                    (r + 1) * 16,
                    outline="#d0d0d0",
                    fill="#ffffff",
                )
                row.append(rect)
            self.preview_rects.append(row)

        self.buffer = np.zeros((self.canvas_size, self.canvas_size), dtype=np.float32)
        self.last_x = None
        self.last_y = None

        self.canvas.bind("<Button-1>", self._start_stroke)
        self.canvas.bind("<B1-Motion>", self._paint)
        self.canvas.bind("<ButtonRelease-1>", self._end_stroke)

        self.root.bind("<Return>", self._predict_and_append)
        self.root.bind("<space>", self._predict_and_append)
        self.root.bind("<Key-p>", self._predict_only)
        self.root.bind("<Key-c>", self._clear)
        self.root.bind("<BackSpace>", self._remove_last_digit)
        self.root.bind("<Escape>", lambda event: self.root.destroy())

    def _start_stroke(self, event):
        self.last_x = event.x
        self.last_y = event.y
        self._draw_brush(event.x, event.y)
        self._refresh_prediction_only()

    def _paint(self, event):
        if self.last_x is None or self.last_y is None:
            self._start_stroke(event)
            return

        dx = event.x - self.last_x
        dy = event.y - self.last_y
        steps = max(abs(dx), abs(dy), 1)

        for i in range(1, steps + 1):
            x = int(self.last_x + dx * i / steps)
            y = int(self.last_y + dy * i / steps)
            self._draw_brush(x, y)

        self.last_x = event.x
        self.last_y = event.y
        self._refresh_prediction_only()

    def _end_stroke(self, event):
        del event
        self.last_x = None
        self.last_y = None
        self._refresh_prediction_only()

    def _draw_brush(self, x, y):
        r = self.brush_radius
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")

        y_min = max(0, y - r)
        y_max = min(self.canvas_size, y + r + 1)
        x_min = max(0, x - r)
        x_max = min(self.canvas_size, x + r + 1)

        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2
        self.buffer[y_min:y_max, x_min:x_max][mask] = 1.0

    def _center_and_square(self, image):
        active = np.argwhere(image > 0.05)
        if active.size == 0:
            return image

        y_min, x_min = active.min(axis=0)
        y_max, x_max = active.max(axis=0)

        crop = image[y_min:y_max + 1, x_min:x_max + 1]
        h, w = crop.shape
        side = max(h, w)
        pad = max(2, side // 8)

        square = np.zeros((side + 2 * pad, side + 2 * pad), dtype=np.float32)
        y_offset = (square.shape[0] - h) // 2
        x_offset = (square.shape[1] - w) // 2
        square[y_offset:y_offset + h, x_offset:x_offset + w] = crop
        return square

    def _resize_average(self, image, out_h, out_w):
        in_h, in_w = image.shape
        y_edges = np.linspace(0, in_h, out_h + 1).astype(int)
        x_edges = np.linspace(0, in_w, out_w + 1).astype(int)

        resized = np.zeros((out_h, out_w), dtype=np.float32)

        for r in range(out_h):
            y0 = y_edges[r]
            y1 = max(y_edges[r + 1], y0 + 1)
            for c in range(out_w):
                x0 = x_edges[c]
                x1 = max(x_edges[c + 1], x0 + 1)
                block = image[y0:y1, x0:x1]
                resized[r, c] = float(np.mean(block)) if block.size else 0.0

        return resized

    def _prepare_input(self):
        if np.max(self.buffer) <= 0:
            return np.zeros(self.image_inputsize, dtype=np.float32)

        normalized = self.buffer.astype(np.float32)
        cropped = self._center_and_square(normalized)
        small = self._resize_average(cropped, self.image_inputsize[0], self.image_inputsize[1])

        max_val = float(np.max(small))
        if max_val > 0:
            small = small / max_val

        return small.astype(np.float32)

    def _update_preview(self, small_image):
        for r in range(self.image_inputsize[0]):
            for c in range(self.image_inputsize[1]):
                value = float(np.clip(small_image[r, c], 0.0, 1.0))
                gray = int(round(255 * (1.0 - value)))
                color = f"#{gray:02x}{gray:02x}{gray:02x}"
                self.preview.itemconfig(self.preview_rects[r][c], fill=color)

    def _predict_current_digit(self):
        small = self._prepare_input()
        self._update_preview(small)
        pred = int(self.model.predict(small))
        return pred, small

    def _refresh_prediction_only(self):
        if np.max(self.buffer) <= 0:
            self.prediction_var.set("Current prediction: -")
            self._update_preview(np.zeros(self.image_inputsize, dtype=np.float32))
            return

        pred, _ = self._predict_current_digit()
        self.prediction_var.set(f"Current prediction: {pred}")

    def _predict_only(self, event=None):
        del event
        if np.max(self.buffer) <= 0:
            self.status_var.set("Canvas is empty. Draw something first.")
            return

        pred, _ = self._predict_current_digit()
        self.prediction_var.set(f"Current prediction: {pred}")
        self.status_var.set("Prediction updated. Press Enter or Space to append it to the sequence.")

    def _predict_and_append(self, event=None):
        del event
        if np.max(self.buffer) <= 0:
            self.status_var.set("Canvas is empty. Draw a digit first.")
            return

        pred, _ = self._predict_current_digit()
        self.sequence_var.set(self.sequence_var.get() + str(pred))
        self.prediction_var.set(f"Current prediction: {pred}")
        self.status_var.set(f"Appended {pred}. Draw the next digit or press C to clear.")
        self._clear_buffer_only()

    def _clear_buffer_only(self):
        self.buffer.fill(0.0)
        self.canvas.delete("all")
        self._update_preview(np.zeros(self.image_inputsize, dtype=np.float32))
        self.last_x = None
        self.last_y = None

    def _clear(self, event=None):
        del event
        self._clear_buffer_only()
        self.prediction_var.set("Current prediction: -")
        self.status_var.set("Canvas cleared.")

    def _remove_last_digit(self, event=None):
        del event
        current = self.sequence_var.get()
        self.sequence_var.set(current[:-1])
        self.status_var.set("Removed the last appended digit.")

    def run(self):
        self.root.mainloop()


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


def parse_args():
    parser = argparse.ArgumentParser(description="Draw a digit and let your saved CNN predict it.")
    parser.add_argument(
        "--params",
        default="trained_parameters_widths=[32, 64]_fc=[128, 64]_input(20, 20)_ysize=10.npz",
        help="Path to the saved .npz parameter file.",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=10,
        help="Input image height used during training.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=10,
        help="Input image width used during training.",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        nargs=2,
        default=(2, 2),
        help="Final max-pool size used by the model.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        nargs=2,
        default=(2, 2),
        help="Final max-pool stride used by the model.",
    )
    parser.add_argument(
        "--canvas-size",
        type=int,
        default=280,
        help="Drawing canvas size in pixels.",
    )
    parser.add_argument(
        "--brush-radius",
        type=int,
        default=12,
        help="Brush radius in pixels.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.params):
        raise FileNotFoundError(
            f"Could not find parameter file: {args.params}\n"
            "Pass the correct path with --params or change the default in this script."
        )

    config = infer_config_from_parameters(
        parameter_path=args.params,
        image_inputsize=(args.image_height, args.image_width),
        pool_size=tuple(args.pool_size),
        stride=tuple(args.stride),
    )

    model = build_model(config)
    app = DigitCanvasApp(
        model=model,
        image_inputsize=config.image_inputsize,
        canvas_size=args.canvas_size,
        brush_radius=args.brush_radius,
    )
    app.run()


if __name__ == "__main__":
    main()

