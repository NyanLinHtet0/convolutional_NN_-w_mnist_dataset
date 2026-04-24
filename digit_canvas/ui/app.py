import tkinter as tk

import numpy as np

from ..feature_extractor import predict_and_collect_feature_maps
from ..preprocessing import prepare_model_input
from .view import DigitCanvasView


class DigitCanvasApp:
    def __init__(self, model, image_inputsize, app_config, feature_map_config):
        self.model = model
        self.image_inputsize = tuple(int(v) for v in image_inputsize)
        self.app_config = app_config
        self.feature_map_config = feature_map_config

        self.root = tk.Tk()
        self.view = DigitCanvasView(
            root=self.root,
            image_inputsize=self.image_inputsize,
            app_config=app_config,
            feature_map_config=feature_map_config,
        )

        self.canvas_size = int(app_config.canvas_size)
        self.brush_radius = int(app_config.brush_radius)

        self.buffer = np.zeros((self.canvas_size, self.canvas_size), dtype=np.float32)
        self.last_x = None
        self.last_y = None

        self._bind_events()

    def _feature_maps_enabled(self) -> bool:
        return bool(getattr(self.app_config, "show_feature_maps", False))

    def _bind_events(self):
        self.view.canvas.bind("<Button-1>", self._start_stroke)
        self.view.canvas.bind("<B1-Motion>", self._paint)
        self.view.canvas.bind("<ButtonRelease-1>", self._end_stroke)

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

        for step in range(1, steps + 1):
            x = int(self.last_x + dx * step / steps)
            y = int(self.last_y + dy * step / steps)
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
        radius = self.brush_radius

        self.view.canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill="white",
            outline="white",
        )

        y_min = max(0, y - radius)
        y_max = min(self.canvas_size, y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(self.canvas_size, x + radius + 1)

        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
        self.buffer[y_min:y_max, x_min:x_max][mask] = 1.0

    def _prepare_input(self):
        return prepare_model_input(self.buffer, self.image_inputsize)

    def _predict_current_digit(self):
        small = self._prepare_input()

        if self._feature_maps_enabled():
            pred, feature_maps = predict_and_collect_feature_maps(self.model, small)
            self.view.update_preview(small)
            self.view.update_feature_maps(feature_maps)
            return pred, small, feature_maps

        pred = int(self.model.predict(small))
        self.view.update_preview(small)
        return pred, small, []

    def _refresh_prediction_only(self):
        if np.max(self.buffer) <= 0:
            self.view.prediction_var.set("Current prediction: -")
            self.view.update_preview(np.zeros(self.image_inputsize, dtype=np.float32))
            self.view.clear_feature_maps()
            return

        pred, _, feature_maps = self._predict_current_digit()
        self.view.prediction_var.set(f"Current prediction: {pred}")

        if self._feature_maps_enabled() and not feature_maps:
            self.view.status_var.set(
                "Prediction updated, but feature maps were not found. "
                "If your CNN stores convolution layers differently, adjust feature_extractor.py."
            )

    def _predict_only(self, event=None):
        del event
        if np.max(self.buffer) <= 0:
            self.view.status_var.set("Canvas is empty. Draw something first.")
            return

        pred, _, feature_maps = self._predict_current_digit()
        self.view.prediction_var.set(f"Current prediction: {pred}")

        if not self._feature_maps_enabled():
            self.view.status_var.set(
                "Prediction updated. Feature maps are disabled for this run. "
                "Run draw.py maps to show all maps, or draw.py maps 50 to sample them."
            )
        elif feature_maps:
            self.view.status_var.set(
                "Prediction updated and feature maps rendered. "
                "Press Enter or Space to append it to the sequence."
            )
        else:
            self.view.status_var.set(
                "Prediction updated. Feature maps could not be collected with the current CNN structure."
            )

    def _predict_and_append(self, event=None):
        del event
        if np.max(self.buffer) <= 0:
            self.view.status_var.set("Canvas is empty. Draw a digit first.")
            return

        pred, _, _ = self._predict_current_digit()
        self.view.sequence_var.set(self.view.sequence_var.get() + str(pred))
        self.view.prediction_var.set(f"Current prediction: {pred}")
        self.view.status_var.set(f"Appended {pred}. Draw the next digit or press C to clear.")
        self._clear_buffer_only()

    def _clear_buffer_only(self):
        self.buffer.fill(0.0)
        self.view.canvas.delete("all")
        self.view.update_preview(np.zeros(self.image_inputsize, dtype=np.float32))
        self.view.clear_feature_maps()
        self.last_x = None
        self.last_y = None

    def _clear(self, event=None):
        del event
        self._clear_buffer_only()
        self.view.prediction_var.set("Current prediction: -")
        self.view.status_var.set("Canvas cleared.")

    def _remove_last_digit(self, event=None):
        del event
        current = self.view.sequence_var.get()
        self.view.sequence_var.set(current[:-1])
        self.view.status_var.set("Removed the last appended digit.")

    def run(self):
        self.root.mainloop()
