import math
import tkinter as tk

import numpy as np


class FeatureMapsPanel(tk.Frame):
    def __init__(
        self,
        master,
        panel_width: int = 1200,
        panel_height: int = 950,
        map_scale: int = 10,
        map_padding: int = 12,
        section_padding: int = 24,
        max_columns: int | None = None,
    ):
        super().__init__(master)

        self.panel_width = int(panel_width)
        self.panel_height = int(panel_height)
        self.map_scale = int(map_scale)
        self.map_padding = int(map_padding)
        self.section_padding = int(section_padding)
        self.max_columns = max_columns

        title = tk.Label(self, text="Convolution Feature Maps", font=("Arial", 11, "bold"))
        title.pack(anchor="w", pady=(0, 6))

        container = tk.Frame(self)
        container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            container,
            width=self.panel_width,
            height=self.panel_height,
            bg="white",
            highlightthickness=1,
            highlightbackground="#555555",
        )
        self.v_scroll = tk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.h_scroll = tk.Scrollbar(container, orient="horizontal", command=self.canvas.xview)

        self.canvas.configure(
            yscrollcommand=self.v_scroll.set,
            xscrollcommand=self.h_scroll.set,
        )

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

    def clear(self):
        self.canvas.delete("all")
        self.canvas.configure(scrollregion=(0, 0, self.panel_width, self.panel_height))

    def render(self, feature_maps: list[np.ndarray]):
        self.clear()

        if not feature_maps:
            self.canvas.create_text(
                12,
                12,
                anchor="nw",
                text="No feature maps to display.",
                font=("Arial", 10),
                fill="#333333",
            )
            return

        x_margin = 12
        y_cursor = 12
        usable_width = max(300, self.panel_width - 2 * x_margin)

        for layer_idx, layer_maps in enumerate(feature_maps):
            maps = np.asarray(layer_maps, dtype=np.float32)

            if maps.ndim == 4 and maps.shape[0] == 1:
                maps = maps[0]

            if maps.ndim != 3:
                continue

            channels, height, width = maps.shape

            header = f"Conv Layer {layer_idx + 1} — {channels} map(s), each {height}x{width}"
            self.canvas.create_text(
                x_margin,
                y_cursor,
                anchor="nw",
                text=header,
                font=("Arial", 10, "bold"),
                fill="#222222",
            )
            y_cursor += 24

            tile_w = width * self.map_scale
            tile_h = height * self.map_scale
            cell_w = tile_w + self.map_padding
            cell_h = tile_h + 20 + self.map_padding

            cols = max(1, usable_width // max(cell_w, 1))
            cols = min(cols, channels)

            if self.max_columns is not None:
                cols = min(cols, int(self.max_columns))

            for channel_idx in range(channels):
                row = channel_idx // cols
                col = channel_idx % cols

                x0 = x_margin + col * cell_w
                y0 = y_cursor + row * cell_h

                self.canvas.create_text(
                    x0,
                    y0,
                    anchor="nw",
                    text=f"Map {channel_idx}",
                    font=("Arial", 8),
                    fill="#444444",
                )

                self._draw_single_map(
                    feature_map=maps[channel_idx],
                    x=x0,
                    y=y0 + 12,
                )

            rows = math.ceil(channels / cols)
            y_cursor += rows * cell_h + self.section_padding

        bbox = self.canvas.bbox("all")
        if bbox is not None:
            self.canvas.configure(scrollregion=bbox)

    def _draw_single_map(self, feature_map: np.ndarray, x: int, y: int):
        norm = self._normalize(feature_map)
        h, w = norm.shape

        self.canvas.create_rectangle(
            x,
            y,
            x + w * self.map_scale,
            y + h * self.map_scale,
            outline="#999999",
            width=1,
        )

        for row in range(h):
            for col in range(w):
                value = float(norm[row, col])
                gray = int(round(255 * (1.0 - value)))
                color = f"#{gray:02x}{gray:02x}{gray:02x}"

                x1 = x + col * self.map_scale
                y1 = y + row * self.map_scale
                x2 = x1 + self.map_scale
                y2 = y1 + self.map_scale

                self.canvas.create_rectangle(
                    x1,
                    y1,
                    x2,
                    y2,
                    outline="",
                    fill=color,
                )

    @staticmethod
    def _normalize(feature_map: np.ndarray) -> np.ndarray:
        arr = np.asarray(feature_map, dtype=np.float32)

        min_val = float(np.min(arr))
        max_val = float(np.max(arr))

        if max_val <= min_val:
            return np.zeros_like(arr, dtype=np.float32)

        return (arr - min_val) / (max_val - min_val)
