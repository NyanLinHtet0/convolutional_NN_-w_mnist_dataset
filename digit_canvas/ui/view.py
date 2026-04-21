import tkinter as tk

import numpy as np

from .feature_maps import FeatureMapsPanel


class DigitCanvasView:
    def __init__(self, root, image_inputsize, app_config, feature_map_config):
        self.root = root
        self.image_inputsize = tuple(int(v) for v in image_inputsize)
        self.app_config = app_config
        self.feature_map_config = feature_map_config

        self.root.title("CNN Digit Canvas Demo")
        self.root.geometry(f"{app_config.window_size[0]}x{app_config.window_size[1]}")
        self.root.minsize(1200, 700)
        self.root.resizable(True, True)

        self.sequence_var = tk.StringVar(master=self.root, value="")
        self.prediction_var = tk.StringVar(master=self.root, value="Current prediction: -")
        self.status_var = tk.StringVar(
            master=self.root,
            value=(
                "Draw with the mouse. Enter/Space = append prediction | "
                "P = predict only | C = clear | Backspace = remove last digit | Esc = quit"
            ),
        )

        self.canvas = None
        self.preview = None
        self.preview_rects = []
        self.feature_maps_panel = None

        self._build_ui()

    def _build_ui(self):
        main = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        main.pack(fill="both", expand=True)

        left = tk.Frame(main)
        right = tk.Frame(main)

        main.add(left, minsize=self.app_config.left_panel_min_width)
        main.add(right, minsize=self.app_config.right_panel_min_width)

        self._build_left_panel(left)
        self._build_right_panel(right)

    def _build_left_panel(self, parent):
        canvas_frame = tk.Frame(parent)
        canvas_frame.pack(fill="x", padx=12, pady=(12, 8))

        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.app_config.canvas_size,
            height=self.app_config.canvas_size,
            bg="black",
            highlightthickness=1,
            highlightbackground="#555555",
        )
        self.canvas.pack(anchor="w")

        output_frame = tk.Frame(parent)
        output_frame.pack(fill="x", padx=12, pady=(0, 8))

        tk.Label(
            output_frame,
            text="Predicted sequence:",
            anchor="w",
            font=("Arial", 11, "bold"),
        ).pack(fill="x")

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

        tk.Label(
            output_frame,
            textvariable=self.prediction_var,
            anchor="w",
            font=("Arial", 11),
        ).pack(fill="x")

        tk.Label(
            output_frame,
            textvariable=self.status_var,
            anchor="w",
            justify="left",
            wraplength=max(350, self.app_config.left_panel_min_width - 40),
            fg="#333333",
            font=("Arial", 10),
        ).pack(fill="x", pady=(4, 4))

        preview_label = tk.Label(parent, text="Model input preview", font=("Arial", 10, "bold"))
        preview_label.pack(anchor="w", padx=12)

        self.preview = tk.Canvas(
            parent,
            width=self.image_inputsize[1] * 16,
            height=self.image_inputsize[0] * 16,
            bg="white",
            highlightthickness=1,
            highlightbackground="#555555",
        )
        self.preview.pack(padx=12, pady=(4, 12), anchor="w")

        self.preview_rects = []
        for row in range(self.image_inputsize[0]):
            rect_row = []
            for col in range(self.image_inputsize[1]):
                rect = self.preview.create_rectangle(
                    col * 16,
                    row * 16,
                    (col + 1) * 16,
                    (row + 1) * 16,
                    outline="#d0d0d0",
                    fill="#ffffff",
                )
                rect_row.append(rect)
            self.preview_rects.append(rect_row)

    def _build_right_panel(self, parent):
        self.feature_maps_panel = FeatureMapsPanel(
            parent,
            panel_width=self.feature_map_config.panel_width,
            panel_height=self.feature_map_config.panel_height,
            map_scale=self.feature_map_config.map_scale,
            map_padding=self.feature_map_config.map_padding,
            section_padding=self.feature_map_config.section_padding,
            max_columns=self.feature_map_config.max_columns,
        )
        self.feature_maps_panel.pack(fill="both", expand=True, padx=12, pady=12)

    def update_preview(self, small_image: np.ndarray):
        for row in range(self.image_inputsize[0]):
            for col in range(self.image_inputsize[1]):
                value = float(np.clip(small_image[row, col], 0.0, 1.0))
                gray = int(round(255 * (1.0 - value)))
                color = f"#{gray:02x}{gray:02x}{gray:02x}"
                self.preview.itemconfig(self.preview_rects[row][col], fill=color)

    def update_feature_maps(self, feature_maps):
        self.feature_maps_panel.render(feature_maps)

    def clear_feature_maps(self):
        self.feature_maps_panel.clear()
