# Draw maps CLI patch

Behavior:

```bash
python draw.py
```

Normal mode. No feature-map panel is created, and feature maps are not collected.

```bash
python draw.py maps
```

Feature-map mode. Shows all collected maps.

```bash
python draw.py maps 50
```

Feature-map mode. For each convolution layer, randomly selects `floor(total_maps * 50 / 100)` maps.

Files changed:

- `config.py`
- `app_runner.py`
- `ui/app.py`
- `ui/view.py`
- `ui/feature_maps.py`

Use `draw_py_snippet.py` only as a guide for your existing `draw.py`, because your current full `draw.py` was not included in the uploaded files.
