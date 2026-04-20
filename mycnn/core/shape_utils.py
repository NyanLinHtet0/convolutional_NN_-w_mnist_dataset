# shape_utils.py
def normalize_pair(value, name):
    if isinstance(value, int):
        return int(value), int(value)

    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[0]), int(value[1])

    raise ValueError(f"{name} must be an int or a length-2 tuple/list.")
    