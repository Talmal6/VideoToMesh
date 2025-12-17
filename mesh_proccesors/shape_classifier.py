def infer_shape_type(class_name: str) -> str:
    name = (class_name or "").lower()

    # cylinder-ish
    if any(k in name for k in ["bottle", "cup", "can"]):
        return "cylinder"

    # default
    return "box"
