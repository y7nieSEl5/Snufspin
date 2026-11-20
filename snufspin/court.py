"""Metric tennis court, with the net at x = 11.885 and the surface at z = 0."""

LENGTH = 23.77
DOUBLES_WIDTH = 10.97
SINGLES_WIDTH = 8.23
SERVICE_DISTANCE = 6.40

COURT_MODEL = "tennis-metric-v1"
_X = {
    "baseline_a": 0.0,
    "service_a": LENGTH / 2 - SERVICE_DISTANCE,
    "service_b": LENGTH / 2 + SERVICE_DISTANCE,
    "baseline_b": LENGTH,
}
_Y = {
    "doubles_low": 0.0,
    "singles_low": (DOUBLES_WIDTH - SINGLES_WIDTH) / 2,
    "center": DOUBLES_WIDTH / 2,
    "singles_high": (DOUBLES_WIDTH + SINGLES_WIDTH) / 2,
    "doubles_high": DOUBLES_WIDTH,
}

LANDMARKS = {
    f"{end}_{side}": (_X[end], _Y[side], 0.0)
    for end in _X
    for side in (
        ("doubles_low", "singles_low", "singles_high", "doubles_high")
        if end.startswith("baseline")
        else ("singles_low", "center", "singles_high")
    )
}

# Painted markings only. The net is not a painted court line.
LINES = {
    "baseline_a": ("baseline_a_doubles_low", "baseline_a_doubles_high"),
    "baseline_b": ("baseline_b_doubles_low", "baseline_b_doubles_high"),
    **{side: (f"baseline_a_{side}", f"baseline_b_{side}")
       for side in _Y if side != "center"},
    "service_a": ("service_a_singles_low", "service_a_singles_high"),
    "service_b": ("service_b_singles_low", "service_b_singles_high"),
    "center_service": ("service_a_center", "service_b_center"),
}
