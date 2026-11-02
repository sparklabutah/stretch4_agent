"""Backends are imported lazily so `mock` never needs MuJoCo and nothing but `stretch` needs the robot libraries."""


def make_backend(name: str, **kw):
    if name == "mock":
        from .mock import MockBackend
        return MockBackend(**kw)
    if name == "sim":
        from .sim import SimBackend
        return SimBackend(**kw)
    if name == "stretch":
        from .stretch import StretchBackend
        return StretchBackend(**kw)
    raise ValueError(f"unknown backend {name!r}; choose mock, sim or stretch")
