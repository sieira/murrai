from dataclasses import dataclass


@dataclass(frozen=True)
class Shape:
    depth: int | None = None
    dimension: int | None = None
    max_time: int | None = None
    heads: int | None = None
    window: int | None = None
    hidden: int | None = None

    def reject_contradictory(self, **knobs):
        mismatches = [
            f"{name}={value}"
            for name, value in knobs.items()
            if value is not None
            and getattr(self, name) is not None
            and getattr(self, name) != value
        ]
        if mismatches:
            raise ValueError(
                f"a checkpoint keeps its own shape; pass no {', '.join(mismatches)}",
            )
