"""Langevin-step steering weight schedules.

Maps outer sampler step index (0 .. num_steps-1) to a scalar that scales logit biases
and the bias L2 penalty in ``LangevinSampler``.

- **constant:** ``steer_w = weight_val`` every step (no min/max, no alpha in config).
- **linear** / **dab_linear:** ``steer_w(step) = weight_val + delta * t(step)`` with
  ``t`` in ``[0, 1]`` linear in step index (0 at first step, 1 at last when ``num_steps > 1``).
"""

from __future__ import annotations

from collections.abc import Callable


def build_step_weight_scheduler(cfg: dict) -> Callable[[int], float]:
    w_val = float(cfg.get("weight_val", 1.0))
    num_steps = max(int(cfg.get("num_steps", 1)), 1)

    ws = cfg.get("weight_schedule")
    if not isinstance(ws, dict):
        ws = {}
    name = str(ws.get("name", "constant")).strip().lower()
    delta = float(ws.get("delta", 0.0))

    if name == "constant":

        def schedule(_step: int) -> float:
            return w_val

    elif name in ("dab_linear", "linear"):

        def schedule(step: int) -> float:
            if num_steps <= 1:
                return w_val
            t = float(step) / float(num_steps - 1)
            return w_val + delta * t

    else:
        raise ValueError(
            f"Unknown weight_schedule.name={name!r}. "
            "Use: constant | dab_linear | linear"
        )

    return schedule
