#!/usr/bin/env python3
"""G1 pilot entry point — thin alias for :mod:`g1_mask_cost_pilot`.

Queue config ``003-g1-model-axis.yaml`` was written against ``scripts/g1_pilot.py``
with a ``--models`` axis; the implementation landed first as
``scripts/g1_mask_cost_pilot.py`` (config 002). Rather than fork the measurement
code — two copies of a measurement harness is how two papers' numbers stop being
comparable — this exposes the same ``main()`` under the expected name. Both
configs therefore run byte-identical measurement logic, and the model axis is
just ``--models a,b,c``.

    python3 scripts/g1_pilot.py --models qwen3:4b,granite3.1-dense:2b \
        --conditions unconstrained,gbnf --repeats 3
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from g1_mask_cost_pilot import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
