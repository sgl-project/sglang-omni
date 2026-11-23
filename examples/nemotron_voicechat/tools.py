# SPDX-License-Identifier: Apache-2.0
"""Example tool functions for the VoiceChat tool graph."""

import random
from collections.abc import Mapping


def generate_random_number(arguments: Mapping[str, object]) -> dict[str, object]:
    low, high = arguments["min"], arguments["max"]
    if not isinstance(low, int) or not isinstance(high, int) or low > high:
        raise ValueError("min and max must be integers with min <= max")
    else:
        return {"result": random.randint(low, high)}
