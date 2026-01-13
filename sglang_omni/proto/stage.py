# SPDX-License-Identifier: Apache-2.0
"""Stage information."""

from dataclasses import dataclass


@dataclass
class StageInfo:
    """Information about a registered stage."""

    name: str
    control_endpoint: str
