# SPDX-License-Identifier: Apache-2.0
"""Compare installed distributions with their declared requirements, as pip check does, and fail only on a conflict outside the reviewed list.

Run it with the interpreter of the environment to check. The reviewed list and
the reason each entry is harmless live in docs/get_started/troubleshooting.md.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib.metadata import distributions

from packaging.requirements import Requirement
from packaging.tags import Tag, parse_tag, sys_tags
from packaging.utils import canonicalize_name

# Note (ruoyu): (distribution, required distribution) pairs whose declared range the project leaves unmet on purpose.
KNOWN_METADATA_CONFLICTS: frozenset[tuple[str, str]] = frozenset(
    {
        ("qwen-tts", "transformers"),
        ("qwen-tts", "accelerate"),
        ("qwen-tts", "onnxruntime"),
        ("descript-audiotools", "protobuf"),
        ("grpcio-health-checking", "protobuf"),
        ("grpcio-reflection", "protobuf"),
    }
)
# Note (ruoyu): the decord 0.6.0 wheel records a cp36 tag although it loads libdecord.so through ctypes.
KNOWN_PLATFORM_CONFLICTS: frozenset[str] = frozenset({"decord"})


@dataclass(frozen=True, kw_only=True)
class InstalledDistribution:
    name: str
    version: str
    requirements: tuple[str, ...]
    wheel_tags: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class PlatformConflict:
    distribution: str
    version: str
    wheel_tags: tuple[str, ...]

    def describe(self) -> str:
        return f"{self.distribution} {self.version} is not supported on this platform, its wheel is tagged {' '.join(self.wheel_tags)}."


@dataclass(frozen=True, kw_only=True)
class RequirementConflict:
    distribution: str
    version: str
    requirement: str
    installed_version: str | None

    @property
    def required_name(self) -> str:
        return canonicalize_name(Requirement(self.requirement).name)

    def describe(self) -> str:
        if self.installed_version is None:
            return f"{self.distribution} {self.version} requires {self.requirement}, which is not installed."
        else:
            return f"{self.distribution} {self.version} has requirement {self.requirement}, but you have {self.required_name} {self.installed_version}."


def installed_distributions() -> list[InstalledDistribution]:
    environment: list[InstalledDistribution] = []
    for distribution in distributions():
        wheel_file = distribution.read_text("WHEEL")
        if wheel_file is None:
            wheel_tags: tuple[str, ...] = ()
        else:
            wheel_tags = tuple(
                line.split(":", 1)[1].strip()
                for line in wheel_file.splitlines()
                if line.startswith("Tag:")
            )
        environment.append(
            InstalledDistribution(
                name=canonicalize_name(distribution.metadata["Name"]),
                version=distribution.version,
                requirements=tuple(distribution.requires or ()),
                wheel_tags=wheel_tags,
            )
        )
    return environment


def first_installed_copies(
    environment: list[InstalledDistribution],
) -> list[InstalledDistribution]:
    # Note (ruoyu): pip's distribution finder skips every copy of a name after the first on sys.path, so later copies never reach pip check.
    first_copies: dict[str, InstalledDistribution] = {}
    for distribution in environment:
        first_copies.setdefault(distribution.name, distribution)
    return list(first_copies.values())


def find_conflicts(
    environment: list[InstalledDistribution],
) -> list[RequirementConflict]:
    environment = first_installed_copies(environment)
    installed_versions = {
        distribution.name: distribution.version for distribution in environment
    }
    conflicts: list[RequirementConflict] = []
    for distribution in environment:
        for requirement_text in distribution.requirements:
            requirement = Requirement(requirement_text)
            installed_version = installed_versions.get(
                canonicalize_name(requirement.name)
            )
            # Note (ruoyu): pip check evaluates markers with no extra selected, so a requirement behind an extra is skipped.
            if requirement.marker is not None and not requirement.marker.evaluate(
                {"extra": ""}
            ):
                pass
            elif installed_version is None:
                conflicts.append(
                    RequirementConflict(
                        distribution=distribution.name,
                        version=distribution.version,
                        requirement=requirement_text,
                        installed_version=None,
                    )
                )
            elif requirement.specifier.contains(installed_version, prereleases=True):
                pass
            else:
                conflicts.append(
                    RequirementConflict(
                        distribution=distribution.name,
                        version=distribution.version,
                        requirement=requirement_text,
                        installed_version=installed_version,
                    )
                )
    return conflicts


def find_platform_conflicts(
    environment: list[InstalledDistribution], supported_tags: frozenset[Tag]
) -> list[PlatformConflict]:
    conflicts: list[PlatformConflict] = []
    for distribution in first_installed_copies(environment):
        recorded_tags: set[Tag] = set()
        for wheel_tag in distribution.wheel_tags:
            recorded_tags.update(parse_tag(wheel_tag))
        if recorded_tags and recorded_tags.isdisjoint(supported_tags):
            conflicts.append(
                PlatformConflict(
                    distribution=distribution.name,
                    version=distribution.version,
                    wheel_tags=distribution.wheel_tags,
                )
            )
        else:
            pass
    return conflicts


def unexpected_conflicts(
    conflicts: list[RequirementConflict],
) -> list[RequirementConflict]:
    return [
        conflict
        for conflict in conflicts
        if (conflict.distribution, conflict.required_name)
        not in KNOWN_METADATA_CONFLICTS
    ]


def unexpected_platform_conflicts(
    conflicts: list[PlatformConflict],
) -> list[PlatformConflict]:
    return [
        conflict
        for conflict in conflicts
        if conflict.distribution not in KNOWN_PLATFORM_CONFLICTS
    ]


def main() -> int:
    environment = installed_distributions()
    conflicts = find_conflicts(environment)
    platform_conflicts = find_platform_conflicts(environment, frozenset(sys_tags()))
    unexpected = unexpected_conflicts(conflicts) + unexpected_platform_conflicts(
        platform_conflicts
    )
    for conflict in conflicts + platform_conflicts:
        if conflict in unexpected:
            print(f"unexpected: {conflict.describe()}", file=sys.stderr)
        else:
            print(f"reviewed: {conflict.describe()}")
    if unexpected:
        print(
            f"{len(unexpected)} requirement conflict(s) outside the reviewed list",
            file=sys.stderr,
        )
        return 1
    else:
        print(
            f"Installed requirements are consistent apart from {len(conflicts) + len(platform_conflicts)} reviewed conflict(s)"
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
