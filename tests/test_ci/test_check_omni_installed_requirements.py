import importlib.util
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).parents[2] / ".github/scripts/check_omni_installed_requirements.py"
)
SPEC = importlib.util.spec_from_file_location(
    "check_omni_installed_requirements", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
# The script's dataclasses resolve their module through sys.modules while the class body runs.
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def installed(name, version, *requirements):
    return MODULE.InstalledDistribution(
        name=name, version=version, requirements=tuple(requirements)
    )


def test_find_conflicts_matches_pip_check():
    """Report unmet pins and missing requirements, skip extras, accept installed pre-releases."""
    environment = [
        installed("qwen-tts", "0.1.1", "transformers==4.57.3", "sox", "onnxruntime"),
        installed("transformers", "5.19.0"),
        installed("sox", "1.5.0"),
        installed(
            "descript-audiotools",
            "0.7.2",
            "protobuf (<3.20,>=3.9.2)",
            "sphinx ; extra == 'docs'",
        ),
        installed("protobuf", "6.33.6"),
        installed("flashinfer-python", "0.7.1", "flash-attn-4>=4.0.0b18"),
        installed("flash-attn-4", "4.0.0b36"),
    ]

    assert MODULE.find_conflicts(environment) == [
        MODULE.RequirementConflict(
            distribution="qwen-tts",
            version="0.1.1",
            requirement="transformers==4.57.3",
            installed_version="5.19.0",
        ),
        MODULE.RequirementConflict(
            distribution="qwen-tts",
            version="0.1.1",
            requirement="onnxruntime",
            installed_version=None,
        ),
        MODULE.RequirementConflict(
            distribution="descript-audiotools",
            version="0.7.2",
            requirement="protobuf (<3.20,>=3.9.2)",
            installed_version="6.33.6",
        ),
    ]


def test_find_platform_conflicts_reads_the_recorded_wheel_tags():
    """A wheel whose recorded tags match none of the supported tags is unsupported, a source install has none."""
    supported = frozenset(
        MODULE.parse_tag("cp312-cp312-manylinux_2_34_x86_64")
        | MODULE.parse_tag("py3-none-any")
    )
    decord = MODULE.InstalledDistribution(
        name="decord",
        version="0.6.0",
        requirements=("numpy (>=1.14.0)",),
        wheel_tags=("cp36-cp36m-manylinux2010_x86_64",),
    )
    environment = [
        decord,
        MODULE.InstalledDistribution(
            name="qwen-tts",
            version="0.1.1",
            requirements=(),
            wheel_tags=("py3-none-any",),
        ),
        MODULE.InstalledDistribution(
            name="sglang-omni", version="0.1.9", requirements=(), wheel_tags=()
        ),
    ]

    assert MODULE.find_platform_conflicts(environment, supported) == [
        MODULE.PlatformConflict(
            distribution="decord",
            version="0.6.0",
            wheel_tags=("cp36-cp36m-manylinux2010_x86_64",),
        )
    ]
    assert (
        MODULE.unexpected_platform_conflicts(
            MODULE.find_platform_conflicts(environment, supported)
        )
        == []
    )


def test_first_installed_copy_decides_like_pip():
    """With a distribution installed twice, pip reads the first copy on sys.path, so its version, requirements and wheel tags decide."""
    supported = frozenset(MODULE.parse_tag("py3-none-any"))
    environment = [
        MODULE.InstalledDistribution(
            name="qwen-tts",
            version="0.1.1",
            requirements=("transformers==4.57.3",),
            wheel_tags=("py3-none-any",),
        ),
        MODULE.InstalledDistribution(
            name="transformers",
            version="5.19.0",
            requirements=(),
            wheel_tags=("py3-none-any",),
        ),
        MODULE.InstalledDistribution(
            name="transformers",
            version="4.57.3",
            requirements=("tokenizers<0.22",),
            wheel_tags=("cp36-cp36m-manylinux2010_x86_64",),
        ),
    ]

    assert MODULE.find_conflicts(environment) == [
        MODULE.RequirementConflict(
            distribution="qwen-tts",
            version="0.1.1",
            requirement="transformers==4.57.3",
            installed_version="5.19.0",
        )
    ]
    assert MODULE.find_platform_conflicts(environment, supported) == []


def test_unexpected_conflicts_leave_only_unreviewed_pairs():
    """A reviewed distribution and requirement pair passes, any other pair fails the check."""
    reviewed = MODULE.RequirementConflict(
        distribution="qwen-tts",
        version="0.1.1",
        requirement="transformers==4.57.3",
        installed_version="5.19.0",
    )
    reviewed_other_version = MODULE.RequirementConflict(
        distribution="qwen-tts",
        version="0.2.0",
        requirement="transformers==4.60.0",
        installed_version="5.19.0",
    )
    unreviewed = MODULE.RequirementConflict(
        distribution="qwen-tts",
        version="0.1.1",
        requirement="torch==2.8.0",
        installed_version="2.14.1",
    )

    assert MODULE.unexpected_conflicts(
        [reviewed, reviewed_other_version, unreviewed]
    ) == [unreviewed]
