"""CPU-only pipeline configuration used by Restage contract tests."""

from sglang_omni.config.schema import PipelineConfig, StageConfig


def pipeline(tp=1):
    factory = "tests.unit_test.fixtures.pipeline_fakes.dummy_factory"
    return PipelineConfig(
        model_path="dummy",
        stages=[
            StageConfig(
                name="cpu", process="front", factory_path=factory, next="engine"
            ),
            StageConfig(
                name="engine",
                process="engine",
                factory_path=factory,
                gpu=list(range(tp)),
                tp_size=tp,
                next="tail",
            ),
            StageConfig(
                name="tail",
                process="tail",
                factory_path=factory,
                gpu=0,
                gpu_memory_fraction=0.4,
                terminal=True,
            ),
        ],
    )
