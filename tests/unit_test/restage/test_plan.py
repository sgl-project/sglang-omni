import json

import pytest
import yaml

from sglang_omni.config.sources import sources_from_config_file
from sglang_omni.config.topology import compile_logical_processes
from sglang_omni.restage.calibration import (
    Constants,
    SharingConstants,
    StageConstants,
    process_capacity,
)
from sglang_omni.restage.plan import SearchSpace, load_plan, rank_shapes, write_plan
from tests.unit_test.restage.conftest_seed import pipeline


def constants(**overrides):
    fields = dict(
        model_path="dummy",
        gpu_name="test-gpu",
        gpu_mem_gib=80.0,
        context_tokens=250,
        audio_seconds=4.5,
        pipeline_throughput=None,
        stages={
            "engine": StageConstants(
                throughput=30.0,
                provenance="PREDICTED share 0.4",
                weights_gib=4.0,
                kv_bytes_per_token=20480,
                delta_s=0.05,
            ),
            "tail": StageConstants(
                throughput=20.0, provenance="PREDICTED share 0.6", weights_gib=1.0
            ),
        },
    )
    fields.update(overrides)
    return Constants(**fields)


def test_process_capacity_keeps_measured_pipeline_and_predicts_splits():
    config = pipeline()
    logical, stages = compile_logical_processes(config)
    engine = next(p for p in logical.processes if p.name == "engine")
    split = process_capacity(engine, stages, constants())
    assert split.throughput == 30.0
    assert split.provenance.startswith("PREDICTED")
    assert split.kv_bytes_per_token == 20480 and split.delta_s == 0.05
    merged_config = pipeline()
    for stage in merged_config.stages:
        if stage.gpu is not None:
            stage.process = "engine"
    logical, stages = compile_logical_processes(merged_config)
    whole = process_capacity(
        logical.processes[-1],
        stages,
        constants(pipeline_throughput=12.0, pipeline_provenance="MEASURED probe"),
    )
    assert whole.throughput == 12.0
    assert whole.provenance == "MEASURED probe"
    assert whole.weights_gib == 5.0


def test_ranking_orders_shapes_by_predicted_utility():
    config = pipeline()
    rows = rank_shapes(config, SearchSpace(devices=[0, 1]), constants())
    candidates = [row for row in rows if row["status"] == "candidate"]
    assert candidates and candidates[0]["rank"] == 1
    utilities = [row["predicted"]["utility"] for row in candidates]
    assert utilities == sorted(utilities, reverse=True)
    by_shape = {row["shape"]: row for row in candidates}
    # Note (Jiaxin Deng): the engine binds the split, so a dedicated engine GPU
    # plus a dedicated tail GPU beats both processes serialized on each GPU.
    assert by_shape["split_backbone1_tails1"]["predicted"]["utility"] == pytest.approx(
        20.0
    )
    assert by_shape["replicate_x2"]["predicted"]["utility"] == pytest.approx(
        2 / (1 / 30 + 1 / 20)
    )
    assert by_shape["replicate_x1"]["fractions"]["engine"] + by_shape["replicate_x1"][
        "fractions"
    ]["tail"] == pytest.approx(0.9, abs=0.01)
    assert all(row["predicted"]["provenance"] == "PREDICTED" for row in candidates)


def test_dimensions_may_not_touch_shape_owned_fields():
    space = SearchSpace(
        devices=[0], dimensions={"mem": [{"engine.gpu_memory_fraction": 0.5}]}
    )
    with pytest.raises(ValueError, match="shapes own"):
        rank_shapes(pipeline(), space, constants())


def test_write_plan_exports_baseline_and_top_candidates(tmp_path):
    config = pipeline()
    destination = tmp_path / "plan"
    summary = write_plan(
        config, SearchSpace(devices=[0, 1]), constants(), destination, top=2
    )
    assert summary["candidates"] >= 4 and summary["rejected"] == 0
    assert len(summary["exported"]) == 2
    configs, predicted = load_plan(destination)
    assert list(configs)[0] == "baseline"
    ranking_rows = json.loads((destination / "ranking.json").read_text())
    baseline_row = next(
        r for r in ranking_rows if r.get("config_file") == "baseline.yaml"
    )
    assert baseline_row["shape"] == "replicate_x1"
    assert set(predicted) == set(configs)
    ranking = json.loads((destination / "ranking.json").read_text())
    assert all("config" not in row for row in ranking)
    top = yaml.safe_load((destination / summary["exported"][0]).read_text())
    assert top["processes"] or top["stages"]
    with pytest.raises(FileExistsError):
        write_plan(config, SearchSpace(devices=[0, 1]), constants(), destination)


def test_qwen3_tts_split_candidates_reload_through_the_serving_loader(tmp_path):
    from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig

    config = Qwen3TTSPipelineConfig(model_path="unused-checkpoint")
    tts_constants = constants(
        stages={
            "tts_engine": StageConstants(
                throughput=30.0,
                provenance="PREDICTED share 0.7",
                weights_gib=3.4,
                kv_bytes_per_token=20480,
                delta_s=0.08,
            ),
            "vocoder": StageConstants(
                throughput=70.0, provenance="PREDICTED share 0.3", weights_gib=1.0
            ),
        },
        pipeline_throughput=21.0,
        pipeline_provenance="MEASURED probe",
    )
    space = SearchSpace(
        devices=[0, 1],
        dimensions={"split": [{}, {"vocoder.process": "vocoder"}]},
    )
    destination = tmp_path / "plan"
    summary = write_plan(config, space, tts_constants, destination, top=3)
    assert summary["rejected"] == 0
    rows = json.loads((destination / "ranking.json").read_text())
    merged = [
        r for r in rows if not r["selections"]["split"] and r["status"] == "candidate"
    ]
    split = [r for r in rows if r["selections"]["split"] and r["status"] == "candidate"]
    assert {r["shape"] for r in merged} == {
        "replicate_x1",
        "replicate_x2",
        "colocate_x2_timeslice",
        "colocate_x2_mps",
        "colocate_x3_timeslice",
        "colocate_x3_mps",
        "colocate_x4_timeslice",
        "colocate_x4_mps",
    }
    assert "split_backbone1_tails1" in {r["shape"] for r in split}
    assert not any(r["shape"].startswith("consolidate") for r in split)
    exported = [r for r in rows if r.get("config_file")]
    assert len(
        {(r["shape"], round(r["predicted"]["utility"], 6)) for r in exported}
    ) == len(exported)
    merged_x1 = next(r for r in merged if r["shape"] == "replicate_x1")
    assert merged_x1["predicted"]["provenance"] == "MEASURED"
    assert merged_x1["predicted"]["utility"] == pytest.approx(21.0)
    for path in [destination / summary["baseline"]] + [
        destination / f for f in summary["exported"]
    ]:
        loaded = sources_from_config_file(str(path))
        assert loaded is not None, path
        data = yaml.safe_load(path.read_text())
        engine = data["stages"]["tts_engine"]
        assert engine["gpu_memory_fraction"] == engine["engine"]["mem_fraction_static"]


def test_colocated_candidates_pass_the_placement_memory_check(tmp_path):
    rows = rank_shapes(pipeline(), SearchSpace(devices=[0]), constants())
    shapes = {row.get("shape"): row for row in rows}
    assert shapes["colocate_x2_timeslice"]["status"] == "candidate", rows
    total = 2 * sum(shapes["colocate_x2_timeslice"]["fractions"].values())
    assert total <= 0.91


def test_measured_sharing_discount_flips_only_its_fan_in_to_measured(tmp_path):
    from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig

    config = Qwen3TTSPipelineConfig(model_path="unused-checkpoint")
    stages = {
        "tts_engine": StageConstants(
            throughput=30.0,
            provenance="PREDICTED share 0.7",
            weights_gib=3.4,
            kv_bytes_per_token=20480,
            delta_s=0.08,
        ),
        "vocoder": StageConstants(
            throughput=70.0, provenance="PREDICTED share 0.3", weights_gib=1.0
        ),
    }
    measured = constants(
        stages=stages,
        pipeline_throughput=21.0,
        pipeline_provenance="MEASURED probe",
        sharing_discounts={
            "mps@2": SharingConstants(value=0.7, provenance="MEASURED c8")
        },
    )
    path = tmp_path / "constants.json"
    measured.save(path)
    rows = rank_shapes(config, SearchSpace(devices=[0]), Constants.load(path))
    by_shape = {r["shape"]: r["predicted"] for r in rows if r["status"] == "candidate"}
    assert by_shape["colocate_x2_mps"]["provenance"] == "MEASURED"
    assert by_shape["colocate_x2_mps"]["sharing"] == {
        "mps@2": {"value": 0.7, "provenance": "MEASURED c8"}
    }
    assert by_shape["colocate_x2_mps"]["utility"] == pytest.approx(2 * 0.7 * 21.0)
    assert by_shape["colocate_x3_mps"]["provenance"] == "PREDICTED"
    assert by_shape["colocate_x3_mps"]["sharing"]["mps@3"]["provenance"].startswith(
        "PRIOR"
    )
    assert by_shape["colocate_x2_timeslice"]["provenance"] == "PREDICTED"
