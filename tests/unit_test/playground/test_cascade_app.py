# SPDX-License-Identifier: Apache-2.0
"""The cascade playground's SMPL-X mesh blob, built at run time from a local model file (a synthetic one here)."""

from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer

APP_PATH = Path(__file__).resolve().parents[3] / "playground" / "cascade" / "app.py"
VERTICES = 6
JOINTS = 55


@pytest.fixture
def playground() -> ModuleType:
    spec = importlib.util.spec_from_file_location("cascade_playground", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def smplx_model(tmp_path: Path) -> Path:
    """An .npz of SMPL-X's layout: 6 vertices, 55 joints, 400 shape and expression directions of which the expressions move vertex 2 only."""
    rng = np.random.default_rng(0)
    weights = rng.random((VERTICES, JOINTS)).astype(np.float32)
    weights /= weights.sum(axis=1, keepdims=True)
    shapedirs = np.zeros((VERTICES, 3, 400))
    shapedirs[:, :, :300] = 1.0
    shapedirs[2, 1, 300] = 0.25
    kintree = np.zeros((2, JOINTS), dtype=np.int64)
    kintree[0] = [2**32 - 1] + list(range(JOINTS - 1))
    kintree[1] = np.arange(JOINTS)
    path = tmp_path / "SMPLX_TEST.npz"
    np.savez(
        path,
        v_template=rng.random((VERTICES, 3)).astype(np.float32),
        f=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32),
        J_regressor=np.full((JOINTS, VERTICES), 1 / VERTICES, dtype=np.float32),
        kintree_table=kintree,
        weights=weights,
        shapedirs=shapedirs,
    )
    return path


def unpack(blob: bytes) -> dict[str, np.ndarray]:
    (length,) = struct.unpack_from("<I", blob)
    header = json.loads(blob[4 : 4 + length])
    assert (4 + length) % 4 == 0
    arrays = {}
    for name, entry in header.items():
        assert entry["offset"] % 4 == 0
        dtype = np.dtype("<" + entry["dtype"])
        count = int(np.prod(entry["shape"]))
        start = 4 + length + entry["offset"]
        arrays[name] = np.frombuffer(blob, dtype, count, start).reshape(entry["shape"])
    return arrays


def test_mesh_blob_holds_the_template_joints_top_skin_weights_and_expressions(
    playground: ModuleType, smplx_model: Path
) -> None:
    arrays = unpack(playground.smplx_mesh_blob(smplx_model))
    model = np.load(smplx_model)
    np.testing.assert_array_equal(arrays["vertices"], model["v_template"])
    np.testing.assert_array_equal(arrays["faces"], model["f"])
    np.testing.assert_allclose(
        arrays["joints"],
        np.broadcast_to(model["v_template"].mean(0), (JOINTS, 3)),
        rtol=1e-5,
    )
    assert arrays["parents"][0] == -1 and list(arrays["parents"][1:4]) == [0, 1, 2]
    assert arrays["skin_indices"].shape == (VERTICES, 4)
    np.testing.assert_allclose(arrays["skin_weights"].sum(1), 1, rtol=1e-6)
    for vertex in range(VERTICES):
        largest = np.sort(model["weights"][vertex])[-4:]
        picked = model["weights"][vertex][arrays["skin_indices"][vertex]]
        np.testing.assert_allclose(np.sort(picked), largest)
    assert list(arrays["expression_vertices"]) == [2]
    assert arrays["expression_dirs"].shape == (1, 3, 100)
    assert arrays["expression_dirs"][0, 1, 0] == np.float16(0.25)


@pytest.mark.asyncio
async def test_the_page_gets_the_mesh_only_from_a_model_given_at_startup(
    playground: ModuleType, smplx_model: Path
) -> None:
    async with TestClient(
        TestServer(playground.create_app("http://127.0.0.1:9"))
    ) as client:
        assert (await client.get("/smplx/mesh.bin")).status == 404
    app = playground.create_app("http://127.0.0.1:9", str(smplx_model))
    async with TestClient(TestServer(app)) as client:
        response = await client.get("/smplx/mesh.bin")
        assert response.status == 200
        assert unpack(await response.read())["vertices"].shape == (VERTICES, 3)
