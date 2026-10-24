# SPDX-License-Identifier: Apache-2.0
"""Mock native generation while exercising request translation and encoding."""

from collections.abc import Iterator
from unittest.mock import Mock, patch

import pytest

import sglang_omni.models.sensenova_u1.generation as generation_module
from sglang_omni.models.sensenova_u1.cache_dit import SenseNovaCacheDit


@pytest.fixture
def t2i_generate_mock() -> Iterator[Mock]:
    with patch.object(generation_module, "t2i_generate", autospec=True) as generate:
        yield generate.mock


@pytest.fixture
def it2i_generate_mock() -> Iterator[Mock]:
    with patch.object(generation_module, "it2i_generate", autospec=True) as generate:
        yield generate.mock


@pytest.fixture
def cache_dit() -> SenseNovaCacheDit:
    return SenseNovaCacheDit(enabled_by_default=False, default_params=None)
