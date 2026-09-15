# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import torch

from tests.unit_test.fixtures import ar_session

bridge_env = ar_session.bridge_env


def test_hidden_state_requests_have_an_empty_native_reporting_object(bridge_env):
    output = SimpleNamespace(
        next_token_ids=torch.tensor([4]), host_token_ids=None, can_run_cuda_graph=False
    )
    result = bridge_env.scheduler._make_batch_result(output)
    assert result.logits_output.hidden_states is None
    assert result.logits_output.customized_info is None
    assert result.next_token_ids.tolist() == [4]
