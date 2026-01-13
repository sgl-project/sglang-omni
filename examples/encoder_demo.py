# SPDX-License-Identifier: Apache-2.0
"""Demo: Encoder with runner + scheduler."""

from __future__ import annotations

import asyncio

import torch
import torch.nn as nn

from sglang_omni.engines.encoder import (
    EncoderEngine,
    EncoderRunner,
    Request,
    Scheduler,
    SchedulerConfig,
)


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleRunner(EncoderRunner):
    def __init__(
        self, device: str, dtype: torch.dtype, input_dim: int, output_dim: int
    ):
        super().__init__(device, dtype)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def load(self) -> None:
        self.model = SimpleMLP(self.input_dim, self.output_dim).to(
            self.device, self.dtype
        )
        self.model.eval()

    def prepare(self, requests: list[Request]) -> torch.Tensor:
        return torch.stack([r.data for r in requests]).to(self.device, self.dtype)

    @torch.inference_mode()
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)

    def unbatch(
        self, requests: list[Request], output: torch.Tensor
    ) -> list[torch.Tensor]:
        return [output[i].cpu() for i in range(len(requests))]


async def main():
    print("=== Encoder Demo ===\n")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    runner = SimpleRunner(device, torch.float32, input_dim=64, output_dim=256)
    scheduler = Scheduler(SchedulerConfig(max_batch_size=4, batch_timeout=0.05))
    engine = EncoderEngine(runner, scheduler)

    await engine.start()

    async def process(req_id: str, data: torch.Tensor) -> tuple[str, torch.Tensor]:
        await engine.add_request(req_id, data)
        return req_id, await engine.get_result(req_id)

    tasks = [process(f"req-{i}", torch.randn(64)) for i in range(10)]
    results = await asyncio.gather(*tasks)

    for req_id, result in results:
        print(f"{req_id}: [64] -> {list(result.shape)}")

    await engine.stop()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
