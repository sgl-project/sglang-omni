"""FP8 linears for the Qwen3-ASR decoder on SM120 (RTX 50 series, RTX PRO 6000).

Loaded only on SM120 (see Qwen3ASRForConditionalGeneration.load_weights).

Every decoder linear (qkv, o_proj, gate_up, down) and the lm_head keeps one FP8 e4m3
weight with one per-tensor scale. Two kernels read that weight:

- decode (M <= 64 rows): weight-only kernel in this file. Decode is bound by weight
  reads, so W stays FP8 in DRAM and SMEM and is converted to bf16 in registers for
  the bf16 mma.sync. X stays bf16.
- prefill (M > 64 rows): W8A8 kernel in w8a8_linear.py. Prefill is compute bound, so
  X is quantized per token and the FP8 MMA runs at 4x the bf16 MMA rate.
"""

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.nn as nn
from cutlass.cute.runtime import from_dlpack, make_fake_stream

from sglang_omni.models.qwen3_asr import w8a8_linear

_CTA_TILER = (16, 64, 128)
_NUM_STAGES = 2


@cute.jit
def copy_X_tile(
    tiled_copy_X: cute.TiledCopy,
    tXgX_tile: cute.Tensor,
    tXsX_stage: cute.Tensor,
    is_full_block: cutlass.Boolean,
    block_first_row: cutlass.Int32,
    thread_row: cutlass.Int32,
    thread_rows: cutlass.Constexpr,
    num_rows: cutlass.Int32,
):
    if is_full_block:
        cute.copy(tiled_copy_X, tXgX_tile, tXsX_stage)
    else:
        # Rows past M stay stale in SMEM, and their output is never stored
        for copy_row in cutlass.range_constexpr(cute.size(tXsX_stage, mode=[1])):
            row = block_first_row + thread_row + copy_row * thread_rows
            if row < num_rows:
                cute.copy(
                    tiled_copy_X,
                    tXgX_tile[None, copy_row, None],
                    tXsX_stage[None, copy_row, None],
                )
            else:
                pass


@cute.kernel
def fp8_linear_kernel(
    mX: cute.Tensor,
    mW: cute.Tensor,
    mScale: cute.Tensor,
    mY: cute.Tensor,
    sX_layout: cute.Layout,
    sW_layout: cute.Layout,
    sX_swizzle: cute.Swizzle,
    sW_swizzle: cute.Swizzle,
    tiled_copy_X: cute.TiledCopy,
    tiled_copy_W: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
    cta_tiler: cutlass.Constexpr = _CTA_TILER,
    num_stages: cutlass.Constexpr = _NUM_STAGES,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    gX = cute.local_tile(mX, cta_tiler, (bidx, bidy, None), proj=(1, None, 1))
    gW = cute.local_tile(mW, cta_tiler, (bidx, bidy, None), proj=(None, 1, 1))
    gY = cute.local_tile(mY, cta_tiler, (bidx, bidy, None), proj=(1, 1, None))
    # mScale is an (M, N) view with stride (0, 1): every token sees the same row scale
    gScale = cute.local_tile(mScale, cta_tiler, (bidx, bidy, None), proj=(1, 1, None))

    @cute.struct
    class SharedStorage:
        x: cute.struct.Align[
            cute.struct.MemRange[mX.element_type, cute.cosize(sX_layout)], 16
        ]
        w: cute.struct.Align[
            cute.struct.MemRange[mW.element_type, cute.cosize(sW_layout)], 16
        ]

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage.size_in_bytes(), byte_alignment=16)
    sX = SharedStorage(storage).x.get_tensor(sX_layout, swizzle=sX_swizzle)
    sW = SharedStorage(storage).w.get_tensor(sW_layout, swizzle=sW_swizzle)

    thr_copy_X = tiled_copy_X.get_slice(tidx)
    thr_copy_W = tiled_copy_W.get_slice(tidx)

    tXgX = thr_copy_X.partition_S(gX)
    tWgW = thr_copy_W.partition_S(gW)

    tXsX = thr_copy_X.partition_D(sX)
    tWsW = thr_copy_W.partition_D(sW)

    k_tile_count = cute.size(tXgX, mode=[3])

    thr_mma = tiled_mma.get_slice(tidx)

    tCsX = thr_mma.partition_A(sX)
    tCsW = thr_mma.partition_B(sW)
    tCgY = thr_mma.partition_C(gY)
    tCgScale = thr_mma.partition_C(gScale)

    tCrX = tiled_mma.make_fragment_A(tCsX[None, None, None, 0])
    tCrC = tiled_mma.make_fragment_C(tCgY)

    # W lives twice in registers: the fp8 copy from SMEM and the bf16 copy for the MMA.
    # Both use the B fragment layout of the MMA, only the element type differs.
    tCrW = cute.make_fragment_like(
        tiled_mma.make_fragment_B(tCsW[None, None, None, 0]),
        cutlass.BFloat16,
    )
    tCrW_fp8 = cute.make_fragment_like(tCrW, mW.element_type)

    tCrC.fill(0.0)

    # X uses ldmatrix. W is 8-bit and does not match the 16-bit ldmatrix order,
    # so W goes SMEM -> registers with a plain autovec_copy.
    atom_copy_s2r_X = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4),
        mX.element_type,
    )
    tiled_copy_s2r_X = cute.make_tiled_copy_A(atom_copy_s2r_X, tiled_mma)
    thr_copy_ldmatrix_X = tiled_copy_s2r_X.get_slice(tidx)

    tCsX_copy_view = thr_copy_ldmatrix_X.partition_S(sX)
    tCrX_copy_view = thr_copy_ldmatrix_X.retile(tCrX)

    num_k_block = cute.size(tCrX, mode=[2])

    # X rows owned by this thread are thread_row + copy_row * thread_rows
    thread_rows = cta_tiler[0] // cute.size(tXsX, mode=[1])
    thread_row = tidx // (cute.size(tiled_mma) // thread_rows)
    block_first_row = bidx * cta_tiler[0]
    is_full_block = block_first_row + cta_tiler[0] <= mX.shape[0]

    # prologue: tiles t0..t(S-1) -> slots 0..S-1, one commit group per tile.
    for stage in cutlass.range_constexpr(num_stages):
        if stage < k_tile_count:
            copy_X_tile(
                tiled_copy_X,
                tXgX[None, None, None, stage],
                tXsX[None, None, None, stage],
                is_full_block,
                block_first_row,
                thread_row,
                thread_rows,
                mX.shape[0],
            )
            cute.copy(
                tiled_copy_W,
                tWgW[None, None, None, stage],
                tWsW[None, None, None, stage],
            )
        else:
            pass
        cute.arch.cp_async_commit_group()

    # SYNC-A: tile t0 is in SMEM
    cute.arch.cp_async_wait_group(num_stages - 1)
    cute.arch.sync_threads()

    for k_tile in range(k_tile_count):
        slot = k_tile % num_stages
        tCsX_p = tCsX_copy_view[None, None, None, slot]
        tCsW_p = tCsW[None, None, None, slot]

        # W: SMEM fp8 -> registers fp8 -> registers bf16, for the whole k_tile
        cute.autovec_copy(tCsW_p, tCrW_fp8)
        tCrW.store(tCrW_fp8.load().to(cutlass.BFloat16))

        for k_block in cutlass.range(num_k_block, unroll_full=True):
            cute.copy(
                tiled_copy_s2r_X,
                tCsX_p[None, None, k_block],
                tCrX_copy_view[None, None, k_block],
            )
            cute.gemm(
                tiled_mma,
                tCrC,
                tCrX[None, None, k_block],
                tCrW[None, None, k_block],
                tCrC,
            )

        # SYNC-B: every thread is done with this slot
        cute.arch.sync_threads()

        if k_tile + num_stages < k_tile_count:
            copy_X_tile(
                tiled_copy_X,
                tXgX[None, None, None, k_tile + num_stages],
                tXsX[None, None, None, slot],
                is_full_block,
                block_first_row,
                thread_row,
                thread_rows,
                mX.shape[0],
            )
            cute.copy(
                tiled_copy_W,
                tWgW[None, None, None, k_tile + num_stages],
                tWsW[None, None, None, slot],
            )
        else:
            pass
        cute.arch.cp_async_commit_group()

        # SYNC-C: tile t(k+1) is in SMEM
        cute.arch.cp_async_wait_group(num_stages - 1)
        cute.arch.sync_threads()

    # epilogue: the scale does not depend on k, so it is applied once after the sum
    tCrScale = cute.make_fragment_like(tCrC)
    cute.autovec_copy(tCgScale, tCrScale)

    tCrY = cute.make_fragment_like(tCrC, mY.element_type)
    tCrY.store((tCrC.load() * tCrScale.load()).to(mY.element_type))

    tCgCoord = thr_mma.partition_C(
        cute.make_identity_tensor((cta_tiler[0], cta_tiler[1]))
    )
    if is_full_block:
        cute.autovec_copy(tCrY, tCgY)
    else:
        for index in cutlass.range(cute.size(tCrY), unroll_full=True):
            row = block_first_row + tCgCoord[index][0]
            if row < mY.shape[0]:
                tCgY[index] = tCrY[index]
            else:
                pass


# Decode configs, measured on the 5080 (gpu-paradise/qwen3-asr-fp8-gemm sweeps).
# (N, K) -> (config for M <= 16, config for 16 < M <= 64), config = (cta_tiler, num_stages).
# Above 16 rows bM = 32: bM = 64 gives one CTA too much M work on small N.
_CONFIG_BY_SHAPE = {
    (4096, 2048): (((16, 32, 64), 3), ((32, 32, 128), 2)),  # qkv
    (2048, 2048): (((16, 32, 128), 3), ((32, 32, 256), 2)),  # o_proj
    (12288, 2048): (((16, 64, 64), 2), ((32, 64, 64), 2)),  # gate_up
    (2048, 6144): (((16, 64, 256), 4), ((32, 32, 256), 2)),  # down
    (151936, 2048): (((16, 32, 64), 2), ((64, 64, 64), 2)),  # lm_head
}


def pick_config(M, N, K):
    if (N, K) in _CONFIG_BY_SHAPE:
        small_m_config, large_m_config = _CONFIG_BY_SHAPE[(N, K)]
    else:
        small_m_config = ((16, 64, 128), 2)
        large_m_config = ((32, 64, 128), 2)
    if M <= 16:
        return small_m_config
    else:
        return large_m_config


_MMA_SHAPE = (16, 8, 16)


# The swizzle works on bytes: shift = log2(row bytes) - base.
# This gives 3 for a 128-byte row of bf16 or fp8.
def make_smem_layout_ab(dtype, copy_bits, smem_tiler):
    major_size = min(smem_tiler[1], 128 * 8 // dtype.width)
    swizzle_bits = min(int(math.log2(major_size * dtype.width // copy_bits)), 3)
    base_bits = int(math.log2(copy_bits // 8))
    shift_bits = int(math.log2(major_size * dtype.width // 8)) - base_bits
    swizzle = cute.make_swizzle(swizzle_bits, base_bits, shift_bits)
    atom = cute.make_layout((8, major_size), stride=(major_size, 1))
    layout = cute.tile_to_shape(atom, smem_tiler, (0, 1, 2))
    return layout, swizzle


def make_gmem_tiled_copy(atom_copy, dtype, copy_bits, minor_size, num_threads):
    copy_elems = copy_bits // dtype.width
    shape_dim_1 = minor_size // copy_elems
    thread_layout = cute.make_layout(
        (num_threads // shape_dim_1, shape_dim_1), stride=(shape_dim_1, 1)
    )
    value_layout = cute.make_layout((1, copy_elems))
    return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)


@cute.jit
def fp8_linear_host(
    mX: cute.Tensor,
    mW: cute.Tensor,
    mScale: cute.Tensor,
    mY: cute.Tensor,
    stream: cuda.CUstream,
    cta_tiler: cutlass.Constexpr = _CTA_TILER,
    num_stages: cutlass.Constexpr = _NUM_STAGES,
):
    bM, bN, bK = cta_tiler
    # M is one atom row (16), each warp owns 16 columns of N
    num_warps = bN // 16
    atom_layout_mnk = (1, num_warps, 1)
    num_threads = num_warps * 32
    M = cute.size(mY, mode=[0])
    N = cute.size(mY, mode=[1])

    # 1. scale [N] -> [M, N] view, stride 0 along M
    mScale = cute.make_tensor(mScale.iterator, cute.make_layout((M, N), stride=(0, 1)))

    # 2. Tiled MMA: bf16 atom, warps along N
    op = cute.nvgpu.warp.MmaF16BF16Op(mX.element_type, cutlass.Float32, _MMA_SHAPE)
    permutation_mnk = (
        atom_layout_mnk[0] * _MMA_SHAPE[0],
        atom_layout_mnk[1] * _MMA_SHAPE[1] * 2,
        atom_layout_mnk[2] * _MMA_SHAPE[2],
    )
    tiled_mma = cute.make_tiled_mma(
        op, cute.make_layout(atom_layout_mnk), permutation_mnk=permutation_mnk
    )

    # 3. SMEM per stage: X bM x bK bf16 + W bN x bK fp8
    copy_bits = 128

    # One pass of the X copy must fit in the X tile. Example: bM = 16, bK = 64 and
    # 8 warps copy 32 rows per pass into a 16-row tile, which gives wrong results.
    x_copy_elems = copy_bits // mX.element_type.width
    assert num_threads * x_copy_elems <= bM * bK, "X copy pass larger than the X tile"
    # Same rule for W.
    w_copy_elems = copy_bits // mW.element_type.width
    assert num_threads * w_copy_elems <= bN * bK, "W copy pass larger than the W tile"

    sX_layout, sX_swizzle = make_smem_layout_ab(
        mX.element_type, copy_bits, (bM, bK, num_stages)
    )
    sW_layout, sW_swizzle = make_smem_layout_ab(
        mW.element_type, copy_bits, (bN, bK, num_stages)
    )

    # 4. g2s async copy: 16 bytes = 8 bf16 (X) or 16 fp8 (W) elements
    atom_g2s_X = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL),
        mX.element_type,
        num_bits_per_copy=copy_bits,
    )
    atom_g2s_W = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL),
        mW.element_type,
        num_bits_per_copy=copy_bits,
    )
    tiled_copy_X = make_gmem_tiled_copy(
        atom_g2s_X, mX.element_type, copy_bits, bK, num_threads
    )
    tiled_copy_W = make_gmem_tiled_copy(
        atom_g2s_W, mW.element_type, copy_bits, bK, num_threads
    )

    # 5. Launch: grid = output tiles. M is dynamic, the last tile can be partial.
    fp8_linear_kernel(
        mX,
        mW,
        mScale,
        mY,
        sX_layout,
        sW_layout,
        sX_swizzle,
        sW_swizzle,
        tiled_copy_X,
        tiled_copy_W,
        tiled_mma,
        cta_tiler,
        num_stages,
    ).launch(
        grid=[cute.ceil_div(M, bM), N // bN, 1],
        block=[num_threads, 1, 1],
        stream=stream,
    )


_FP8_MAX = 448.0
# Decode batches have at most this many rows. Above it the call is prefill.
_DECODE_MAX_ROWS = 64
_DECODER_LINEAR_NAMES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")
_COMPILED_KERNELS: dict[tuple, object] = {}


def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    """bf16 [N, K] -> fp8 [N, K], fp32 scale vector [N], fp32 scale.

    One per-tensor scale. The weight-only kernel reads a per-row scale vector,
    so the vector holds the same value N times.
    """
    amax = weight.float().abs().max().clamp(min=1e-12)
    scale = amax / _FP8_MAX
    weight_fp8 = (weight.float() / scale).to(torch.float8_e4m3fn).contiguous()
    scale_vector = scale.expand(weight.shape[0]).contiguous()
    return weight_fp8, scale_vector, scale.item()


def wrap_rows(tensor: torch.Tensor) -> cute.Tensor:
    # Only M is dynamic, so one compiled kernel serves every batch size
    return from_dlpack(tensor, assumed_align=16).mark_compact_shape_dynamic(
        mode=0, stride_order=(0, 1), divisibility=1
    )


def wrap_weight(weight_fp8: torch.Tensor) -> cute.Tensor:
    # FP8 DLPack is not in every torch version: byte view + element_type
    mW = from_dlpack(weight_fp8.view(torch.uint8), assumed_align=16)
    mW.element_type = cutlass.Float8E4M3FN
    return mW


def get_compiled(M: int, N: int, K: int, device: torch.device):
    config = pick_config(M, N, K)
    key = (N, K, config)
    if key in _COMPILED_KERNELS:
        return _COMPILED_KERNELS[key]
    else:
        pass
    # M is dynamic, so the sample M does not matter
    x = torch.empty(M, K, device=device, dtype=torch.bfloat16)
    y = torch.empty(M, N, device=device, dtype=torch.bfloat16)
    weight_fp8 = torch.empty(N, K, device=device, dtype=torch.float8_e4m3fn)
    scale = torch.empty(N, device=device, dtype=torch.float32)
    tile, num_stages = config
    _COMPILED_KERNELS[key] = cute.compile(
        fp8_linear_host,
        wrap_rows(x),
        wrap_weight(weight_fp8),
        from_dlpack(scale, assumed_align=16),
        wrap_rows(y),
        make_fake_stream(),
        tile,
        num_stages,
    )
    return _COMPILED_KERNELS[key]


def precompile(N: int, K: int, device: torch.device) -> None:
    # Compile both decode groups and the prefill kernel before CUDA graph capture
    for M in [1, _DECODE_MAX_ROWS]:
        get_compiled(M, N, K, device)
    w8a8_linear.precompile(N, K, device)


def fp8_weight_only_linear(
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    scale_vector: torch.Tensor,
) -> torch.Tensor:
    """bf16 X [M, K] @ (fp8 W [N, K] * scale)^T -> bf16 Y [M, N]. For decode."""
    x = x.contiguous()
    M = x.shape[0]
    N, K = weight_fp8.shape
    y = torch.empty(M, N, device=x.device, dtype=torch.bfloat16)
    if M == 0:
        return y
    else:
        pass
    compiled = get_compiled(M, N, K, x.device)
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    compiled(
        wrap_rows(x),
        wrap_weight(weight_fp8),
        from_dlpack(scale_vector, assumed_align=16),
        wrap_rows(y),
        stream,
    )
    return y


class Fp8LinearMethod:
    """quant_method for a module that holds weight (fp8), weight_scale_vector and weight_scale."""

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        # The loader calls this on every quant_method. The weights are ready already.
        pass

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        leading_shape = x.shape[:-1]
        x_2d = x.reshape(-1, x.shape[-1])
        if x_2d.shape[0] > _DECODE_MAX_ROWS:
            y = w8a8_linear.w8a8_linear(x_2d, layer.weight, layer.weight_scale)
        else:
            y = fp8_weight_only_linear(x_2d, layer.weight, layer.weight_scale_vector)
        if bias is not None:
            y = y + bias
        else:
            pass
        return y.reshape(*leading_shape, y.shape[-1])


def set_fp8_weight(module: nn.Module, weight: torch.Tensor) -> None:
    weight_fp8, scale_vector, scale = quantize_weight(weight)
    module.register_buffer("weight", weight_fp8, persistent=False)
    module.register_buffer("weight_scale_vector", scale_vector, persistent=False)
    module.weight_scale = scale
    module.quant_method = Fp8LinearMethod()
    precompile(weight_fp8.shape[0], weight_fp8.shape[1], weight_fp8.device)


class Fp8LMHead(nn.Module):
    """FP8 copy of the tied lm_head. The embedding table itself stays bf16."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        set_fp8_weight(self, weight.detach())


def install_fp8_linears(language_model: nn.Module) -> int:
    """Swap lm_head and qkv / o / gate_up / down of every decoder layer to FP8.

    Only language_model.model is walked, so the audio tower is never touched.
    """
    language_model.lm_head = Fp8LMHead(language_model.lm_head.weight)
    count = 0
    for name, module in language_model.model.named_modules():
        short_name = name.split(".")[-1]
        if short_name not in _DECODER_LINEAR_NAMES:
            continue
        else:
            pass
        weight = module.weight.detach()
        # Drop the bf16 parameter, so its memory goes back to the KV cache
        del module.weight
        set_fp8_weight(module, weight)
        count += 1
    for layer in language_model.model.layers:
        # SM100 DualGemm reads gate_up_proj.weight directly and skips quant_method
        layer.mlp.dual_gemm.mode = None
    repack_on_device(language_model)
    return count


def repack_on_device(module: nn.Module) -> None:
    """Move every tensor of module to the host and back, in fresh device memory.

    The FP8 weights and the quantize temporaries land in the allocator segments of
    the freed bf16 weights, so empty_cache cannot give those segments back (about
    1 GB on Qwen3-ASR-1.7B). After this round trip the tensors are packed and the
    KV cache gets the memory.
    """
    tensors = []
    seen_ids = set()
    candidates = list(module.parameters()) + list(module.buffers())
    for tensor in candidates:
        if not tensor.is_cuda or id(tensor) in seen_ids:
            continue
        else:
            seen_ids.add(id(tensor))
            tensors.append(tensor)

    host_copies = []
    for tensor in tensors:
        host_copies.append(tensor.data.to("cpu"))
        tensor.data = torch.empty(0, device=tensor.device, dtype=tensor.dtype)
    torch.cuda.empty_cache()

    for tensor, host_copy in zip(tensors, host_copies):
        tensor.data = host_copy.to(tensor.device)
