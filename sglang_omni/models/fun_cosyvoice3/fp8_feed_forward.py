"""FP8 feed forward for the Fun-CosyVoice3 DiT on SM120.

out = W4 x fp8(gelu(fp8(norm) @ W3^T + b3)) + b4. The fused LayerNorm stores FP8 for
GEMM 3, and the GEMM 3 epilogue stores FP8 for GEMM 4, so no separate quantize runs.
"""

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm120_utils
import torch
from cutlass.cute.runtime import from_dlpack, make_fake_stream
from sglang.kernels.kda_kernels.layernorm_modulate_triton import (
    fused_layernorm_modulate_fp8_quant_raw,
)
from sglang.srt.utils.custom_op import register_custom_op

CTA_TILER = (64, 128, 64)
NUM_STAGES = 2
# UE8M0 scale byte, 0x7F is exactly 1.0. Static scales run in the epilogue.
UNIT_SCALE_BYTE = 0x7F
GELU_CUBIC = 0.044715
GELU_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


@cute.jit
def copy_A_tile(
    tiled_copy_A: cute.TiledCopy,
    tAgA_tile: cute.Tensor,
    tAsA_stage: cute.Tensor,
    is_full_block: cutlass.Boolean,
    block_first_row: cutlass.Int32,
    thread_row: cutlass.Int32,
    thread_rows: cutlass.Constexpr,
    num_rows: cutlass.Int32,
):
    if is_full_block:
        cute.copy(tiled_copy_A, tAgA_tile, tAsA_stage)
    else:
        # Rows past M stay stale in SMEM, and their output is never stored
        for copy_row in cutlass.range_constexpr(cute.size(tAsA_stage, mode=[1])):
            row = block_first_row + thread_row + copy_row * thread_rows
            if row < num_rows:
                cute.copy(
                    tiled_copy_A,
                    tAgA_tile[None, copy_row, None],
                    tAsA_stage[None, copy_row, None],
                )
            else:
                pass


def make_unit_scale_A(thr_mma, tidx, tile_mk):
    reference = cute.make_rmem_tensor(
        cute.make_layout(tile_mk, stride=(0, 0)),
        cutlass.Float8E8M0FNU,
    )
    fragment = sm120_utils.partition_fragment_SFA(reference, thr_mma, tidx)
    cute.recast_tensor(fragment, cutlass.Uint8).fill(UNIT_SCALE_BYTE)
    return fragment


def make_unit_scale_B(thr_mma, tidx, tile_nk):
    reference = cute.make_rmem_tensor(
        cute.make_layout(tile_nk, stride=(0, 0)),
        cutlass.Float8E8M0FNU,
    )
    fragment = sm120_utils.partition_fragment_SFB(reference, thr_mma, tidx)
    cute.recast_tensor(fragment, cutlass.Uint8).fill(UNIT_SCALE_BYTE)
    return fragment


@cute.kernel
def gemm3_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    mBias: cute.Tensor,
    inverse_scale: cutlass.Float32,
    weight_scale: cutlass.Float32,
    sA_layout: cute.Layout,
    sB_layout: cute.Layout,
    sA_swizzle: cute.Swizzle,
    sB_swizzle: cute.Swizzle,
    tiled_copy_A: cute.TiledCopy,
    tiled_copy_B: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
    cta_tiler: cutlass.Constexpr = CTA_TILER,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()

    gA = cute.local_tile(
        mA[None, None, bidz], cta_tiler, (bidx, bidy, None), proj=(1, None, 1)
    )
    gB = cute.local_tile(
        mB[None, None, bidz], cta_tiler, (bidx, bidy, None), proj=(None, 1, 1)
    )
    gC = cute.local_tile(
        mC[None, None, bidz], cta_tiler, (bidx, bidy, None), proj=(1, 1, None)
    )

    # bias depends only on the column, so bidy alone picks its tile
    column_tiler = (cta_tiler[1],)
    gBias = cute.local_tile(mBias, column_tiler, (bidy,))

    # M stride 0: every row reads the same column vector
    row_broadcast_layout = cute.make_layout(
        (cta_tiler[0], cta_tiler[1]),
        stride=(0, 1),
    )
    gBias_mn = cute.make_tensor(gBias.iterator, row_broadcast_layout)

    @cute.struct
    class SharedStorageAB:
        a: cute.struct.Align[
            cute.struct.MemRange[mA.element_type, cute.cosize(sA_layout)], 16
        ]
        b: cute.struct.Align[
            cute.struct.MemRange[mB.element_type, cute.cosize(sB_layout)], 16
        ]

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(
        SharedStorageAB.size_in_bytes(),
        byte_alignment=16,
    )
    sA = SharedStorageAB(storage).a.get_tensor(sA_layout, swizzle=sA_swizzle)
    sB = SharedStorageAB(storage).b.get_tensor(sB_layout, swizzle=sB_swizzle)

    thr_copy_A = tiled_copy_A.get_slice(tidx)
    thr_copy_B = tiled_copy_B.get_slice(tidx)

    tAgA = thr_copy_A.partition_S(gA)
    tBgB = thr_copy_B.partition_S(gB)

    tAsA = thr_copy_A.partition_D(sA)
    tBsB = thr_copy_B.partition_D(sB)

    k_tile_count = cute.size(tAgA, mode=[3])

    thr_mma = tiled_mma.get_slice(tidx)

    tCsA = thr_mma.partition_A(sA)
    tCsB = thr_mma.partition_B(sB)
    tCgC = thr_mma.partition_C(gC)

    tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
    tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
    tCrC = tiled_mma.make_fragment_C(tCgC)

    tCgBias = thr_mma.partition_C(gBias_mn)

    tCrC.fill(0.0)

    tCrSFA = make_unit_scale_A(thr_mma, tidx, (cta_tiler[0], cta_tiler[2]))
    tCrSFB = make_unit_scale_B(thr_mma, tidx, (cta_tiler[1], cta_tiler[2]))

    atom_copy_s2r_A = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x16x8bOp(transpose=False, num_matrices=4),
        mA.element_type,
    )

    atom_copy_s2r_B = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x16x8bOp(transpose=False, num_matrices=4),
        mB.element_type,
    )

    tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_copy_s2r_A, tiled_mma)
    tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_copy_s2r_B, tiled_mma)

    thr_copy_ldmatrix_A = tiled_copy_s2r_A.get_slice(tidx)
    thr_copy_ldmatrix_B = tiled_copy_s2r_B.get_slice(tidx)

    tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
    tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
    tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
    tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)

    num_k_block = cute.size(tCrA, mode=[2])

    # A rows owned by this thread are thread_row + copy_row * thread_rows
    thread_rows = cta_tiler[0] // cute.size(tAsA, mode=[1])
    thread_row = tidx // (cute.size(tiled_mma) // thread_rows)
    block_first_row = bidx * cta_tiler[0]
    is_full_block = block_first_row + cta_tiler[0] <= mA.shape[0]

    # Copy is conditional, commit is not, so the wait counter stays right at the tail.
    copy_A_tile(
        tiled_copy_A,
        tAgA[None, None, None, 0],
        tAsA[None, None, None, 0],
        is_full_block,
        block_first_row,
        thread_row,
        thread_rows,
        mA.shape[0],
    )
    cute.copy(tiled_copy_B, tBgB[None, None, None, 0], tBsB[None, None, None, 0])
    cute.arch.cp_async_commit_group()

    if k_tile_count > 1:
        copy_A_tile(
            tiled_copy_A,
            tAgA[None, None, None, 1],
            tAsA[None, None, None, 1],
            is_full_block,
            block_first_row,
            thread_row,
            thread_rows,
            mA.shape[0],
        )
        cute.copy(tiled_copy_B, tBgB[None, None, None, 1], tBsB[None, None, None, 1])
    else:
        pass
    cute.arch.cp_async_commit_group()

    cute.arch.cp_async_wait_group(1)
    cute.arch.sync_threads()

    for k_tile in range(k_tile_count):
        slot = k_tile % NUM_STAGES
        tCsA_p = tCsA_copy_view[None, None, None, slot]
        tCsB_p = tCsB_copy_view[None, None, None, slot]

        for k_block in cutlass.range(num_k_block, unroll_full=True):
            cute.copy(
                tiled_copy_s2r_A,
                tCsA_p[None, None, k_block],
                tCrA_copy_view[None, None, k_block],
            )
            cute.copy(
                tiled_copy_s2r_B,
                tCsB_p[None, None, k_block],
                tCrB_copy_view[None, None, k_block],
            )
            cute.gemm(
                tiled_mma,
                tCrC,
                [
                    tCrA[None, None, k_block],
                    tCrSFA[None, None, k_block],
                ],
                [
                    tCrB[None, None, k_block],
                    tCrSFB[None, (None, k_block)],
                ],
                tCrC,
            )

        cute.arch.sync_threads()

        if k_tile + 2 < k_tile_count:
            copy_A_tile(
                tiled_copy_A,
                tAgA[None, None, None, k_tile + 2],
                tAsA[None, None, None, slot],
                is_full_block,
                block_first_row,
                thread_row,
                thread_rows,
                mA.shape[0],
            )
            cute.copy(
                tiled_copy_B,
                tBgB[None, None, None, k_tile + 2],
                tBsB[None, None, None, slot],
            )
        else:
            pass
        cute.arch.cp_async_commit_group()

        cute.arch.cp_async_wait_group(1)
        cute.arch.sync_threads()

    tCrBias = cute.make_fragment_like(tCrC, mBias.element_type)
    cute.autovec_copy(tCgBias, tCrBias)

    # note (emre): the static scale keeps GELU below FP8_MAX for real inputs.
    hidden = tCrC.load() * weight_scale + tCrBias.load().to(cutlass.Float32)
    cubic = hidden + GELU_CUBIC * hidden * hidden * hidden
    gelu = 0.5 * hidden * (1.0 + cute.math.tanh(GELU_SQRT_2_OVER_PI * cubic))
    scaled = gelu * inverse_scale
    scaled = cute.where(scaled > FP8_MAX, FP8_MAX, scaled)

    tCrD = cute.make_fragment_like(tCrC, mC.element_type)
    tCrD[None] = scaled.to(mC.element_type)
    tCgCoord = thr_mma.partition_C(
        cute.make_identity_tensor((cta_tiler[0], cta_tiler[1]))
    )
    if is_full_block:
        cute.autovec_copy(tCrD, tCgC)
    else:
        for index in cutlass.range_constexpr(cute.size(tCrD)):
            row = block_first_row + tCgCoord[index][0]
            if row < mA.shape[0]:
                tCgC[index] = tCrD[index]
            else:
                pass


ATOM_LAYOUT_MNK = (2, 2, 1)
NUM_THREADS = 2 * 2 * 1 * 32  # atom layout carpimi x 32
MMA_SHAPE = (16, 8, 32)


def make_smem_layout_ab(dtype, copy_bits, smem_tiler):
    major_size = smem_tiler[1]
    row_bytes = major_size * dtype.width // 8
    chunk_bytes = copy_bits // 8
    swizzle_bits = min(int(math.log2(row_bytes // chunk_bytes)), 3)
    base_bits = int(math.log2(chunk_bytes))
    shift_bits = int(math.log2(128 // chunk_bytes))
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
def gemm3_host(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    mBias: cute.Tensor,
    inverse_scale: cutlass.Float32,
    weight_scale: cutlass.Float32,
    stream: cuda.CUstream,
):
    bM, bN, bK = CTA_TILER

    # torch (l,m,k)/(l,n,k)/(l,m,n) -> CuTe (m,k,l)/(n,k,l)/(m,n,l)
    mA = cute.make_tensor(mA.iterator, cute.select(mA.layout, mode=[1, 2, 0]))
    mB = cute.make_tensor(mB.iterator, cute.select(mB.layout, mode=[1, 2, 0]))
    mC = cute.make_tensor(mC.iterator, cute.select(mC.layout, mode=[1, 2, 0]))

    op = cute.nvgpu.warp.MmaMXF8Op(
        mA.element_type, cutlass.Float32, cutlass.Float8E8M0FNU
    )
    permutation_mnk = (
        ATOM_LAYOUT_MNK[0] * MMA_SHAPE[0],
        ATOM_LAYOUT_MNK[1] * MMA_SHAPE[1],
        ATOM_LAYOUT_MNK[2] * MMA_SHAPE[2],
    )
    tiled_mma = cute.make_tiled_mma(
        op, cute.make_layout(ATOM_LAYOUT_MNK), permutation_mnk=permutation_mnk
    )

    ab_copy_bits = 128
    sA_layout, sA_swizzle = make_smem_layout_ab(
        mA.element_type, ab_copy_bits, (bM, bK, NUM_STAGES)
    )
    sB_layout, sB_swizzle = make_smem_layout_ab(
        mB.element_type, ab_copy_bits, (bN, bK, NUM_STAGES)
    )

    atom_g2s = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL),
        mA.element_type,
        num_bits_per_copy=ab_copy_bits,
    )
    tiled_copy_A = make_gmem_tiled_copy(
        atom_g2s, mA.element_type, ab_copy_bits, bK, NUM_THREADS
    )
    tiled_copy_B = make_gmem_tiled_copy(
        atom_g2s, mB.element_type, ab_copy_bits, bK, NUM_THREADS
    )

    grid = cute.ceil_div(mC.shape, (bM, bN, 1))
    gemm3_kernel(
        mA,
        mB,
        mC,
        mBias,
        inverse_scale,
        weight_scale,
        sA_layout,
        sB_layout,
        sA_swizzle,
        sB_swizzle,
        tiled_copy_A,
        tiled_copy_B,
        tiled_mma,
    ).launch(
        grid=grid,
        block=[NUM_THREADS, 1, 1],
        stream=stream,
    )


GELU_SCALE = 1.0 / 8.0
LAYER_NORM_EPS = 1e-6
FP8_MAX = 448.0
# note (emre): measured crossover against the compiled BF16 path, M 1000 0.90x, M 1300 1.08x.
FP8_MIN_ROWS = 1200
compiled_kernels: dict[tuple[int, int], object] = {}


def wrap_rows(tensor: torch.Tensor) -> cute.Tensor:
    # Only the row count is dynamic, so one kernel serves every packed length
    return from_dlpack(tensor, assumed_align=16).mark_compact_shape_dynamic(
        mode=1, stride_order=(0, 1, 2), divisibility=1
    )


def fake_fp8_feed_forward(
    h: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    norm_scale: torch.Tensor,
    in_weight_fp8: torch.Tensor,
    in_bias: torch.Tensor,
    in_weight_scale: float,
    out_weight_fp8: torch.Tensor,
    out_bias: torch.Tensor,
    out_weight_scale: torch.Tensor,
    gelu_scale: torch.Tensor,
) -> torch.Tensor:
    return h.new_empty(h.shape, dtype=torch.bfloat16)


@register_custom_op(
    op_name="fun_cosyvoice3_fp8_feed_forward",
    mutates_args=[],
    fake_impl=fake_fp8_feed_forward,
)
def fp8_feed_forward(
    h: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    norm_scale: torch.Tensor,
    in_weight_fp8: torch.Tensor,
    in_bias: torch.Tensor,
    in_weight_scale: float,
    out_weight_fp8: torch.Tensor,
    out_bias: torch.Tensor,
    out_weight_scale: torch.Tensor,
    gelu_scale: torch.Tensor,
) -> torch.Tensor:
    """h: (1, M, K) BF16 before ff_norm, scale, shift: (1, K) BF16, in_weight_fp8:
    (1, N, K) FP8, out_weight_fp8: (K, N) FP8."""
    assert h.is_contiguous()
    norm_fp8 = fused_layernorm_modulate_fp8_quant_raw(
        h, scale, shift, norm_scale, LAYER_NORM_EPS
    )
    hidden = torch.empty(
        (1, h.shape[1], in_weight_fp8.shape[1]),
        device=h.device,
        dtype=torch.float8_e4m3fn,
    )
    tensors = (
        wrap_rows(norm_fp8),
        from_dlpack(in_weight_fp8.detach(), assumed_align=16),
        wrap_rows(hidden),
        from_dlpack(in_bias.detach(), assumed_align=16),
        1.0 / GELU_SCALE,
        in_weight_scale,
    )
    key = (in_weight_fp8.shape[1], in_weight_fp8.shape[2])
    if key not in compiled_kernels:
        compiled_kernels[key] = cute.compile(gemm3_host, *tensors, make_fake_stream())
    else:
        pass
    stream = cuda.CUstream(torch.cuda.current_stream(h.device).cuda_stream)
    compiled_kernels[key](*tensors, stream)
    with torch.autocast(device_type="cuda", enabled=False):
        out = torch._scaled_mm(  # noqa: leading-underscore
            hidden[0],
            out_weight_fp8,
            gelu_scale,
            out_weight_scale,
            out_dtype=torch.bfloat16,
        )
    return (out + out_bias).unsqueeze(0)
