# SPDX-License-Identifier: Apache-2.0
"""Rank top-k/top-p candidates and draw samples using a Philox exponential race."""

from __future__ import annotations

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    libdevice = None
    HAS_TRITON = False

if HAS_TRITON:
    UNIFORM_MANTISSA_BITS = tl.constexpr(23)

    @triton.jit
    def ordered_keys(
        scores: tl.tensor,
        candidate_indices: tl.tensor,
        is_candidate: tl.tensor,
        INDEX_BITS: tl.constexpr,
    ) -> tl.tensor:
        score_bits = tl.where(scores == 0.0, 0.0, scores).to(tl.uint32, bitcast=True)
        sign_bit = tl.full(scores.shape, 1, tl.uint32) << 31
        all_bits = sign_bit | (sign_bit - 1)
        ordered_bits = score_bits ^ tl.where(
            (score_bits & sign_bit) != 0, all_bits, sign_bit
        )
        keys = (ordered_bits.to(tl.int64) << INDEX_BITS) | (
            (1 << INDEX_BITS) - 1 - candidate_indices
        ).to(tl.int64)
        return tl.where(is_candidate, keys, 0)

    @triton.jit
    def key_scores(keys: tl.tensor, INDEX_BITS: tl.constexpr) -> tl.tensor:
        ordered_bits = (keys >> INDEX_BITS).to(tl.uint32)
        sign_bit = tl.full(keys.shape, 1, tl.uint32) << 31
        all_bits = sign_bit | (sign_bit - 1)
        score_bits = ordered_bits ^ tl.where(
            (ordered_bits & sign_bit) != 0, sign_bit, all_bits
        )
        return score_bits.to(tl.float32, bitcast=True)

    @triton.jit
    def key_indices(keys: tl.tensor, INDEX_BITS: tl.constexpr) -> tl.tensor:
        return ((1 << INDEX_BITS) - 1 - (keys & ((1 << INDEX_BITS) - 1))).to(tl.int32)

    @triton.jit
    def exponential_noise(
        seed: tl.tensor,
        step: tl.tensor,
        row: tl.tensor,
        stream_id: tl.tensor,
        candidate_indices: tl.tensor,
    ) -> tl.tensor:
        random_bits, _, _, _ = tl.philox(
            seed ^ ((step >> 32) << 32),
            candidate_indices.to(tl.uint32),
            row,
            stream_id,
            (step & 0xFFFFFFFF).to(tl.uint32),
        )
        uniform = (
            (random_bits >> (32 - UNIFORM_MANTISSA_BITS)).to(tl.float32) + 0.5
        ) * (1.0 / (1 << UNIFORM_MANTISSA_BITS))
        return -libdevice.log(uniform)

    @triton.jit
    def candidate_selection(
        main_logits: tl.tensor,
        main_is_candidate: tl.tensor,
        tail_logits: tl.tensor,
        tail_is_candidate: tl.tensor,
        temperature: tl.tensor,
        INDEX_BITS: tl.constexpr,
        SELECTED: tl.constexpr,
        HAS_TAIL: tl.constexpr,
    ) -> tuple[tl.tensor, ...]:
        main_scores = tl.where(
            main_is_candidate,
            tl.math.div_rn(
                main_logits, tl.broadcast_to(temperature, main_logits.shape)
            ),
            float("-inf"),
        )
        tail_scores = tl.where(
            tail_is_candidate,
            tl.math.div_rn(
                tail_logits, tl.broadcast_to(temperature, tail_logits.shape)
            ),
            float("-inf"),
        )
        row_maximum = tl.maximum(
            tl.max(main_scores, axis=0), tl.max(tail_scores, axis=0)
        )
        main_weights = libdevice.exp(main_scores - row_maximum)
        tail_weights = libdevice.exp(tail_scores - row_maximum)
        weight_sum = tl.sum(main_weights, axis=0) + tl.sum(tail_weights, axis=0)
        main_width: tl.constexpr = main_logits.shape[0]
        main_keys = ordered_keys(
            main_scores, tl.arange(0, main_width), main_is_candidate, INDEX_BITS
        )
        tail_keys = ordered_keys(
            tail_scores,
            main_width + tl.arange(0, tail_logits.shape[0]),
            tail_is_candidate,
            INDEX_BITS,
        )
        top_keys = tl.topk(main_keys, SELECTED)
        if HAS_TAIL:
            top_keys = tl.topk(
                tl.reshape(
                    tl.join(top_keys, tl.topk(tail_keys, SELECTED)), [2 * SELECTED]
                ),
                SELECTED,
            )
        else:
            pass
        return (
            main_keys,
            main_weights,
            tail_keys,
            tail_weights,
            row_maximum,
            weight_sum,
            top_keys,
        )

    @triton.jit
    def selected_candidates(
        top_keys: tl.tensor,
        row_maximum: tl.tensor,
        weight_sum: tl.tensor,
        top_k: tl.tensor,
        top_p: tl.tensor,
        INDEX_BITS: tl.constexpr,
    ) -> tuple[tl.tensor, ...]:
        selected_count: tl.constexpr = top_keys.shape[0]
        ranks = tl.arange(0, selected_count)
        selected_indices = key_indices(top_keys, INDEX_BITS)
        selected_weights = libdevice.exp(key_scores(top_keys, INDEX_BITS) - row_maximum)
        cumulative_probabilities = tl.cumsum(
            tl.math.div_rn(
                selected_weights, tl.broadcast_to(weight_sum, selected_weights.shape)
            ),
            axis=0,
        )
        is_kept = (ranks == 0) | ((ranks < top_k) & (cumulative_probabilities <= top_p))
        last_cumulative_probability = tl.sum(
            tl.where(ranks == selected_count - 1, cumulative_probabilities, 0.0), axis=0
        )
        holds_kept_set = (top_k <= selected_count) | (
            last_cumulative_probability > top_p
        )
        return selected_indices, selected_weights, is_kept, holds_kept_set

    @triton.jit
    def search_threshold(
        main_keys: tl.tensor,
        main_probabilities: tl.tensor,
        tail_keys: tl.tensor,
        tail_probabilities: tl.tensor,
        top_k: tl.tensor,
        top_p: tl.tensor,
        KEY_BITS: tl.constexpr,
        SEARCH_ROUNDS: tl.constexpr,
    ) -> tl.tensor:
        lower_key = tl.full([], 0, tl.int64)
        upper_key = tl.full([], 1, tl.int64) << KEY_BITS
        for _ in range(SEARCH_ROUNDS):
            middle_key = (lower_key + upper_key) >> 1
            main_is_above = main_keys >= middle_key
            tail_is_above = tail_keys >= middle_key
            above_count = tl.sum(main_is_above.to(tl.int64), axis=0) + tl.sum(
                tail_is_above.to(tl.int64), axis=0
            )
            above_probability = tl.sum(
                tl.where(main_is_above, main_probabilities, 0.0), axis=0
            ) + tl.sum(tl.where(tail_is_above, tail_probabilities, 0.0), axis=0)
            fits = (above_count <= top_k) & (above_probability <= top_p)
            upper_key = tl.where(fits, middle_key, upper_key)
            lower_key = tl.where(fits, lower_key, middle_key + 1)
        return upper_key

    @triton.jit
    def race_selected(
        selected_indices: tl.tensor,
        selected_weights: tl.tensor,
        is_kept: tl.tensor,
        seed: tl.tensor,
        step: tl.tensor,
        row: tl.tensor,
        stream_id: tl.tensor,
    ) -> tl.tensor:
        race = tl.where(
            is_kept,
            selected_weights
            / exponential_noise(seed, step, row, stream_id, selected_indices),
            -1.0,
        )
        chosen_rank = tl.argmax(race, axis=0)
        ranks = tl.arange(0, selected_indices.shape[0])
        return tl.sum(tl.where(ranks == chosen_rank, selected_indices, 0), axis=0)

    @triton.jit
    def race_above_threshold(
        main_keys: tl.tensor,
        main_weights: tl.tensor,
        tail_keys: tl.tensor,
        tail_weights: tl.tensor,
        threshold_key: tl.tensor,
        top_key: tl.tensor,
        seed: tl.tensor,
        step: tl.tensor,
        row: tl.tensor,
        stream_id: tl.tensor,
        INDEX_BITS: tl.constexpr,
    ) -> tl.tensor:
        main_is_kept = (main_keys >= threshold_key) | (main_keys == top_key)
        tail_is_kept = (tail_keys >= threshold_key) | (tail_keys == top_key)
        main_race = tl.where(
            main_is_kept,
            main_weights
            / exponential_noise(
                seed, step, row, stream_id, key_indices(main_keys, INDEX_BITS)
            ),
            -1.0,
        )
        tail_race = tl.where(
            tail_is_kept,
            tail_weights
            / exponential_noise(
                seed, step, row, stream_id, key_indices(tail_keys, INDEX_BITS)
            ),
            -1.0,
        )
        best_race = tl.maximum(tl.max(main_race, axis=0), tl.max(tail_race, axis=0))
        chosen_key = tl.maximum(
            tl.max(tl.where(main_race == best_race, main_keys, 0), axis=0),
            tl.max(tl.where(tail_race == best_race, tail_keys, 0), axis=0),
        )
        return key_indices(tl.where(chosen_key > 0, chosen_key, top_key), INDEX_BITS)

    @triton.jit
    def draw_candidate(
        main_keys: tl.tensor,
        main_weights: tl.tensor,
        tail_keys: tl.tensor,
        tail_weights: tl.tensor,
        row_maximum: tl.tensor,
        weight_sum: tl.tensor,
        top_keys: tl.tensor,
        top_k: tl.tensor,
        top_p: tl.tensor,
        seed: tl.tensor,
        step: tl.tensor,
        row: tl.tensor,
        stream_id: tl.tensor,
        INDEX_BITS: tl.constexpr,
        KEY_BITS: tl.constexpr,
    ) -> tl.tensor:
        selected_indices, selected_weights, is_kept, holds_kept_set = (
            selected_candidates(
                top_keys, row_maximum, weight_sum, top_k, top_p, INDEX_BITS
            )
        )
        if holds_kept_set:
            chosen_index = race_selected(
                selected_indices, selected_weights, is_kept, seed, step, row, stream_id
            )
        else:
            threshold_key = search_threshold(
                main_keys,
                tl.math.div_rn(
                    main_weights, tl.broadcast_to(weight_sum, main_weights.shape)
                ),
                tail_keys,
                tl.math.div_rn(
                    tail_weights, tl.broadcast_to(weight_sum, tail_weights.shape)
                ),
                top_k,
                top_p,
                KEY_BITS,
                KEY_BITS,
            )
            chosen_index = race_above_threshold(
                main_keys,
                main_weights,
                tail_keys,
                tail_weights,
                threshold_key,
                tl.max(top_keys, axis=0),
                seed,
                step,
                row,
                stream_id,
                INDEX_BITS,
            )
        return chosen_index

else:
    pass
