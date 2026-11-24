#include "ming3_asr/ming3_asr_model.h"

#include <stdexcept>

namespace ming3_asr {

namespace mx = mlx::core;

GatedDeltaResult GatedDelta(const mx::array &queries, const mx::array &keys,
                            const mx::array &values, const mx::array &decay,
                            const mx::array &beta, const mx::array &state) {
  if (queries.ndim() != 4 || queries.shape(0) < 1 || queries.shape(1) < 1 ||
      queries.shape(2) < 1 || queries.shape(3) < 1 ||
      queries.shape(3) % 32 != 0) {
    throw std::invalid_argument(
        "Ming GatedDelta queries must have positive [batch, tokens, heads, "
        "head_width] dimensions with head_width divisible by 32");
  } else {
  }
  const auto expect = [](const mx::array &tensor, const mx::Shape &shape,
                         const std::string &name) {
    if (tensor.shape() != shape) {
      throw std::invalid_argument("Ming GatedDelta " + name +
                                  " shape: expected " +
                                  nlohmann::json(shape).dump() + ", got " +
                                  nlohmann::json(tensor.shape()).dump());
    } else {
    }
    if (tensor.dtype() != mx::float32) {
      throw std::invalid_argument("Ming GatedDelta " + name +
                                  " must have float32 dtype");
    } else {
    }
  };
  expect(queries, queries.shape(), "queries");
  expect(keys, queries.shape(), "keys");
  expect(decay, queries.shape(), "decay");
  expect(beta, {queries.shape(0), queries.shape(1), queries.shape(2)}, "beta");
  expect(
      state,
      {queries.shape(0), queries.shape(2), queries.shape(3), queries.shape(3)},
      "state");
  if (values.shape() != queries.shape() ||
      (values.dtype() != mx::float32 && values.dtype() != mx::bfloat16 &&
       values.dtype() != mx::float16)) {
    throw std::invalid_argument(
        "Ming GatedDelta values must match queries shape and have "
        "float32, bfloat16 or float16 dtype");
  } else {
  }
  static const auto kernel = mx::fast::metal_kernel(
      "ming_delta",
      {"queries", "keys", "values", "decay", "beta", "state", "length"},
      {"output", "next_state"}, R"(
    const uint batch_head = thread_position_in_grid.z;
    const uint batch = batch_head / heads;
    const uint head = batch_head % heads;
    const uint value_index = thread_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    constexpr uint per_lane = width / 32;
    const uint state_offset = (batch_head * width + value_index) * width;
    float memory[per_lane];
    for (uint part = 0; part < per_lane; ++part) {
      memory[part] = state[state_offset + lane * per_lane + part];
    }
    for (int token = 0; token < length; ++token) {
      const uint head_offset = (batch * length + token) * heads + head;
      const uint feature_offset = head_offset * width;
      float prediction = 0.0f;
      for (uint part = 0; part < per_lane; ++part) {
        const uint feature = feature_offset + lane * per_lane + part;
        memory[part] *= decay[feature];
        prediction += memory[part] * keys[feature];
      }
      prediction = simd_sum(prediction);
      const float correction =
          (float(values[feature_offset + value_index]) - prediction) *
          beta[head_offset];
      float result = 0.0f;
      for (uint part = 0; part < per_lane; ++part) {
        const uint feature = feature_offset + lane * per_lane + part;
        memory[part] += keys[feature] * correction;
        result += memory[part] * queries[feature];
      }
      result = simd_sum(result);
      if (lane == 0) {
        output[feature_offset + value_index] = result;
      }
    }
    for (uint part = 0; part < per_lane; ++part) {
      next_state[state_offset + lane * per_lane + part] = memory[part];
    }
  )");
  const auto result = kernel(
      {queries, keys, values, decay, beta, state, mx::array(queries.shape(1))},
      {values.shape(), state.shape()}, {mx::float32, mx::float32},
      {32, values.shape(3), values.shape(0) * values.shape(2)}, {32, 4, 1},
      {{"heads", values.shape(2)}, {"width", values.shape(3)}}, std::nullopt,
      false, {});
  return {result[0], result[1]};
}

} // namespace ming3_asr
