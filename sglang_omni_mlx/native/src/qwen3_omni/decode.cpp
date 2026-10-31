// SPDX-License-Identifier: Apache-2.0
#include "mlx/random.h"
#include "qwen3_omni/generation.h"
#include <limits>

namespace qwen3_omni {
mx::array sample(const mx::array &input, const std::vector<int> &history,
                 const Sampling &options) {
  auto logits = mx::astype(input, mx::float32);
  int vocabulary = logits.shape(-1);
  if (options.repetition_penalty != 1 && !history.empty()) {
    std::vector<float> penalties(vocabulary, 1);
    for (int token : history) {
      penalties.at(token) = options.repetition_penalty;
    }
    auto penalty = mx::array(penalties.data(), {1, vocabulary});
    logits = mx::where(logits < 0, logits * penalty, logits / penalty);
  } else {
  }
  if (options.temperature == 0 || options.top_k == 1) {
    return mx::argmax(logits, -1);
  } else {
  }
  logits = logits / options.temperature;
  auto negative_infinity = mx::array(-std::numeric_limits<float>::infinity());
  if (options.top_k > 0 && options.top_k < vocabulary) {
    auto selected = rows(mx::argpartition(-logits, options.top_k - 1, -1), 0,
                         options.top_k, -1);
    logits = mx::put_along_axis(mx::full(logits.shape(), negative_infinity),
                                selected,
                                mx::take_along_axis(logits, selected, -1), -1);
  } else {
  }
  if (options.top_p < 1) {
    auto sorted = mx::argsort(-logits, -1);
    auto ranked = mx::take_along_axis(logits, sorted, -1);
    auto probabilities = mx::softmax(ranked, -1);
    auto cumulative = mx::cumsum(probabilities, -1) - probabilities;
    auto filtered =
        mx::where(cumulative < options.top_p, ranked, negative_infinity);
    return mx::reshape(
        mx::take_along_axis(
            sorted, mx::expand_dims(mx::random::categorical(filtered), -1), -1),
        {1});
  } else {
    return mx::random::categorical(logits);
  }
}
TextGeneration generate_text_tokens(
    mx::array logits, int maximum, const std::set<int> &eos_tokens,
    const Sampling &options, const Cancellation &cancelled,
    const std::function<std::vector<mx::array>(const mx::array &)> &advance,
    const std::function<void(const std::vector<int> &)> &consume) {
  struct SynchronizeOnExit {
    ~SynchronizeOnExit() { mx::synchronize(); }
  } synchronize;
  TextGeneration result{{}, 0, false};
  for (int i = 0; i < maximum; ++i) {
    check_cancelled(cancelled);
    auto device_token = sample(logits, result.tokens, options);
    if (i + 1 < maximum) {
      mx::async_eval({device_token});
      auto next = advance(device_token);
      mx::async_eval(next);
      logits = next.at(0);
    } else {
    }
    int token = mx::astype(device_token, mx::int32).item<int>();
    check_cancelled(cancelled);
    ++result.completion_tokens;
    if (eos_tokens.contains(token)) {
      result.stopped = true;
      return result;
    } else {
      result.tokens.push_back(token);
      consume(result.tokens);
    }
  }
  return result;
}
} // namespace qwen3_omni
