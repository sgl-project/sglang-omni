#include "CZhuyin.h"
#include "Engine/ParselessLM.h"
#include "Engine/gramambular2/reading_grid.h"
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

using Grid = Formosa::Gramambular2::ReadingGrid;
struct VoxtZhuyin {
  std::shared_ptr<McBopomofo::ParselessLM> lm;
  std::vector<std::string> results;
};

extern "C" VoxtZhuyin *voxt_zhuyin_create(const char *path) {
  if (!path)
    return nullptr;
  try {
    auto engine = std::make_unique<VoxtZhuyin>();
    engine->lm = std::make_shared<McBopomofo::ParselessLM>();
    if (!engine->lm->open(path) || !engine->lm->hasUnigrams("ㄋㄧˇ"))
      return nullptr;
    return engine.release();
  } catch (...) {
    return nullptr;
  }
}

extern "C" void voxt_zhuyin_destroy(VoxtZhuyin *engine) { delete engine; }

static std::string sentence(Grid &grid) {
  std::string text;
  for (const auto &node : grid.walk().nodes) {
    const auto value = node->value();
    if (value.rfind("MACRO@", 0) == 0)
      return {};
    text += value;
  }
  return text;
}

extern "C" int voxt_zhuyin_convert(VoxtZhuyin *engine, const char *input,
                                   size_t cursor) {
  if (!engine || !input)
    return -2;
  engine->results.clear();
  try {
    std::string source(input);
    std::vector<std::string> readings;
    for (size_t start = 0; start < source.size();) {
      const auto end = source.find('-', start);
      readings.push_back(
          source.substr(start, end == std::string::npos ? end : end - start));
      if (end == std::string::npos)
        break;
      start = end + 1;
    }
    if (readings.empty() || readings.size() > 32 || cursor >= readings.size())
      return -1;
    Grid grid(engine->lm);
    for (const auto &reading : readings)
      if (!grid.insertReading(reading))
        return -1;
    std::unordered_set<std::string> seen;
    auto append = [&](std::string text) {
      if (!text.empty() && seen.insert(text).second)
        engine->results.push_back(std::move(text));
    };
    append(sentence(grid));
    const auto candidates = grid.candidatesAt(cursor);
    size_t considered = 0;
    for (const auto &candidate : candidates) {
      if (engine->results.size() >= 10 || ++considered > 64)
        break;
      // Never reuse an overridden grid: each result is independent.
      Grid alternative(engine->lm);
      for (const auto &reading : readings)
        alternative.insertReading(reading);
      if (alternative.overrideCandidate(cursor, candidate))
        append(sentence(alternative));
    }
    return static_cast<int>(engine->results.size());
  } catch (...) {
    engine->results.clear();
    return -2;
  }
}

extern "C" const char *voxt_zhuyin_candidate(const VoxtZhuyin *engine,
                                             size_t index) {
  if (!engine || index >= engine->results.size())
    return nullptr;
  return engine->results[index].c_str();
}
