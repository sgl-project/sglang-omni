#include "CPinyin.h"
#include "Engine/MemoryMappedFile.h"
#include "Engine/ParselessPhraseDB.h"
#include "Engine/gramambular2/reading_grid.h"
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using Grid = Formosa::Gramambular2::ReadingGrid;
using Unigram = Formosa::Gramambular2::LanguageModel::Unigram;

namespace {

// A word reached through the other z/zh, c/ch, s/sh spelling, per syllable.
constexpr double kFuzzyPenalty = -0.3;
// A lone initial ("n" for one syllable) only bridges the grid; words win over
// it.
constexpr double kLoneInitialPenalty = -2.0;
// Words kept for one abbreviated reading, best first.
constexpr size_t kAbbreviatedLimit = 24;
// Longer abbreviated readings are not looked up (the data has almost no 5+
// syllable words).
constexpr size_t kAbbreviatedMaxSyllables = 4;

const std::unordered_set<std::string> kInitials = {
    "b", "p", "m",  "f",  "d",  "t", "n", "l", "g", "k", "h", "j",
    "q", "x", "zh", "ch", "sh", "r", "z", "c", "s", "y", "w"};

std::vector<std::string> Split(const std::string &text, char separator) {
  std::vector<std::string> parts;
  for (size_t start = 0; start <= text.size();) {
    const auto end = text.find(separator, start);
    parts.push_back(text.substr(
        start, end == std::string::npos ? std::string::npos : end - start));
    if (end == std::string::npos)
      break;
    start = end + 1;
  }
  return parts;
}

std::string InitialOf(const std::string &syllable) {
  for (const char *twin : {"zh", "ch", "sh"})
    if (syllable.rfind(twin, 0) == 0)
      return twin;
  std::string first = syllable.substr(0, 1);
  return kInitials.count(first) ? first : "";
}

std::string Flat(const std::string &initial) {
  return initial.size() == 2 ? initial.substr(0, 1) : initial;
}

// The other spelling of a z/zh, c/ch, s/sh syllable; others unchanged.
std::string Flipped(const std::string &syllable) {
  const auto initial = InitialOf(syllable);
  if (initial == "zh" || initial == "ch" || initial == "sh")
    return initial.substr(0, 1) + syllable.substr(2);
  if (initial == "z" || initial == "c" || initial == "s")
    return initial + "h" + syllable.substr(1);
  return syllable;
}

// Same rule as PinyinSegmenter.matches.
bool Matches(const std::string &typed, const std::string &syllable,
             bool fuzzy) {
  if (kInitials.count(typed)) {
    const auto initial = InitialOf(syllable);
    return fuzzy ? Flat(initial) == Flat(typed) : initial == typed;
  }
  return typed == syllable || (fuzzy && Flipped(typed) == syllable);
}

// "key value score" rows of the sorted McBopomofo format.
struct Row {
  std::string key;
  std::string value;
  double score = 0;
};

Row Parse(std::string_view line) {
  Row row;
  const auto first = line.find(' ');
  row.key = std::string(line.substr(0, first));
  if (first == std::string_view::npos)
    return row;
  const auto second = line.find(' ', first + 1);
  row.value = std::string(line.substr(
      first + 1, second == std::string_view::npos ? std::string_view::npos
                                                  : second - first - 1));
  if (second != std::string_view::npos) {
    try {
      row.score = std::stod(std::string(line.substr(second + 1)));
    } catch (...) {
    }
  }
  return row;
}

// Readings are syllables or initials joined by '-'. Every unigram's raw value
// is the dictionary reading it came from ("ni-hao" for 你好 typed as n-h).
class PinyinLM : public Formosa::Gramambular2::LanguageModel {
public:
  bool fuzzy = false;

  bool open(const char *path) {
    if (!file_.open(path))
      return false;
    db_ = std::make_unique<McBopomofo::ParselessPhraseDB>(
        file_.data(), file_.length(), /*validate_pragma=*/true);
    return true;
  }

  // Every syllable the segmenter knows; abbreviated lookups expand initials
  // through it.
  void setSyllables(std::vector<std::string> syllables) {
    syllables_ = std::move(syllables);
    cache_.clear();
  }

  void setFuzzy(bool value) {
    if (value != fuzzy)
      cache_.clear();
    fuzzy = value;
  }

  std::vector<Unigram> getUnigrams(const std::string &reading) override {
    if (const auto found = cache_.find(reading); found != cache_.end())
      return found->second;
    if (cache_.size() > 1024)
      cache_.clear();
    return cache_[reading] = lookup(reading);
  }

  bool hasUnigrams(const std::string &reading) override {
    return !getUnigrams(reading).empty();
  }

private:
  McBopomofo::MemoryMappedFile file_;
  std::unique_ptr<McBopomofo::ParselessPhraseDB> db_;
  std::unordered_map<std::string, std::vector<Unigram>> cache_;
  std::vector<std::string> syllables_;

  std::vector<Unigram> lookup(const std::string &reading) {
    if (!db_ || reading.empty())
      return {};
    const auto typed = Split(reading, '-');
    const bool abbreviated =
        std::any_of(typed.begin(), typed.end(),
                    [](const auto &t) { return kInitials.count(t) > 0; });
    std::vector<Row> rows =
        abbreviated ? abbreviatedRows(typed) : exactRows(typed);
    std::stable_sort(rows.begin(), rows.end(), [](const Row &a, const Row &b) {
      return a.score > b.score;
    });
    std::unordered_set<std::string> seen;
    std::vector<Unigram> result;
    for (auto &row : rows) {
      if (row.value.empty() || !seen.insert(row.value).second)
        continue;
      result.emplace_back(std::move(row.value), row.score, std::move(row.key));
      if (abbreviated && result.size() >= kAbbreviatedLimit)
        break;
    }
    return result;
  }

  // Full syllables: the reading itself, and with fuzzy every flipped spelling.
  std::vector<Row> exactRows(const std::vector<std::string> &typed) {
    std::vector<std::pair<std::string, int>> keys = {{"", 0}};
    for (const auto &syllable : typed) {
      std::vector<std::pair<std::string, int>> next;
      for (const auto &[key, flips] : keys) {
        const auto joined = key.empty() ? key : key + "-";
        next.emplace_back(joined + syllable, flips);
        if (fuzzy && Flipped(syllable) != syllable && next.size() < 64)
          next.emplace_back(joined + Flipped(syllable), flips + 1);
      }
      keys = std::move(next);
    }
    std::vector<Row> rows;
    for (const auto &[key, flips] : keys) {
      for (const auto &line : db_->findRows(key + " ")) {
        auto row = Parse(line);
        row.score += flips * kFuzzyPenalty;
        rows.push_back(std::move(row));
      }
    }
    return rows;
  }

  // Some pieces are initials. Every first syllable the first piece can stand
  // for, followed by how the second piece starts, narrows the sorted rows to
  // scan: n-h scans "na-h", "ni-h", … instead of every row starting with n.
  std::vector<Row> abbreviatedRows(const std::vector<std::string> &typed) {
    if (typed.size() > kAbbreviatedMaxSyllables)
      return {};
    std::vector<std::string> firsts;
    for (const auto &syllable : syllables_)
      if (Matches(typed[0], syllable, fuzzy))
        firsts.push_back(syllable);
    std::vector<std::string> seconds; // after "first-"; empty for one piece
    if (typed.size() > 1) {
      const auto &second = typed[1];
      const auto end = typed.size() > 2 ? "-" : " ";
      if (kInitials.count(second))
        seconds.push_back(fuzzy ? Flat(second) : second);
      else {
        seconds.push_back(second + end);
        if (fuzzy && Flipped(second) != second)
          seconds.push_back(Flipped(second) + end);
      }
    }
    std::vector<std::string> prefixes;
    for (const auto &first : firsts) {
      if (seconds.empty())
        prefixes.push_back(first + " ");
      for (const auto &second : seconds)
        prefixes.push_back(first + "-" + second);
    }
    std::vector<Row> rows;
    for (const auto &prefix : prefixes) {
      for (const auto &line : db_->findRows(prefix)) {
        const auto space = line.find(' ');
        if (space == std::string_view::npos)
          continue;
        const auto key = line.substr(0, space);
        if (static_cast<size_t>(std::count(key.begin(), key.end(), '-')) + 1 !=
            typed.size())
          continue;
        const auto syllables = Split(std::string(key), '-');
        bool all = true;
        for (size_t i = 0; i < typed.size() && all; ++i)
          all = Matches(typed[i], syllables[i], fuzzy);
        if (!all)
          continue;
        auto row = Parse(line);
        if (typed.size() == 1)
          row.score += kLoneInitialPenalty;
        rows.push_back(std::move(row));
      }
    }
    return rows;
  }
};

} // namespace

struct VoxtPinyin {
  std::shared_ptr<PinyinLM> lm;
  // Text, syllables replaced, dictionary reading.
  struct Result {
    std::string text;
    size_t length;
    std::string reading;
  };
  std::vector<Result> results;
  double score = 0;
};

extern "C" VoxtPinyin *voxt_pinyin_create(const char *path) {
  if (!path)
    return nullptr;
  try {
    auto engine = std::make_unique<VoxtPinyin>();
    engine->lm = std::make_shared<PinyinLM>();
    // A pinyin file must know "ni"; Zhuyin data or an empty file does not.
    if (!engine->lm->open(path) || !engine->lm->hasUnigrams("ni"))
      return nullptr;
    return engine.release();
  } catch (...) {
    return nullptr;
  }
}

extern "C" void voxt_pinyin_destroy(VoxtPinyin *engine) { delete engine; }

extern "C" void voxt_pinyin_set_syllables(VoxtPinyin *engine,
                                          const char *syllables) {
  if (!engine || !syllables)
    return;
  std::vector<std::string> list;
  for (auto &syllable : Split(syllables, ' '))
    if (!syllable.empty())
      list.push_back(std::move(syllable));
  engine->lm->setSyllables(std::move(list));
}

extern "C" void voxt_pinyin_set_fuzzy(VoxtPinyin *engine, int fuzzy) {
  if (engine)
    engine->lm->setFuzzy(fuzzy != 0);
}

extern "C" int voxt_pinyin_convert(VoxtPinyin *engine, const char *input) {
  if (!engine || !input)
    return -2;
  engine->results.clear();
  engine->score = 0;
  try {
    const auto syllables = Split(input, '-');
    if (syllables.empty() || syllables.size() > 32)
      return -1;
    Grid grid(engine->lm);
    for (const auto &syllable : syllables)
      if (!grid.insertReading(syllable))
        return -1;
    std::unordered_set<std::string> seen;
    auto append = [&](std::string text, size_t length, std::string reading) {
      if (!text.empty() && seen.insert(text).second)
        engine->results.push_back(
            {std::move(text), length, std::move(reading)});
    };
    // The best sentence for everything typed, then words that start at the
    // first syllable, longest first: picking one commits it and the rest
    // is converted again.
    std::string sentence;
    std::string reading;
    for (const auto &node : grid.walk().nodes) {
      sentence += node->value();
      engine->score += node->score();
      reading +=
          (reading.empty() ? "" : "-") + node->currentUnigram().rawValue();
    }
    append(sentence, syllables.size(), reading);
    for (const auto &candidate : grid.candidatesAt(0)) {
      if (engine->results.size() >= 64)
        break;
      size_t length = 1;
      for (char c : candidate.reading)
        length += c == '-';
      append(candidate.value, length, candidate.rawValue);
    }
    return static_cast<int>(engine->results.size());
  } catch (...) {
    engine->results.clear();
    return -2;
  }
}

extern "C" const char *voxt_pinyin_candidate(const VoxtPinyin *engine,
                                             size_t index) {
  if (!engine || index >= engine->results.size())
    return nullptr;
  return engine->results[index].text.c_str();
}

extern "C" size_t voxt_pinyin_candidate_length(const VoxtPinyin *engine,
                                               size_t index) {
  if (!engine || index >= engine->results.size())
    return 0;
  return engine->results[index].length;
}

extern "C" const char *voxt_pinyin_candidate_reading(const VoxtPinyin *engine,
                                                     size_t index) {
  if (!engine || index >= engine->results.size())
    return nullptr;
  return engine->results[index].reading.c_str();
}

extern "C" double voxt_pinyin_score(const VoxtPinyin *engine) {
  return engine ? engine->score : 0;
}
