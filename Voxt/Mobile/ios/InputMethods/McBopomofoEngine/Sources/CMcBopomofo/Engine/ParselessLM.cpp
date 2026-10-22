// Copyright (c) 2022 and onwards The McBopomofo Authors.
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use,
// copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.

#include "ParselessLM.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace McBopomofo {

bool ParselessLM::isLoaded() const { return db_ != nullptr; }

bool ParselessLM::open(const char* path) {
  if (!mmapedFile_.open(path)) {
    return false;
  }
  db_ = std::unique_ptr<ParselessPhraseDB>(new ParselessPhraseDB(
      mmapedFile_.data(), mmapedFile_.length(), /*validate_pragma=*/true));
  return true;
}

void ParselessLM::close() {
  mmapedFile_.close();
  db_ = nullptr;
}

bool ParselessLM::open(std::unique_ptr<ParselessPhraseDB> db) {
  if (db_ != nullptr) {
    return false;
  }

  db_ = std::move(db);
  return true;
}

std::vector<Formosa::Gramambular2::LanguageModel::Unigram>
ParselessLM::getUnigrams(const std::string& key) {
  if (db_ == nullptr) {
    return {};
  }

  std::vector<Formosa::Gramambular2::LanguageModel::Unigram> results;
  for (const auto& row : db_->findRows(key + " ")) {
    std::string value;
    double score = 0;

    // Move ahead until we encounter the first space. This is the key.
    const auto* it = row.begin();
    while (it != row.end() && *it != ' ') {
      ++it;
    }

    // The key is std::string(row.begin(), it), which we don't need.

    // Read past the space.
    if (it != row.end()) {
      ++it;
    }

    if (it != row.end()) {
      // Now it is the start of the value portion.
      const auto* value_begin = it;

      // Move ahead until we encounter the second space. This is the
      // value.
      while (it != row.end() && *it != ' ') {
        ++it;
      }
      value = std::string(value_begin, it);
    }

    // Read past the space. The remainder, if it exists, is the score.
    if (it != row.end()) {
      ++it;
    }

    if (it != row.end()) {
      score = std::stod(std::string(it, row.end()));
    }
    results.emplace_back(std::move(value), score);
  }
  return results;
}

bool ParselessLM::hasUnigrams(const std::string& key) {
  if (db_ == nullptr) {
    return false;
  }

  return db_->findFirstMatchingLine(key + " ") != nullptr;
}

std::vector<ParselessLM::FoundReading> ParselessLM::getReadings(
    const std::string& value) const {
  if (db_ == nullptr) {
    return {};
  }

  std::vector<ParselessLM::FoundReading> results;

  // We append a space so that we only find rows with the exact value. We
  // are taking advantage of the fact that a well-form row in this LM must
  // be in the format of "key value score".
  std::string actualValue = value + " ";

  for (const auto& row : db_->reverseFindRows(actualValue)) {
    std::string key;
    double score = 0;

    // Move ahead until we encounter the first space. This is the key.
    auto it = row.begin();
    while (it != row.end() && *it != ' ') {
      ++it;
    }

    key = std::string(row.begin(), it);

    // Read past the space.
    if (it != row.end()) {
      ++it;
    }

    if (it != row.end()) {
      // Now it is the start of the value portion, but we move ahead
      // until we encounter the second space to skip this part.
      while (it != row.end() && *it != ' ') {
        ++it;
      }
    }

    // Read past the space. The remainder, if it exists, is the score.
    if (it != row.end()) {
      ++it;
    }

    if (it != row.end()) {
      score = std::stod(std::string(it, row.end()));
    }
    results.emplace_back(ParselessLM::FoundReading{key, score});
  }
  return results;
}

}  // namespace McBopomofo
