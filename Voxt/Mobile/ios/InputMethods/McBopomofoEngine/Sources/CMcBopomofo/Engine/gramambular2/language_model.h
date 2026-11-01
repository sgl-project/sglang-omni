// Copyright (c) 2022 and onwards Lukhnos Liu.
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

#ifndef SRC_ENGINE_GRAMAMBULAR2_LANGUAGE_MODEL_H_
#define SRC_ENGINE_GRAMAMBULAR2_LANGUAGE_MODEL_H_

#include <string>
#include <utility>
#include <vector>

namespace Formosa::Gramambular2 {

// Represents an n-gram model. For our purposes, only unigrams are used.
class LanguageModel {
 public:
  class Unigram;

  virtual ~LanguageModel() = default;

  // Returns unigrams matching the reading, or an empty vector if none is found.
  virtual std::vector<Unigram> getUnigrams(const std::string& reading) = 0;
  virtual bool hasUnigrams(const std::string& reading) = 0;

  // An immutable unigram with an actual value, along with a score, which is
  // usually a log probability from a language model.
  class Unigram {
   public:
    explicit Unigram(std::string val = "", double sc = 0,
                     std::string rawValue = "")
        : value_(std::move(val)), score_(sc), rawValue_(std::move(rawValue)) {}

    [[nodiscard]] const std::string& value() const { return value_; }
    [[nodiscard]] const std::string& rawValue() const { return rawValue_; }
    [[nodiscard]] double score() const { return score_; }

   private:
    std::string value_;
    double score_;
    std::string rawValue_;
  };
};

}  // namespace Formosa::Gramambular2

#endif  // SRC_ENGINE_GRAMAMBULAR2_LANGUAGE_MODEL_H_
