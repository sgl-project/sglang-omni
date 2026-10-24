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

#ifndef SRC_ENGINE_GRAMAMBULAR2_READING_GRID_H_
#define SRC_ENGINE_GRAMAMBULAR2_READING_GRID_H_

#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "language_model.h"

namespace Formosa::Gramambular2 {

// A grid for deriving the most likely hidden values from a series of
// observations. For our purpose, the observations are Bopomofo readings, and
// the hidden values are the actual Mandarin words. This can also be used for
// segmentation: in that case, the observations are Mandarin words, and the
// hidden values are the most likely groupings.
//
// While we use the terminology from hidden Markov model (HMM), the actual
// implementation is a much simpler Bayesian inference, since the underlying
// language model consists of only unigrams. Once we have put all plausible
// unigrams as nodes on the grid, a simple DAG shortest-path walk will give us
// the maximum likelihood estimation (MLE) for the hidden values.
class ReadingGrid {
 public:
  explicit ReadingGrid(std::shared_ptr<LanguageModel> lm)
      : lm_(std::move(lm)) {}

  void clear();

  [[nodiscard]] size_t length() const { return readings_.size(); }

  [[nodiscard]] size_t cursor() const { return cursor_; }

  void setCursor(size_t cursor);

  [[nodiscard]] std::string readingSeparator() const { return separator_; }

  void setReadingSeparator(const std::string& separator);

  bool insertReading(const std::string& reading);

  // Delete the reading before the cursor, like Backspace. Cursor will decrement
  // by one.
  bool deleteReadingBeforeCursor();

  // Delete the reading after the cursor, like Del. Cursor is unmoved.
  bool deleteReadingAfterCursor();

  static constexpr size_t kMaximumSpanLength = 8;
  static constexpr char kDefaultSeparator[] = "-";

  // A Node consists of a set of unigrams, a reading, and a spanning length.
  // The spanning length denotes the length of the node in the grid. The grid
  // is responsible for constructing its nodes. For Mandarin multi-character
  // phrases, the grid will join separate readings into a single combined
  // reading, and use that reading to retrieve the unigrams with that reading.
  // Node with two-character phrases (so two readings, or two syllables) will
  // then have a spanning length of 2.
  class Node {
   public:
    enum class OverrideType {
      kNone,
      // Override the node with a unigram value and a score such that the node
      // will almost always be favored by the walk.
      kOverrideValueWithHighScore,
      // Override the node with a unigram value but with the score of the
      // top unigram. For example, if the unigrams in the node are ("a", -1),
      // ("b", -2), ("c", -10), overriding using this type for "c" will cause
      // the node to return the value "c" with the score -1. This is used for
      // soft-override such as from a suggestion. The node with the override
      // value will very likely be favored by a walk, but it does not prevent
      // other nodes from prevailing, which would be the case if
      // kOverrideValueWithHighScore was used.
      kOverrideValueWithScoreFromTopUnigram
    };

    Node(std::string reading, size_t spanningLength,
         std::vector<LanguageModel::Unigram> unigrams)
        : reading_(std::move(reading)),
          spanningLength_(spanningLength),
          unigrams_(std::move(unigrams)),
          unigramIter_(unigrams_.begin()),
          overrideType_(OverrideType::kNone) {}

    // An explicit copy ctor is needed since unigramIter_ must come from
    // this.unigram_, not copied.
    Node(const Node& o)
        : reading_(o.reading_),
          spanningLength_(o.spanningLength_),
          unigrams_(o.unigrams_),
          unigramIter_(
              std::next(unigrams_.begin(),
                        std::distance(o.unigrams_.begin(), o.unigramIter_))),
          overrideType_(o.overrideType_) {}

    [[nodiscard]] const std::string& reading() const { return reading_; }

    [[nodiscard]] size_t spanningLength() const { return spanningLength_; }

    [[nodiscard]] const std::vector<LanguageModel::Unigram>& unigrams() const {
      return unigrams_;
    }

    // Returns the top or overridden unigram.
    [[nodiscard]] LanguageModel::Unigram currentUnigram() const;

    [[nodiscard]] std::string value() const;

    [[nodiscard]] double score() const;

    [[nodiscard]] bool isOverridden() const;

    void reset();

    bool selectOverrideUnigram(const std::string& value, OverrideType type);

    // A sufficiently high score to cause the walk to go through an overriding
    // node. Although this can be 0, setting it to a positive value has the
    // desirable side effect that it reduces the competition of "free-floating"
    // multiple-character phrases. For example, if the user override for
    // reading "a b c" is "A B c", using the uppercase as the overriding node,
    // now the standalone c may have to compete with a phrase with reading "bc",
    // which in some pathological cases may actually cause the shortest path to
    // be A->bc, especially when A and B use the zero overriding score, as they
    // leave "c" alone to compete with "bc", and whether the path A-B is favored
    // now solely depends on that competition. A positive value favors the route
    // A->B, which gives "c" a better chance.
    static constexpr double kOverridingScore = 42;

   protected:
    const std::string reading_;
    const size_t spanningLength_;
    const std::vector<LanguageModel::Unigram> unigrams_;
    std::vector<LanguageModel::Unigram>::const_iterator unigramIter_;
    OverrideType overrideType_;
  };

  using NodePtr = std::shared_ptr<Node>;

  // Find, in a span at the cursor, the first node satisfying the predicate.
  // Returns std::nullopt if not found.
  std::optional<NodePtr> findInSpan(
      size_t cursor,
      const std::function<bool(const NodePtr&)>& predicate) const;

  struct WalkResult {
    std::vector<NodePtr> nodes;
    size_t totalReadings = 0;
    size_t vertices = 0;
    size_t edges = 0;
    uint64_t elapsedMicroseconds = 0;

    // Convenient method for finding the node at the cursor. Returns
    // nodes.cend() if the value of cursor argument doesn't make sense. An
    // optional ourCursorPastNode argument can be used to obtain the cursor
    // position that is right past the node at cursor, and will be only be set
    // if it's not nullptr and the returned iterator is not nodes.cend().
    std::vector<NodePtr>::const_iterator findNodeAt(
        size_t cursor, size_t* outCursorPastNode = nullptr) const;

    std::vector<std::string> valuesAsStrings() const;
    std::vector<std::string> readingsAsStrings() const;

    // Makes a copy with the nodes also being copies instead of refernces to
    // those in the current grid.
    //
    // For performance reasons, nodes is a vector of shared ptrs to those
    // nodes in the referenced grid. If the intent is to have the walk capture
    // the current state of the grid before the grid mutates, the default
    // behavior will not be enough. Instead, use this to make sure that the
    // nodes are correctly copied.
    WalkResult copyWithFixedNodes() const {
      std::vector<NodePtr> copiedNodes;
      for (const auto& n : nodes) {
        copiedNodes.push_back(std::make_shared<Node>(*n));
      }

      return WalkResult{copiedNodes, totalReadings, vertices, edges,
                        elapsedMicroseconds};
    }
  };

  WalkResult walk();

  struct Candidate {
    Candidate(std::string r, std::string v, std::string rv = "")
        : reading(std::move(r)), value(std::move(v)), rawValue(std::move(rv)) {}
    const std::string reading;
    const std::string value;
    const std::string rawValue;
  };

  // Returns all candidate values at the location. If spans are not empty and
  // loc is at the end of the spans, (loc - 1) is used, so that the caller does
  // not have to care about this boundary condition.
  std::vector<Candidate> candidatesAt(size_t loc);

  // Adds weight to the node with the unigram that has the designated candidate
  // value and applies the desired override type, essentially resulting in user
  // override. An overridden node would influence the grid walk to favor walking
  // through it.
  bool overrideCandidate(size_t loc, const Candidate& candidate,
                         Node::OverrideType overrideType =
                             Node::OverrideType::kOverrideValueWithHighScore);

  // Same as the method above, but since the string candidate value is used, if
  // there are multiple nodes (of different spanning length) that have the same
  // unigram value, it's not guaranteed which node will be selected.
  bool overrideCandidate(size_t loc, const std::string& candidate,
                         Node::OverrideType overrideType =
                             Node::OverrideType::kOverrideValueWithHighScore);

  // A span is a collection of nodes that share the same starting location.
  class Span {
   public:
    void clear();
    void add(const NodePtr& node);
    void removeNodesOfOrLongerThan(size_t length);
    [[nodiscard]] const NodePtr& nodeOf(size_t length) const;
    [[nodiscard]] size_t maxLength() const { return maxLength_; }

   protected:
    std::array<NodePtr, kMaximumSpanLength> nodes_;
    size_t maxLength_ = 0;
  };

  // A language model wrapper that always returns score-ranked unigrams.
  class ScoreRankedLanguageModel : public LanguageModel {
   public:
    explicit ScoreRankedLanguageModel(std::shared_ptr<LanguageModel> lm)
        : lm_(std::move(lm)) {
      assert(lm_ != nullptr);
    }
    std::vector<Unigram> getUnigrams(const std::string& reading) override;
    bool hasUnigrams(const std::string& reading) override;

   protected:
    std::shared_ptr<LanguageModel> lm_;
  };

  [[nodiscard]] const std::vector<Span>& spans() const { return spans_; }

  [[nodiscard]] const std::vector<std::string>& readings() const {
    return readings_;
  }

 protected:
  size_t cursor_ = 0;
  std::string separator_ = kDefaultSeparator;
  std::vector<std::string> readings_;
  std::vector<Span> spans_;
  ScoreRankedLanguageModel lm_;

  // Internal methods for maintaining the grid.

  enum class EditType {
    kInsertion,
    kDeletion,
  };

  void expandGridAt(size_t loc);
  void shrinkGridAt(size_t loc);
  void removeAffectedNodes(size_t loc);
  void insert(size_t loc, const NodePtr& node);
  std::string combineReading(std::vector<std::string>::const_iterator begin,
                             std::vector<std::string>::const_iterator end);
  bool hasNodeAt(size_t loc, size_t readingLen, const std::string& reading);
  void update(size_t loc, EditType editType);

  // Internal implementation of overrideCandidate, with an optional reading.
  bool overrideCandidate(size_t loc, const std::string* reading,
                         const std::string& value,
                         Node::OverrideType overrideType);

  struct NodeInSpan {
    NodePtr node;
    size_t spanIndex = 0;
  };

  // Find all nodes that overlap with the location. The return value is a list
  // of nodes along with their starting location in the grid.
  std::vector<NodeInSpan> overlappingNodesAt(size_t loc) const;
};

}  // namespace Formosa::Gramambular2

#endif  // SRC_ENGINE_GRAMAMBULAR2_READING_GRID_H_
