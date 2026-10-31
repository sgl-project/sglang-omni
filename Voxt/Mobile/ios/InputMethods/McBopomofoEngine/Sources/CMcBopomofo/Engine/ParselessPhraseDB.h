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

#ifndef SRC_ENGINE_PARSELESSPHRASEDB_H_
#define SRC_ENGINE_PARSELESSPHRASEDB_H_

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace McBopomofo {

constexpr std::string_view SORTED_PRAGMA_HEADER =
    "# format org.openvanilla.mcbopomofo.sorted\n";

// Defines phrase database that consists of (key, value, score) rows that are
// pre-sorted by the byte value of the keys. It is way faster than FastLM
// because it does not need to parse anything. Instead, it relies on the fact
// that the database is already sorted, and binary search is used to find the
// rows.
class ParselessPhraseDB {
 public:
  ParselessPhraseDB(const char* buf, size_t length,
                    bool validate_pragma = false);

  ParselessPhraseDB(const ParselessPhraseDB&) = delete;
  ParselessPhraseDB(ParselessPhraseDB&&) = delete;
  ParselessPhraseDB& operator=(const ParselessPhraseDB&) = delete;
  ParselessPhraseDB& operator=(ParselessPhraseDB&&) = delete;

  // Find the rows that match the key. Note that prefix match is used. If you
  // need exact match, the key will need to have a delimiter (usually a space)
  // at the end.
  std::vector<std::string_view> findRows(const std::string_view& key) const;

  const char* findFirstMatchingLine(const std::string_view& key) const;

  // Find the rows whose text past the key column plus the field separator
  // is a prefix match of the given value. For example, if the row is
  // "foo bar -1.00", the values "b", "ba", "bar", "bar ", "bar -1.00" are
  // valid prefix matches, whereas the value "barr" isn't. This performs linear
  // scan since, unlike lookup-by-key, it cannot take advantage of the fact that
  // the underlying data is sorted by keys.
  std::vector<std::string> reverseFindRows(const std::string_view& value) const;

  static bool ValidatePragma(const char* buf, size_t length);

  // Convenient function for validating and returning a DB instance. nullptr if
  // the block is empty or is not valid.
  static std::unique_ptr<ParselessPhraseDB> CreateValidatedDB(const char* buf,
                                                              size_t length);

 private:
  const char* begin_;
  const char* end_;
};

}  // namespace McBopomofo

#endif  // SRC_ENGINE_PARSELESSPHRASEDB_H_
