// SPDX-License-Identifier: Apache-2.0
// The normalizer follows mlx-audio's zonos2/textnorm.py rule for rule. Python's
// re is Unicode-aware (\w, \d, \s, \b); PCRE2 with UTF and UCP matches the same
// classes, and numbers are read as Python's int() reads them, in any decimal
// script and of any length.
#include "zonos3/zonos3_text.h"

#define PCRE2_CODE_UNIT_WIDTH 8
#include <CommonCrypto/CommonDigest.h>
#include <CoreFoundation/CoreFoundation.h>
#include <pcre2.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

#include "common/text.h"

namespace zonos3 {

namespace {

// The SuperBPE id layout the backbone was trained with.
constexpr int kTokenOffset = 4;
constexpr int kBeginOfText = 2;
constexpr int kEndOfText = 3;

// The zero of every Unicode decimal digit run (category Nd, Unicode 15.0, as
// Python's unicodedata); a digit's value is its distance from the run's zero.
constexpr std::array<uint32_t, 68> kDecimalZeros = {
    0x30,    0x660,   0x6f0,   0x7c0,   0x966,   0x9e6,   0xa66,   0xae6,
    0xb66,   0xbe6,   0xc66,   0xce6,   0xd66,   0xde6,   0xe50,   0xed0,
    0xf20,   0x1040,  0x1090,  0x17e0,  0x1810,  0x1946,  0x19d0,  0x1a80,
    0x1a90,  0x1b50,  0x1bb0,  0x1c40,  0x1c50,  0xa620,  0xa8d0,  0xa900,
    0xa9d0,  0xa9f0,  0xaa50,  0xabf0,  0xff10,  0x104a0, 0x10d30, 0x11066,
    0x110f0, 0x11136, 0x111d0, 0x112f0, 0x11450, 0x114d0, 0x11650, 0x116c0,
    0x11730, 0x118e0, 0x11950, 0x11c50, 0x11d50, 0x11da0, 0x11f50, 0x16a60,
    0x16ac0, 0x16b50, 0x1d7ce, 0x1d7d8, 0x1d7e2, 0x1d7ec, 0x1d7f6, 0x1e140,
    0x1e2f0, 0x1e4f0, 0x1e950, 0x1fbf0};

constexpr std::array<const char *, 10> kOnes = {
    "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine"};
constexpr std::array<const char *, 10> kTeens = {
    "ten",     "eleven",  "twelve",    "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"};
constexpr std::array<const char *, 10> kTens = {
    "",      "",      "twenty",  "thirty", "forty",
    "fifty", "sixty", "seventy", "eighty", "ninety"};
// Scales by their power of ten, largest first.
constexpr std::array<std::pair<int, const char *>, 4> kScales = {
    {{12, "trillion"}, {9, "billion"}, {6, "million"}, {3, "thousand"}}};
constexpr std::array<const char *, 12> kMonthNames = {
    "january", "february", "march",     "april",   "may",      "june",
    "july",    "august",   "september", "october", "november", "december"};

const std::map<std::string, std::string> &OrdinalExceptions() {
  static const std::map<std::string, std::string> exceptions = {
      {"zero", "zeroth"}, {"one", "first"}, {"two", "second"},
      {"three", "third"}, {"five", "fifth"}, {"eight", "eighth"},
      {"nine", "ninth"},  {"twelve", "twelfth"}};
  return exceptions;
}

const std::map<std::string, std::string> &MonthsByName() {
  static const std::map<std::string, std::string> months = {
      {"jan", "january"},   {"january", "january"},
      {"feb", "february"},  {"february", "february"},
      {"mar", "march"},     {"march", "march"},
      {"apr", "april"},     {"april", "april"},
      {"may", "may"},       {"jun", "june"},
      {"june", "june"},     {"jul", "july"},
      {"july", "july"},     {"aug", "august"},
      {"august", "august"}, {"sep", "september"},
      {"sept", "september"}, {"september", "september"},
      {"oct", "october"},   {"october", "october"},
      {"nov", "november"},  {"november", "november"},
      {"dec", "december"},  {"december", "december"}};
  return months;
}

struct Currency {
  const char *major_singular;
  const char *major_plural;
  const char *minor_singular;
  const char *minor_plural;
};

const std::map<std::string, Currency> &Currencies() {
  static const std::map<std::string, Currency> currencies = {
      {"$", {"dollar", "dollars", "cent", "cents"}},
      {"€", {"euro", "euros", "cent", "cents"}},
      {"£", {"pound", "pounds", "penny", "pence"}}};
  return currencies;
}

const std::map<std::string, std::string> &Quantities() {
  static const std::map<std::string, std::string> quantities = {
      {"k", "thousand"},       {"m", "million"},
      {"b", "billion"},        {"bn", "billion"},
      {"thousand", "thousand"}, {"million", "million"},
      {"billion", "billion"},  {"trillion", "trillion"}};
  return quantities;
}

const std::map<std::string, std::pair<std::string, std::string>> &Units() {
  static const std::map<std::string, std::pair<std::string, std::string>>
      units = {{"kg", {"kilogram", "kilograms"}},
               {"g", {"gram", "grams"}},
               {"mg", {"milligram", "milligrams"}},
               {"km", {"kilometer", "kilometers"}},
               {"cm", {"centimeter", "centimeters"}},
               {"mm", {"millimeter", "millimeters"}},
               {"m", {"meter", "meters"}},
               {"mi", {"mile", "miles"}},
               {"mph", {"mile per hour", "miles per hour"}},
               {"lb", {"pound", "pounds"}},
               {"lbs", {"pound", "pounds"}},
               {"ft", {"foot", "feet"}},
               {"in", {"inch", "inches"}},
               {"l", {"liter", "liters"}},
               {"ml", {"milliliter", "milliliters"}},
               {"hz", {"hertz", "hertz"}},
               {"khz", {"kilohertz", "kilohertz"}},
               {"mhz", {"megahertz", "megahertz"}},
               {"ghz", {"gigahertz", "gigahertz"}},
               {"kbps", {"kilobit per second", "kilobits per second"}},
               {"mbps", {"megabit per second", "megabits per second"}},
               {"gbps", {"gigabit per second", "gigabits per second"}},
               {"°c", {"degree celsius", "degrees celsius"}},
               {"°f", {"degree fahrenheit", "degrees fahrenheit"}}};
  return units;
}

// The reference joins its units longest first, ties in table order.
constexpr const char *kUnitPattern =
    "kbps|mbps|gbps|mph|lbs|khz|mhz|ghz|kg|mg|km|cm|mm|mi|lb|ft|in|ml|hz|"
    "°c|°f|g|m|l";
constexpr const char *kNumberPattern =
    "-?(?:\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.\\d+)?";

// A match's text and its named groups; an unmatched group reads as empty.
class Match {
public:
  Match(const pcre2_code *code, pcre2_match_data *match_data,
        const std::string &subject)
      : code_(code), match_data_(match_data), subject_(subject) {}

  std::string Whole() const { return Group(0); }
  std::string Named(const char *name) const {
    const int number = pcre2_substring_number_from_name(
        code_, reinterpret_cast<PCRE2_SPTR>(name));
    if (number < 0) {
      throw std::logic_error(std::string("no group named ") + name);
    } else {
    }
    return Group(number);
  }
  bool HasNamed(const char *name) const { return !Named(name).empty(); }

private:
  std::string Group(int number) const {
    const PCRE2_SIZE *ovector = pcre2_get_ovector_pointer(match_data_);
    if (number >= static_cast<int>(pcre2_get_ovector_count(match_data_)) ||
        ovector[2 * number] == PCRE2_UNSET) {
      return "";
    } else {
      return subject_.substr(ovector[2 * number],
                             ovector[2 * number + 1] - ovector[2 * number]);
    }
  }

  const pcre2_code *code_;
  pcre2_match_data *match_data_;
  const std::string &subject_;
};

class Regex {
public:
  explicit Regex(const std::string &pattern, bool caseless = false) {
    int error_code = 0;
    PCRE2_SIZE error_offset = 0;
    code_ = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern.c_str()),
                          PCRE2_ZERO_TERMINATED,
                          PCRE2_UTF | PCRE2_UCP | (caseless ? PCRE2_CASELESS : 0),
                          &error_code, &error_offset, nullptr);
    if (code_ == nullptr) {
      throw std::logic_error("cannot compile " + pattern);
    } else {
    }
  }
  ~Regex() { pcre2_code_free(code_); }
  Regex(const Regex &) = delete;
  Regex &operator=(const Regex &) = delete;

  // Python's re.sub: every non-overlapping match, left to right, replaced by
  // replace's result.
  std::string
  Substitute(const std::string &text,
             const std::function<std::string(const Match &)> &replace) const {
    std::unique_ptr<pcre2_match_data, decltype(&pcre2_match_data_free)>
        match_data(pcre2_match_data_create_from_pattern(code_, nullptr),
                   pcre2_match_data_free);
    std::string out;
    size_t offset = 0;
    size_t copied = 0;
    while (offset <= text.size()) {
      const int result =
          pcre2_match(code_, reinterpret_cast<PCRE2_SPTR>(text.data()),
                      text.size(), offset, 0, match_data.get(), nullptr);
      if (result < 0) {
        break;
      } else {
      }
      const PCRE2_SIZE *ovector = pcre2_get_ovector_pointer(match_data.get());
      const size_t begin = ovector[0];
      const size_t end = ovector[1];
      out.append(text, copied, begin - copied);
      out += replace(Match(code_, match_data.get(), text));
      copied = end;
      if (end == begin) {
        // An empty match moves on one code point, as re.sub does.
        if (end < text.size()) {
          size_t next = end + 1;
          while (next < text.size() &&
                 (static_cast<uint8_t>(text[next]) & 0xC0) == 0x80) {
            ++next;
          }
          out.append(text, end, next - end);
          copied = next;
          offset = next;
        } else {
          break;
        }
      } else {
        offset = end;
      }
    }
    out.append(text, copied, std::string::npos);
    return out;
  }

private:
  pcre2_code *code_ = nullptr;
};

// The decimal digits of text in ASCII; a code point that is not a decimal
// digit is kept as it is.
std::string AsciiDigits(const std::string &text) {
  std::string out;
  for (const uint32_t code_point : omni_common::CodePoints(text)) {
    const auto zero = std::upper_bound(kDecimalZeros.begin(),
                                       kDecimalZeros.end(), code_point);
    if (zero != kDecimalZeros.begin() && code_point - *(zero - 1) < 10) {
      out += static_cast<char>('0' + (code_point - *(zero - 1)));
    } else {
      omni_common::AppendUtf8(code_point, out);
    }
  }
  return out;
}

// A nonnegative integer as decimal digits without leading zeros ("0" for
// zero), so values of any length keep Python's int() arithmetic.
struct Integer {
  bool negative = false;
  std::string digits = "0";

  static Integer Parse(const std::string &text) {
    Integer value;
    std::string ascii = AsciiDigits(text);
    if (!ascii.empty() && ascii.front() == '-') {
      value.negative = true;
      ascii.erase(0, 1);
    } else {
    }
    const size_t first = ascii.find_first_not_of('0');
    value.digits = first == std::string::npos ? "0" : ascii.substr(first);
    if (value.digits == "0") {
      value.negative = false;
    } else {
    }
    return value;
  }
  bool IsZero() const { return digits == "0"; }
  // The value when it fits in a long long, else none.
  std::optional<long long> Small() const {
    if (digits.size() > 17) {
      return std::nullopt;
    } else {
      const long long magnitude = std::stoll(digits);
      return negative ? -magnitude : magnitude;
    }
  }
};

std::string IntegerWords(const Integer &value);

std::string SmallWords(long long value) {
  return IntegerWords(Integer::Parse(std::to_string(value)));
}

std::string IntegerWords(const Integer &value) {
  if (value.negative) {
    return "negative " + IntegerWords(Integer{false, value.digits});
  } else {
  }
  const std::string &digits = value.digits;
  if (digits.size() <= 3) {
    const int number = std::stoi(digits);
    if (number < 10) {
      return kOnes[number];
    } else if (number < 20) {
      return kTeens[number - 10];
    } else if (number < 100) {
      return number % 10 == 0 ? std::string(kTens[number / 10])
                              : std::string(kTens[number / 10]) + " " +
                                    kOnes[number % 10];
    } else {
      const std::string words = std::string(kOnes[number / 100]) + " hundred";
      return number % 100 == 0 ? words
                               : words + " " + SmallWords(number % 100);
    }
  } else {
  }
  for (const auto &[power, scale_name] : kScales) {
    if (static_cast<int>(digits.size()) > power) {
      const Integer quotient =
          Integer::Parse(digits.substr(0, digits.size() - power));
      const Integer rest = Integer::Parse(digits.substr(digits.size() - power));
      const std::string words = IntegerWords(quotient) + " " + scale_name;
      return rest.IsZero() ? words : words + " " + IntegerWords(rest);
    } else {
    }
  }
  return digits;
}

std::string YearWords(long long value) {
  if (1000 <= value && value <= 1999) {
    const long long prefix = value / 100;
    const long long rest = value % 100;
    return rest == 0 ? SmallWords(prefix) + " hundred"
                     : SmallWords(prefix) + " " + SmallWords(rest);
  } else if (2000 <= value && value <= 2009) {
    const long long rest = value - 2000;
    return rest == 0 ? "two thousand" : "two thousand " + SmallWords(rest);
  } else if (2010 <= value && value <= 2099) {
    return "twenty " + SmallWords(value - 2000);
  } else {
    return SmallWords(value);
  }
}

std::string DigitWords(const std::string &digits) {
  std::string words;
  for (const char digit : AsciiDigits(digits)) {
    words += (words.empty() ? "" : " ") + std::string(kOnes[digit - '0']);
  }
  return words;
}

std::string WithoutCommas(const std::string &value) {
  std::string out;
  for (const char character : value) {
    if (character != ',') {
      out += character;
    } else {
    }
  }
  return out;
}

std::string NumberWords(const std::string &raw_value) {
  std::string value = WithoutCommas(raw_value);
  const bool negative = !value.empty() && value.front() == '-';
  if (negative) {
    value.erase(0, 1);
  } else {
  }
  const std::string prefix = negative ? "negative " : "";
  const size_t point = value.find('.');
  if (point != std::string::npos) {
    const std::string integer = value.substr(0, point);
    return prefix +
           IntegerWords(Integer::Parse(integer.empty() ? "0" : integer)) +
           " point " + DigitWords(value.substr(point + 1));
  } else if (omni_common::CodePointCount(value) > 1 && value.front() == '0') {
    // Only an ASCII leading zero reads digit by digit, as in Python.
    return prefix + DigitWords(value);
  } else {
    return prefix + IntegerWords(Integer::Parse(value));
  }
}

std::string OrdinalWords(const Integer &value) {
  std::string words = IntegerWords(value);
  const size_t last_space = words.rfind(' ');
  const size_t last_begin = last_space == std::string::npos ? 0 : last_space + 1;
  const std::string last = words.substr(last_begin);
  const auto exception = OrdinalExceptions().find(last);
  if (exception != OrdinalExceptions().end()) {
    return words.substr(0, last_begin) + exception->second;
  } else if (!last.empty() && last.back() == 'y') {
    return words.substr(0, words.size() - 1) + "ieth";
  } else {
    return words + "th";
  }
}

bool NumericIsOne(const std::string &value) {
  const std::string ascii = AsciiDigits(WithoutCommas(value));
  char *end = nullptr;
  const double number = std::strtod(ascii.c_str(), &end);
  return end != ascii.c_str() && *end == '\0' && number == 1.0;
}

// str.lower(): Unicode's default lowercase mapping.
std::string UnicodeLower(const std::string &text) {
  CFStringRef source = CFStringCreateWithBytes(
      kCFAllocatorDefault, reinterpret_cast<const UInt8 *>(text.data()),
      static_cast<CFIndex>(text.size()), kCFStringEncodingUTF8, false);
  if (source == nullptr) {
    return text;
  } else {
  }
  CFMutableStringRef lowered =
      CFStringCreateMutableCopy(kCFAllocatorDefault, 0, source);
  CFRelease(source);
  CFStringLowercase(lowered, nullptr);
  const CFIndex length = CFStringGetLength(lowered);
  CFIndex byte_count = 0;
  CFStringGetBytes(lowered, CFRangeMake(0, length), kCFStringEncodingUTF8, 0,
                   false, nullptr, 0, &byte_count);
  std::string out(static_cast<size_t>(byte_count), '\0');
  CFStringGetBytes(lowered, CFRangeMake(0, length), kCFStringEncodingUTF8, 0,
                   false, reinterpret_cast<UInt8 *>(out.data()), byte_count,
                   nullptr);
  CFRelease(lowered);
  return out;
}

std::string AsciiLower(std::string text) {
  for (char &character : text) {
    if (character >= 'A' && character <= 'Z') {
      character = static_cast<char>(character - 'A' + 'a');
    } else {
    }
  }
  return text;
}

std::optional<std::string> FormatDate(long long month, long long day,
                                      const std::string &year) {
  if (month < 1 || month > 12 || day < 1 || day > 31) {
    return std::nullopt;
  } else {
  }
  std::string words = std::string(kMonthNames[month - 1]) + " " +
                      OrdinalWords(Integer::Parse(std::to_string(day)));
  if (!year.empty()) {
    long long year_number = *Integer::Parse(year).Small();
    if (year_number < 100) {
      year_number += year_number < 50 ? 2000 : 1900;
    } else {
    }
    words += " " + YearWords(year_number);
  } else {
  }
  return words;
}

long long SmallValue(const std::string &digits) {
  return *Integer::Parse(digits).Small();
}

std::string ReplaceDate(const Match &match) {
  const std::optional<std::string> words =
      FormatDate(SmallValue(match.Named("month")),
                 SmallValue(match.Named("day")), match.Named("year"));
  return words.has_value() ? *words : match.Whole();
}

std::string ReplaceMonthDate(const Match &match) {
  std::string month = match.Named("month");
  while (!month.empty() && month.back() == '.') {
    month.pop_back();
  }
  const auto found = MonthsByName().find(UnicodeLower(month));
  const long long day = SmallValue(match.Named("day"));
  if (found == MonthsByName().end() || day < 1 || day > 31) {
    return match.Whole();
  } else {
  }
  std::string words = found->second + " " +
                      OrdinalWords(Integer::Parse(std::to_string(day)));
  if (match.HasNamed("year")) {
    words += " " + YearWords(SmallValue(match.Named("year")));
  } else {
  }
  return words;
}

std::string ReplaceTime(const Match &match) {
  const long long hour = SmallValue(match.Named("hour"));
  const long long minute = SmallValue(match.Named("minute"));
  const std::string second = match.Named("second");
  const std::string suffix = match.Named("suffix");
  long long spoken_hour = hour;
  if (!suffix.empty()) {
    spoken_hour = hour % 12 == 0 ? 12 : hour % 12;
  } else {
  }
  std::string words = SmallWords(spoken_hour);
  if (minute != 0) {
    words += " " + (minute < 10 ? "oh " + SmallWords(minute) : SmallWords(minute));
  } else {
  }
  if (!second.empty() && SmallValue(second) != 0) {
    words += " and " + SmallWords(SmallValue(second)) + " seconds";
  } else {
  }
  if (!suffix.empty()) {
    words += (suffix.front() == 'a' || suffix.front() == 'A') ? " a m" : " p m";
  } else {
  }
  return words;
}

std::string ReplaceMoney(const Match &match) {
  const Currency &currency = Currencies().at(match.Named("symbol"));
  const std::string amount = WithoutCommas(match.Named("amount"));
  const std::string quantity = match.Named("quantity");
  if (!quantity.empty()) {
    const std::string major = NumericIsOne(amount) ? currency.major_singular
                                                   : currency.major_plural;
    return NumberWords(amount) + " " + Quantities().at(AsciiLower(quantity)) +
           " " + major;
  } else {
  }
  std::string integer = amount;
  long long cents = 0;
  const size_t point = amount.find('.');
  if (point != std::string::npos) {
    integer = amount.substr(0, point);
    const std::vector<uint32_t> fraction =
        omni_common::CodePoints(amount.substr(point + 1) + "00");
    std::string first_two;
    omni_common::AppendUtf8(fraction[0], first_two);
    omni_common::AppendUtf8(fraction[1], first_two);
    cents = SmallValue(first_two);
  } else {
  }
  const Integer dollars = Integer::Parse(integer);
  std::vector<std::string> parts;
  if (!dollars.IsZero()) {
    const bool is_one = !dollars.negative && dollars.digits == "1";
    parts.push_back(IntegerWords(dollars) + " " +
                    (is_one ? currency.major_singular : currency.major_plural));
  } else {
  }
  if (cents != 0) {
    parts.push_back(SmallWords(cents) + " " +
                    (cents == 1 ? currency.minor_singular
                                : currency.minor_plural));
  } else {
  }
  if (parts.empty()) {
    return std::string("zero ") + currency.major_plural;
  } else if (parts.size() == 1) {
    return parts[0];
  } else {
    return parts[0] + " and " + parts[1];
  }
}

std::string ReplaceUnit(const Match &match) {
  const std::string number = match.Named("number");
  const auto found = Units().find(UnicodeLower(match.Named("unit")));
  if (found == Units().end()) {
    // The reference's table lookup raises for a caseless match it does not
    // list, such as a long s.
    throw std::invalid_argument("cannot normalize the unit " +
                                match.Named("unit"));
  } else {
  }
  const auto &[singular, plural] = found->second;
  return NumberWords(number) + " " + (NumericIsOne(number) ? singular : plural);
}

std::string ReplaceFraction(const Match &match) {
  const Integer numerator = Integer::Parse(match.Named("numerator"));
  const Integer denominator = Integer::Parse(match.Named("denominator"));
  const bool is_one = numerator.digits == "1";
  std::string denominator_words;
  if (denominator.digits == "2") {
    denominator_words = is_one ? "half" : "halves";
  } else if (denominator.digits == "4") {
    denominator_words = is_one ? "quarter" : "quarters";
  } else {
    denominator_words = OrdinalWords(denominator) + (is_one ? "" : "s");
  }
  return IntegerWords(numerator) + " " + denominator_words;
}

std::string ReplacePhone(const Match &match) {
  // Python keeps every decimal digit and then drops a leading ASCII "1".
  std::vector<uint32_t> digits;
  for (const uint32_t code_point : omni_common::CodePoints(match.Whole())) {
    std::string one;
    omni_common::AppendUtf8(code_point, one);
    const std::string ascii = AsciiDigits(one);
    if (ascii.size() == 1 && ascii[0] >= '0' && ascii[0] <= '9') {
      digits.push_back(code_point);
    } else {
    }
  }
  if (digits.size() == 11 && digits.front() == '1') {
    digits.erase(digits.begin());
  } else {
  }
  std::string text;
  for (const uint32_t code_point : digits) {
    omni_common::AppendUtf8(code_point, text);
  }
  return DigitWords(text);
}

std::string ReplaceAcronym(const Match &match) {
  std::string words;
  for (const char letter : AsciiLower(match.Whole())) {
    words += (words.empty() ? "" : " ") + std::string(1, letter);
  }
  return words;
}

std::string ReplaceAll(const std::string &text, const std::string &from,
                       const std::string &to) {
  std::string out;
  size_t offset = 0;
  while (true) {
    const size_t found = text.find(from, offset);
    if (found == std::string::npos) {
      out.append(text, offset, std::string::npos);
      return out;
    } else {
      out.append(text, offset, found - offset);
      out += to;
      offset = found + from.size();
    }
  }
}

std::string Sha256Hex(const std::string &bytes) {
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.data());
  static constexpr char kHex[] = "0123456789abcdef";
  std::string hex;
  for (const unsigned char byte : digest) {
    hex += kHex[byte >> 4];
    hex += kHex[byte & 0x0F];
  }
  return hex;
}

} // namespace

std::string NormalizeEnglishText(const std::string &input) {
  // Python's \w is a letter, a number or "_" (no combining marks, which
  // PCRE2's UCP \w also takes), its \b the edge of a run of them, and its \s
  // str.isspace().
  static const std::string word = "[\\p{L}\\p{N}_]";
  static const std::string word_body = "\\p{L}\\p{N}_";
  static const std::string boundary =
      "(?:(?<=" + word + ")(?!" + word + ")|(?<!" + word + ")(?=" + word + "))";
  static const std::string space_body =
      "\\t\\n\\x0b\\x0c\\r\\x1c-\\x1f \\x{85}\\x{a0}\\x{1680}\\x{2000}-"
      "\\x{200a}\\x{2028}\\x{2029}\\x{202f}\\x{205f}\\x{3000}";
  static const std::string space_class = "[" + space_body + "]";
  static const std::string not_after_word = "(?<!" + word + ")";
  static const std::string not_before_word = "(?!" + word + ")";
  static const std::string number = std::string(kNumberPattern);
  static const Regex iso_date(boundary +
                              "(?P<year>\\d{4})-(?P<month>\\d{1,2})-"
                              "(?P<day>\\d{1,2})" +
                              boundary);
  static const Regex slash_date(boundary +
                                "(?P<month>\\d{1,2})/(?P<day>\\d{1,2})/"
                                "(?P<year>\\d{2,4})" +
                                boundary);
  static const Regex month_date(
      boundary +
          "(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
          "jun(?:e)?|jul(?:y)?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|"
          "nov(?:ember)?|dec(?:ember)?)\\.?" +
          space_class + "+(?P<day>\\d{1,2})(?:st|nd|rd|th)?(?:," + space_class +
          "*|" + space_class + "+)?(?P<year>\\d{4})?" + boundary,
      true);
  static const Regex time(not_after_word +
                          "(?P<hour>[01]?\\d|2[0-3]):(?P<minute>[0-5]\\d)"
                          "(?::(?P<second>[0-5]\\d))?(?:" +
                          space_class + "*(?P<suffix>[aApP]\\.?" + space_class +
                          "?[mM]\\.?))?" + boundary);
  static const Regex money(not_after_word + "(?P<symbol>[$€£])" +
                           space_class + "*(?P<amount>" + number + ")(?:" +
                           space_class +
                           "*(?P<quantity>thousand|million|billion|trillion|"
                           "bn|BN|k|K|m|M|b|B)" +
                           boundary + ")?");
  static const Regex percent(not_after_word + "(?P<number>" + number + ")" +
                             space_class + "*%");
  static const Regex unit(not_after_word + "(?P<number>" + number + ")" +
                              space_class + "*(?P<unit>" + kUnitPattern + ")" +
                              boundary,
                          true);
  static const Regex fraction(not_after_word +
                              "(?P<numerator>\\d+)/(?P<denominator>\\d+)" +
                              not_before_word);
  static const Regex ordinal(not_after_word +
                             "(?P<number>-?\\d{1,3}(?:,\\d{3})+|-?\\d+)"
                             "(?:st|nd|rd|th)" +
                             boundary);
  static const Regex decimal("(?<![" + word_body +
                             ".])(?P<number>-?(?:\\d{1,3}(?:,\\d{3})+|\\d+)"
                             "\\.\\d+)(?![" +
                             word_body + ".])");
  static const std::string separator = "[-." + space_body + "]?";
  static const Regex phone(not_after_word + "(?:\\+?1" + separator +
                           ")?(?:(?:\\(?\\d{3}\\)?" + separator + ")?\\d{3}" +
                           separator + "\\d{4})" + not_before_word);
  static const Regex integer("(?<![" + word_body +
                             ".])(?P<number>-?(?:\\d{1,3}(?:,\\d{3})+|\\d+))(?![" +
                             word_body + ".])");
  static const Regex acronym(boundary + "[A-Z]{2,}" + boundary);
  static const Regex space(space_class + "+");
  static const Regex space_before_punctuation(space_class + "+([.!?,;:])");

  std::string text = iso_date.Substitute(input, ReplaceDate);
  text = month_date.Substitute(text, ReplaceMonthDate);
  text = slash_date.Substitute(text, ReplaceDate);
  text = time.Substitute(text, ReplaceTime);
  text = phone.Substitute(text, ReplacePhone);
  text = money.Substitute(text, ReplaceMoney);
  text = percent.Substitute(text, [](const Match &match) {
    return NumberWords(match.Named("number")) + " percent";
  });
  text = unit.Substitute(text, ReplaceUnit);
  text = fraction.Substitute(text, ReplaceFraction);
  text = ordinal.Substitute(text, [](const Match &match) {
    return OrdinalWords(Integer::Parse(WithoutCommas(match.Named("number"))));
  });
  const auto number_words = [](const Match &match) {
    return NumberWords(match.Named("number"));
  };
  text = decimal.Substitute(text, number_words);
  text = integer.Substitute(text, number_words);
  text = acronym.Substitute(text, ReplaceAcronym);
  text = ReplaceAll(text, "&", " and ");
  text = ReplaceAll(text, "@", " at ");
  text = space_before_punctuation.Substitute(text, [](const Match &match) {
    const std::string whole = match.Whole();
    return whole.substr(whole.size() - 1);
  });
  text = space.Substitute(text, [](const Match &) { return std::string(" "); });
  return omni_common::StripUnicodeWhitespace(text);
}

TextFrontend::TextFrontend(const std::filesystem::path &model_directory,
                           const nlohmann::json &tokenizer_spec,
                           int text_pad_id)
    : tokenizer_(model_directory) {
  if (tokenizer_spec.at("kind") != "superbpe" ||
      tokenizer_spec.at("offset") != kTokenOffset ||
      tokenizer_spec.at("bos") != kBeginOfText ||
      tokenizer_spec.at("eos") != kEndOfText) {
    throw std::runtime_error("unsupported SuperBPE id convention");
  } else if (text_pad_id !=
             tokenizer_spec.at("vocab_size").get<int>() + kTokenOffset) {
    throw std::runtime_error("SuperBPE vocabulary does not fit the text column");
  } else {
  }
  const std::filesystem::path path =
      model_directory / tokenizer_spec.at("path").get<std::string>();
  std::ifstream stream(path, std::ios::binary);
  const std::string bytes((std::istreambuf_iterator<char>(stream)), {});
  if (!stream.good() && !stream.eof()) {
    throw std::runtime_error("cannot read " + path.string());
  } else if (Sha256Hex(bytes) != tokenizer_spec.at("sha256").get<std::string>()) {
    throw std::runtime_error(path.string() +
                             " does not match the backbone's SuperBPE SHA-256");
  } else {
  }
}

std::string TextFrontend::Normalize(const std::string &text,
                                    const std::string &language) {
  const std::string tag = UnicodeLower(ReplaceAll(language, "-", "_"));
  if (omni_common::StripUnicodeWhitespace(text).empty() ||
      (tag != "en" && tag != "en_us" && tag != "en_gb")) {
    return text;
  } else {
    return NormalizeEnglishText(text);
  }
}

std::vector<int> TextFrontend::Encode(const std::string &text) const {
  std::vector<int> ids = {kBeginOfText};
  for (const int id : tokenizer_.Encode(text)) {
    ids.push_back(id + kTokenOffset);
  }
  ids.push_back(kEndOfText);
  return ids;
}

} // namespace zonos3
