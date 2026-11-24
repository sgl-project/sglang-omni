package learn

import (
	"regexp"
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/Voxt/internal/dict"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// The word lists and the rules of keep are Voxt 1.x's
// (DictionaryHistoryScanCandidateValidator).

const (
	maxTermChars        = 48
	maxLatinWords       = 6
	maxLatinLetters     = 32
	maxCJKChars         = 6
	evidenceWindow      = 18 // characters kept on each side of a term to judge its context
	termBreaks          = "\r\n\v\f\u0085\u2028\u2029"
	sentencePunctuation = ".,:;!?，。：；！？、"
)

var (
	genericEnglishTerms = []string{"button", "company", "email", "file", "flight", "message", "model", "neither",
		"office", "prompt", "schedule", "setting", "station", "token", "train"}
	genericEnglishReferenceStarters = []string{"my", "our", "your", "their", "this", "that", "these", "those"}
	genericEnglishReferenceEndings  = []string{"company", "content", "data", "feature", "file", "function", "issue",
		"message", "model", "problem", "prompt", "result", "rule", "setting", "term", "text"}
	genericCJKTerms = []string{"会议", "公司", "地铁", "文件", "机场", "航班", "订单", "提示词", "模型", "火车",
		"邮件", "设置", "车次", "酒店", "高铁"}
	// In this order: the first prefix that fits is the one judged.
	genericChineseReferencePrefixes = []string{"这个", "那个", "这些", "那些", "这种", "那种", "我们", "你们", "他们",
		"她们", "它们", "我的", "你的", "他的", "她的", "它的", "我们的", "你们的", "他们的", "她们的", "它们的"}
	genericChineseReferenceSuffixes = []string{"规则", "问题", "功能", "内容", "结果", "消息", "词汇", "词语", "文本",
		"数据", "文件", "设置", "模型", "提示词", "方案", "接口", "公司", "事情", "情况"}
	travelKeywords = []string{"flight", "flights", "train", "trains", "station", "route", "航班", "车次", "列车",
		"火车", "高铁", "机票", "动车"}

	transportIdentifier = regexp.MustCompile(`^[A-Za-z]{1,3}\p{Nd}{2,4}$`)
)

// keep says whether a term a model returned for records is worth
// suggesting: a short term, not a generic word or a reference phrase, not
// a number, an address or a path; found in the records, and not the end
// of a route or a transport number there.
func keep(term string, records []string) bool {
	term = strings.TrimSpace(term)
	lower := strings.ToLower(term)
	characters := utf8.RuneCountInString(term)
	if dict.Check(dict.Clean(store.Term{Term: term})) != nil || characters > maxTermChars || strings.ContainsAny(term, termBreaks+sentencePunctuation) {
		return false
	}
	if strings.IndexFunc(term, func(r rune) bool { return !unicode.IsDigit(r) }) < 0 {
		return false
	}
	latinLetters, cjkChars := 0, 0
	for _, r := range term {
		switch {
		case r >= 'A' && r <= 'Z', r >= 'a' && r <= 'z':
			latinLetters++
		case r >= 0x3400 && r <= 0x4DBF, r >= 0x4E00 && r <= 0x9FFF, r >= 0x3040 && r <= 0x30FF, r >= 0x31F0 && r <= 0x31FF, r >= 0xAC00 && r <= 0xD7AF:
			cjkChars++
		}
	}
	words := strings.Fields(lower)
	latinWords := 0
	for _, word := range words {
		if strings.IndexFunc(word, func(r rune) bool { return r >= 'a' && r <= 'z' }) >= 0 {
			latinWords++
		}
	}
	if latinLetters > 0 && (latinWords > maxLatinWords || latinLetters > maxLatinLetters) {
		return false
	}
	if latinLetters == 0 && cjkChars > maxCJKChars {
		return false
	}
	if slices.Contains(genericEnglishTerms, lower) || slices.Contains(genericCJKTerms, term) {
		return false
	}
	if len(words) >= 2 && len(words) <= 4 && slices.Contains(genericEnglishReferenceStarters, words[0]) &&
		slices.Contains(genericEnglishReferenceEndings, words[len(words)-1]) {
		return false
	}
	if characters >= 3 && characters <= 8 {
		for _, prefix := range genericChineseReferencePrefixes {
			if remainder, found := strings.CutPrefix(term, prefix); found {
				if remainder != "" && slices.ContainsFunc(genericChineseReferenceSuffixes, func(suffix string) bool { return strings.HasSuffix(remainder, suffix) }) {
					return false
				}
				break
			}
		}
	}
	// A dot or a colon has already ruled out the addresses and the
	// absolute Windows paths; CI/CD is a term, a/b/c is a path.
	if strings.HasPrefix(term, "/") || strings.HasPrefix(term, "~/") || strings.Contains(term, `\`) || strings.Count(term, "/") >= 2 {
		return false
	}
	sample, found := evidence(lower, records)
	return found && !contextSpecific(term, sample)
}

// evidence is the first record that has the term (folded), cut to the
// characters around it.
func evidence(lower string, records []string) (string, bool) {
	for _, record := range records {
		foldedRecord := strings.ToLower(record)
		at := strings.Index(foldedRecord, lower)
		if at < 0 {
			continue
		}
		characters := []rune(record)
		start := utf8.RuneCountInString(foldedRecord[:at])
		end := start + utf8.RuneCountInString(lower)
		return string(characters[max(0, start-evidenceWindow):min(len(characters), end+evidenceWindow)]), true
	}
	return "", false
}

// contextSpecific says whether, around sample's mention of a trip, term
// is the end of a route or a transport number: names only for that one
// sentence.
func contextSpecific(term, sample string) bool {
	foldedSample, foldedTerm := strings.ToLower(sample), strings.ToLower(term)
	if !slices.ContainsFunc(travelKeywords, func(keyword string) bool { return strings.Contains(foldedSample, keyword) }) {
		return false
	}
	return strings.Contains(sample, term+"到") || strings.Contains(sample, "到"+term) ||
		strings.Contains(foldedSample, "from "+foldedTerm) || strings.Contains(foldedSample, "to "+foldedTerm) ||
		strings.Contains(foldedSample, foldedTerm+" to ") ||
		transportIdentifier.MatchString(term)
}
