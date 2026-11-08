package session

import (
	"strings"
	"unicode"
)

// endWord is the word that ends the recording (settings.Features.EndWord)
// when the speech model streams, "" otherwise: only a stream's live text
// can hear it.
func (r *run) endWord() string {
	if _, ok := r.streamer(); !ok {
		return ""
	}
	return r.settings.Features.EndWord
}

// hearPartial passes a live text to the recording loop while it listens
// for the end word (r.heard); the newest wins.
func (r *run) hearPartial(text string) {
	if r.heard == nil {
		return
	}
	select {
	case <-r.heard:
	default:
	}
	r.heard <- text
}

// withoutEndWord is text without word at its end, and whether word was
// there (Voxt 1.x's VoiceEndCommandFlow): the text's last word, alone or
// after a space or a punctuation mark, ignoring case and the spaces and
// punctuation after it. The punctuation before it goes with it.
func withoutEndWord(text, word string) (string, bool) {
	word = strings.Join(strings.Fields(word), " ")
	trimmed := []rune(strings.TrimRightFunc(text, isDelimiter))
	start := len(trimmed) - len([]rune(word))
	if word == "" || start < 0 || !strings.EqualFold(string(trimmed[start:]), word) || start > 0 && !isDelimiter(trimmed[start-1]) {
		return text, false
	}
	return strings.TrimRightFunc(string(trimmed[:start]), isDelimiter), true
}

// isDelimiter says whether r separates the end word from what comes
// before it: a space, a punctuation mark or a symbol.
func isDelimiter(r rune) bool {
	return unicode.IsSpace(r) || unicode.IsPunct(r) || unicode.IsSymbol(r)
}
