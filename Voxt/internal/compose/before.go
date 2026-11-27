package compose

import "strings"

// note (luojiaxuan): A dictation cleaned up while it is spoken (the session's live clean-up)
// goes to the model in blocks: the first block alone, as a whole
// dictation is; a later block with the cleaned text before it as context
// (Input.Before), so that the model reads what it refers to and the lists
// it continues, and answers with the block's text only.

// note (luojiaxuan): userMessage is in's user message for Transcribe and Translate: its
// input, after the cleaned text before it when there is one.
func userMessage(in Input) string {
	if in.Task != Transcribe || strings.TrimSpace(in.Before) == "" {
		return UserMessage(in.Text)
	}
	return BeforeMessage(in.Before, in.Text)
}

// note (luojiaxuan): BeforeMessage is the user message of a block after the first: the
// cleaned text before it, then its input.
func BeforeMessage(before, text string) string {
	return wrap("before", before) + "\n\n" + UserMessage(text)
}

// note (luojiaxuan): SentenceEnds are the byte offsets in text where a sentence or a line
// ends, after its mark and closing quotes and before the white space
// after them: where Pieces cuts first. The end of text is one when text
// ends with a sentence mark. A mark inside a snippet's mark is none.
func SentenceEnds(text string) []int {
	var ends []int
	for _, m := range boundaries[0].FindAllStringIndex(text, -1) {
		if inMark(text, m[0]) {
			continue
		}
		end := m[0] + len(strings.TrimRight(text[m[0]:m[1]], " \t\r\n"))
		if strings.TrimSpace(text[:end]) == "" || len(ends) > 0 && ends[len(ends)-1] >= end {
			continue
		}
		ends = append(ends, end)
	}
	return ends
}
