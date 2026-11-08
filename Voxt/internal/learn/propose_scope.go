package learn

import (
	"regexp"
	"strings"
	"unicode"
)

// Voxt 1.x's rules for picking a text's line that holds the insertion:
// the field may also hold the app's own lines (a terminal's prompt and
// command output).

// scopedBaseline is the line of the baseline the insertion is in, without
// a terminal's prompt, and whether the insertion was found: at its
// expected rune position at (-1: unknown) first, else anywhere.
func scopedBaseline(inserted, baseline string, at int) (string, bool) {
	ins, found := insertedAt(baseline, inserted, at)
	if !found {
		return "", false
	}
	return lineScope(baseline, ins, strings.TrimSpace(inserted)), true
}

// scopedCurrent is the part of a later read that continues the baseline
// line: the whole read on one line, or its best matching line ("" when
// none is close, as after the field was cleared).
func scopedCurrent(inserted, baselineLine, current string) string {
	text := strings.TrimSpace(current)
	if !strings.Contains(text, "\n") {
		return stripPrompt(text)
	}
	line, _ := bestLine(inserted, baselineLine, text)
	return line
}

// finalScope is scopedCurrent for the judgement: a text without a matching
// line is compared whole.
func finalScope(inserted, baselineLine, final string) string {
	if !strings.Contains(final, "\n") {
		return stripPrompt(final)
	}
	if line, ok := bestLine(inserted, baselineLine, final); ok {
		return line
	}
	return stripPrompt(final)
}

// lineScope is the line of text that the range ins covers, without its
// prompt; when that is empty, the fallback.
func lineScope(text string, ins [2]int, fallback string) string {
	rs := []rune(text)
	start, end := lineBounds(rs, ins)
	line := stripPrompt(string(rs[start:end]))
	if line == "" {
		return fallback
	}
	return line
}

func lineBounds(rs []rune, ins [2]int) (int, int) {
	start, end := ins[0], ins[1]
	for start > 0 && rs[start-1] != '\n' {
		start--
	}
	for end < len(rs) && rs[end] != '\n' {
		end++
	}
	return start, end
}

// stripPrompt removes a terminal's "> ".
func stripPrompt(text string) string {
	line := strings.TrimSpace(text)
	if strings.HasPrefix(line, "> ") {
		line = line[2:]
	} else if line == ">" {
		line = ""
	}
	return strings.TrimSpace(line)
}

// insertedAt finds the inserted text at its expected position, else with
// findInserted.
func insertedAt(text, inserted string, at int) ([2]int, bool) {
	if at >= 0 {
		rs, needle := []rune(text), []rune(strings.TrimSpace(inserted))
		if end := at + len(needle); len(needle) > 0 && end <= len(rs) && foldedEqual(rs[at:end], needle) {
			return [2]int{at, end}, true
		}
	}
	return findInserted(text, inserted)
}

// findInserted finds the inserted text in text, ignoring case, else also
// ignoring whitespace (a field that wrapped the line). The range is runes.
func findInserted(text, inserted string) ([2]int, bool) {
	rs, needle := []rune(text), []rune(strings.TrimSpace(inserted))
	if len(needle) == 0 {
		return [2]int{}, false
	}
	for i := 0; i+len(needle) <= len(rs); i++ {
		if foldedEqual(rs[i:i+len(needle)], needle) {
			return [2]int{i, i + len(needle)}, true
		}
	}
	collapsed, starts := collapseSpace(rs)
	bare, _ := collapseSpace(needle)
	if len(bare) == 0 {
		return [2]int{}, false
	}
	for i := 0; i+len(bare) <= len(collapsed); i++ {
		if foldedEqual(collapsed[i:i+len(bare)], bare) {
			return [2]int{starts[i], starts[i+len(bare)-1] + 1}, true
		}
	}
	return [2]int{}, false
}

// collapseSpace drops the whitespace and says where each rune left was.
func collapseSpace(rs []rune) ([]rune, []int) {
	out, starts := make([]rune, 0, len(rs)), make([]int, 0, len(rs))
	for i, r := range rs {
		if !unicode.IsSpace(r) {
			out, starts = append(out, r), append(starts, i)
		}
	}
	return out, starts
}

func foldedEqual(a, b []rune) bool {
	for i := range a {
		if unicode.ToLower(a[i]) != unicode.ToLower(b[i]) {
			return false
		}
	}
	return true
}

const minLineScore = 6

var spaces = regexp.MustCompile(`\s+`)

// bestLine is the line of text most like the inserted text or the baseline
// line, if any is close enough.
func bestLine(inserted, baselineLine, text string) (string, bool) {
	best, bestScore := "", -1<<62
	for _, raw := range strings.Split(text, "\n") {
		line := stripPrompt(raw)
		if line == "" {
			continue
		}
		score := max(lineSimilarity(inserted, line), lineSimilarity(baselineLine, line))
		if isNoiseLine(line) {
			score -= max(minLineScore, len([]rune(line))/4)
		}
		if score > bestScore {
			best, bestScore = line, score
		}
	}
	threshold := max(minLineScore, max(len([]rune(normalizeLine(inserted))), len([]rune(normalizeLine(baselineLine))))/4)
	if best == "" || bestScore < threshold {
		return "", false
	}
	return best, true
}

func normalizeLine(line string) string {
	return strings.TrimSpace(spaces.ReplaceAllString(strings.ToLower(stripPrompt(line)), " "))
}

// lineSimilarity scores how alike two lines are: the shared prefix and
// suffix, twice the longest common run, and the shorter one when one
// holds the other.
func lineSimilarity(a, b string) int {
	x, y := []rune(normalizeLine(a)), []rune(normalizeLine(b))
	if len(x) == 0 || len(y) == 0 {
		return 0
	}
	prefix := 0
	for prefix < min(len(x), len(y)) && x[prefix] == y[prefix] {
		prefix++
	}
	suffix := 0
	for suffix < min(len(x), len(y))-prefix && x[len(x)-1-suffix] == y[len(y)-1-suffix] {
		suffix++
	}
	score := prefix + suffix + longestCommonRun(x, y)*2
	if strings.Contains(string(x), string(y)) || strings.Contains(string(y), string(x)) {
		score += min(len(x), len(y))
	}
	return score
}

func longestCommonRun(x, y []rune) int {
	best := 0
	previous := make([]int, len(y)+1)
	for i := range x {
		current := make([]int, len(y)+1)
		for j := range y {
			if x[i] == y[j] {
				current[j+1] = previous[j] + 1
				best = max(best, current[j+1])
			}
		}
		previous = current
	}
	return best
}

// isNoiseLine is a shell's own output: 1.x penalises these lines so that
// the command itself, not its echo, is compared.
func isNoiseLine(line string) bool {
	for _, prefix := range []string{"zsh:", "bash:", "fish:", "~/", "/"} {
		if strings.HasPrefix(line, prefix) {
			return true
		}
	}
	return strings.Contains(line, "command not found:") || strings.Contains(line, "no such file or directory")
}
