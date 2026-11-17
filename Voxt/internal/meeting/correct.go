package meeting

// The transcript's correction (#4018): after the final pass the language
// model fixes what speech recognition got wrong, with the dictionary and
// the speech context dictation sends, before the summary reads the text.

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TranscriptContext is the user's context for correcting a transcript.
type TranscriptContext struct {
	// Terms are the dictionary's enabled terms, Notes its notes and Speech
	// the speech context (settings.Dictionary), as dictation sends them.
	Terms  []compose.Term
	Notes  string
	Speech string
}

// A correction request holds at most this many lines and runes of text.
const (
	correctBatchLines = 30
	correctBatchRunes = 3000
)

const correctPrompt = `You proofread a meeting transcript that speech recognition wrote. The user message is a JSON array of its lines: [{"i": number, "text": "..."}].

1. Fix what recognition got wrong, from the context of the meeting: words and characters that sound like the right ones, misheard names, product names and terms (write the glossary's terms as it gives them), and the punctuation.
2. Drop stutters ("I I I") and filler sounds (um, uh, 嗯, 呃). Keep everything else as it was said, in the language it was said in: do not rewrite, summarize, translate, merge, split, reorder or add anything.
3. Leave a word you cannot make out as it is.

Answer with only a JSON array, one item for each line, with its i: [{"i": number, "text": "the corrected line"}].`

// CorrectTranscript has the language model correct a stored meeting's
// lines in place: their times, speakers and order stay. A line the
// answer leaves out or empties stays as it was recognised.
func (m *Manager) CorrectTranscript(ctx context.Context, id string) error {
	mt, segs, err := m.cfg.Store.Meeting(ctx, id)
	if err != nil {
		return err
	}
	if m.cfg.LLM == nil {
		return fmt.Errorf("meeting: no language model")
	}
	chat, err := m.cfg.LLM(ctx, m.cfg.Settings().LLM)
	if err != nil {
		return err
	}
	var tc TranscriptContext
	if m.cfg.Context != nil {
		tc = m.cfg.Context(ctx)
	}
	for _, batch := range correctionBatches(segs) {
		fixed, err := correctBatch(ctx, chat, tc, batch)
		if err != nil {
			return err
		}
		for _, s := range batch {
			if t, ok := fixed[s.Idx]; ok && t != s.Text {
				if err := m.cfg.Store.UpdateSegmentText(ctx, mt.Seq, s.Idx, t); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// correctionBatches splits the lines with text into requests.
func correctionBatches(segs []store.MeetingSegment) [][]store.MeetingSegment {
	var out [][]store.MeetingSegment
	var cur []store.MeetingSegment
	runes := 0
	for _, s := range segs {
		if strings.TrimSpace(s.Text) == "" {
			continue
		}
		n := len([]rune(s.Text))
		if len(cur) > 0 && (len(cur) == correctBatchLines || runes+n > correctBatchRunes) {
			out = append(out, cur)
			cur, runes = nil, 0
		}
		cur = append(cur, s)
		runes += n
	}
	if len(cur) > 0 {
		out = append(out, cur)
	}
	return out
}

type correctLine struct {
	I    int    `json:"i"`
	Text string `json:"text"`
}

// correctBatch asks for one batch; it returns the corrected text by
// segment index, without the lines the answer has no text for.
func correctBatch(ctx context.Context, chat SummaryChat, tc TranscriptContext, batch []store.MeetingSegment) (map[int]string, error) {
	lines := make([]correctLine, len(batch))
	var text strings.Builder
	for i, s := range batch {
		lines[i] = correctLine{I: s.Idx, Text: s.Text}
		text.WriteString(s.Text + "\n")
	}
	user, err := json.Marshal(lines)
	if err != nil {
		return nil, err
	}
	zero := 0.0
	resp, err := chat.Chat(ctx, llm.Request{Temperature: &zero, Messages: []llm.Message{
		{Role: llm.System, Content: correctSystem(tc, text.String())},
		{Role: llm.User, Content: string(user)},
	}})
	if err != nil {
		return nil, err
	}
	return parseCorrection(resp.Text, batch), nil
}

// correctSystem is the system prompt: the rules, then the user's context.
func correctSystem(tc TranscriptContext, text string) string {
	parts := []string{correctPrompt}
	if s := strings.TrimSpace(tc.Speech); s != "" {
		parts = append(parts, "What the meeting is about, in the user's words:\n"+s)
	}
	if g := compose.Glossary(tc.Terms, text, compose.English); g != "" {
		parts = append(parts, "Glossary:\n"+g)
	}
	if n := strings.TrimSpace(tc.Notes); n != "" {
		parts = append(parts, "The user's dictionary notes:\n"+n)
	}
	return strings.Join(parts, "\n\n")
}

// parseCorrection reads the answer's array (bare or in a code fence);
// an answer that does not parse corrects nothing.
func parseCorrection(raw string, batch []store.MeetingSegment) map[int]string {
	start, end := strings.IndexByte(raw, '['), strings.LastIndexByte(raw, ']')
	out := map[int]string{}
	if start < 0 || end < start {
		return out
	}
	var lines []correctLine
	if json.Unmarshal([]byte(raw[start:end+1]), &lines) != nil {
		return out
	}
	asked := map[int]bool{}
	for _, s := range batch {
		asked[s.Idx] = true
	}
	for _, l := range lines {
		if t := strings.TrimSpace(l.Text); t != "" && asked[l.I] {
			out[l.I] = t
		}
	}
	return out
}
