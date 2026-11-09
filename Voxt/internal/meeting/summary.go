package meeting

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
)

const (
	VarTranscriptRecord = "{{TRANSCRIPT_RECORD}}"
	summaryWindowMax    = 12000
)

// DefaultSummaryPrompt is the built-in meeting summary template (Voxt 1.x en-transcript-summary.txt).
const DefaultSummaryPrompt = `Your task is to generate a clear, credible, and concise transcript summary based on the provided transcript and return one valid JSON object. Please strictly follow the requirements below:

User's main language:
{{USER_MAIN_LANGUAGE}}

Transcript:
` + VarTranscriptRecord + `

When generating the summary, please adhere to the following specifications:
1. Regardless of the language used in the transcript, the final summary must be output in the user's main language.
2. The main body of the summary should be within 1200 characters (for Chinese) or maintain equivalent conciseness (for non-Chinese languages), prioritizing efficiency over mere character count.
3. Prioritize extracting the following content:
   - Context and background
   - Key discussion points and notable viewpoints
   - Decisions reached or clear conclusions
   - Risks, blockers, and unresolved issues
   - Follow-up items that still need attention
4. If there are explicit or strongly implied to-do tasks in the transcript, include them in the "todo_list" field of the JSON.
5. Strictly base on the content of the transcript; do not fabricate any unmentioned facts. If information is insufficient, use conservative expressions.
6. The title should be brief and accurately summarize the transcript theme.
7. Translation rules:
   - Non-user-main-language content in the transcript must be translated into the user's main language.
   - Proper nouns, product names, URLs, and code snippets can be kept in their original form.
8. The content field is plain text only. Do not use markdown, bullets, tables, code fences, or HTML. If paragraph breaks are needed inside the JSON string, use normal JSON newline escapes. Do not double-escape line breaks.

The output must be JSON only, with no surrounding markdown or explanation, and must conform exactly to this JSON Schema:
{
  "type": "object",
  "additionalProperties": false,
  "required": ["transcript_summary", "todo_list"],
  "properties": {
    "transcript_summary": {
      "type": "object",
      "additionalProperties": false,
      "required": ["title", "content"],
      "properties": {
        "title": {
          "type": "string",
          "minLength": 1,
          "maxLength": 80,
          "description": "Brief transcript theme"
        },
        "content": {
          "type": "string",
          "minLength": 1,
          "description": "Plain-text summary covering context, key points, decisions, risks or blockers, and unresolved issues. Use JSON newline escapes only when paragraph breaks improve readability."
        }
      }
    },
    "todo_list": {
      "type": "array",
      "items": {
        "type": "string",
        "minLength": 1,
        "description": "One task with owner and deadline when mentioned"
      }
    }
  }
}

Note: If there are no to-do tasks, the "todo_list" field must be an empty array. Ensure the JSON is properly formatted, with no trailing commas, and the content accurately reflects the transcript in fluent, natural language conforming to the user's main language expression habits.`

// Summary is a generated meeting summary.
type Summary struct {
	Title       string
	Body        string
	Todos       []string
	GeneratedAt time.Time
}

// EncodeJSON stores a summary on the meetings table.
func (s Summary) EncodeJSON() (string, error) {
	var doc struct {
		Title       string   `json:"title"`
		Body        string   `json:"body"`
		Todos       []string `json:"todos"`
		GeneratedAt int64    `json:"generated_at"`
	}
	doc.Title = s.Title
	doc.Body = s.Body
	doc.Todos = s.Todos
	if s.GeneratedAt.IsZero() {
		doc.GeneratedAt = time.Now().UnixMilli()
	} else {
		doc.GeneratedAt = s.GeneratedAt.UnixMilli()
	}
	b, err := json.Marshal(doc)
	return string(b), err
}

// ParseStoredSummary decodes the meetings.summary column.
func ParseStoredSummary(raw string) (Summary, bool) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return Summary{}, false
	}
	var doc struct {
		Title       string   `json:"title"`
		Body        string   `json:"body"`
		Todos       []string `json:"todos"`
		GeneratedAt int64    `json:"generated_at"`
	}
	if json.Unmarshal([]byte(raw), &doc) != nil {
		return Summary{}, false
	}
	s := Summary{Title: doc.Title, Body: doc.Body, Todos: doc.Todos}
	if doc.GeneratedAt > 0 {
		s.GeneratedAt = time.UnixMilli(doc.GeneratedAt)
	}
	return s, true
}

// SpeakerLabel is how a segment speaker appears in transcripts and exports.
func SpeakerLabel(speaker string, names map[string]string) string {
	if speaker == "me" {
		return "Me"
	}
	if n := strings.TrimSpace(names[speaker]); n != "" {
		return n
	}
	switch speaker {
	case "":
		return "Them"
	default:
		if len(speaker) == 2 && speaker[0] == 's' && speaker[1] >= '1' && speaker[1] <= '4' {
			return fmt.Sprintf("Speaker %c", speaker[1])
		}
		return speaker
	}
}

// TranscriptRecord formats segments for summary prompts (Voxt 1.x llmInputText).
func TranscriptRecord(segs []Segment, names map[string]string) string {
	lines := make([]string, 0, len(segs))
	for _, seg := range segs {
		text := strings.TrimSpace(seg.Text)
		if text == "" {
			continue
		}
		lines = append(lines, fmt.Sprintf("%s：%s", SpeakerLabel(seg.Speaker, names), text))
	}
	return strings.TrimSpace(strings.Join(lines, "\n"))
}

// RenderSummaryPrompt fills template variables for a summary request.
func RenderSummaryPrompt(template, mainLanguage, transcriptRecord string) string {
	if strings.TrimSpace(template) == "" {
		template = DefaultSummaryPrompt
	}
	out := strings.TrimSpace(template)
	if !strings.Contains(out, asr.VarUserMainLanguage) {
		out += "\n\nUser main language: " + strings.TrimSpace(mainLanguage)
	} else {
		out = strings.ReplaceAll(out, asr.VarUserMainLanguage, strings.TrimSpace(mainLanguage))
	}
	if strings.Contains(out, VarTranscriptRecord) {
		out = strings.ReplaceAll(out, VarTranscriptRecord, transcriptRecord)
	} else {
		out += "\n\nTranscript:\n" + transcriptRecord
	}
	return strings.TrimSpace(out)
}

// SummaryWindows splits a long transcript for hierarchical summary (default 12000 runes per window).
func SummaryWindows(text string, maxRunes int) []string {
	if maxRunes <= 0 {
		maxRunes = summaryWindowMax
	}
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}
	paragraphs := strings.Split(text, "\n")
	var chunks []string
	for _, p := range paragraphs {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		if len([]rune(p)) > maxRunes {
			chunks = append(chunks, splitLongText(p, maxRunes)...)
			continue
		}
		chunks = append(chunks, p)
	}
	if len(chunks) == 0 {
		return splitLongText(text, maxRunes)
	}
	var out []string
	var cur string
	for _, p := range chunks {
		candidate := p
		if cur != "" {
			candidate = cur + "\n" + p
		}
		if len([]rune(candidate)) > maxRunes {
			if cur != "" {
				out = append(out, cur)
			}
			cur = p
			continue
		}
		cur = candidate
	}
	if cur != "" {
		out = append(out, cur)
	}
	return out
}

func splitLongText(text string, max int) []string {
	runes := []rune(text)
	if len(runes) <= max {
		return []string{text}
	}
	var out []string
	for len(runes) > max {
		cut := max
		best := -1
		scanFrom := cut
		if scanFrom > max/2 {
			scanFrom = max / 2
		}
		for i := cut; i > scanFrom; i-- {
			if isSentenceBreak(runes[i-1]) {
				best = i
				break
			}
		}
		if best > 0 {
			cut = best
		}
		out = append(out, strings.TrimSpace(string(runes[:cut])))
		runes = []rune(strings.TrimSpace(string(runes[cut:])))
	}
	if len(runes) > 0 {
		out = append(out, string(runes))
	}
	return out
}

func isSentenceBreak(r rune) bool {
	switch r {
	case '.', '!', '?', '。', '！', '？', '\n':
		return true
	default:
		return false
	}
}

// ParseSummary decodes model output (JSON, fenced JSON, or plain text body).
func ParseSummary(raw string) (Summary, error) {
	raw = normalizeSummaryPayload(raw)
	if raw == "" {
		return Summary{}, fmt.Errorf("meeting: empty summary")
	}
	var payload summaryPayload
	if json.Unmarshal([]byte(raw), &payload) == nil {
		if s := summaryFromPayload(payload); s.Body != "" || len(s.Todos) > 0 {
			if s.Title == "" {
				s.Title = "Transcript Summary"
			}
			if s.GeneratedAt.IsZero() {
				s.GeneratedAt = time.Now()
			}
			return s, nil
		}
	}
	body := strings.TrimSpace(raw)
	if body == "" {
		return Summary{}, fmt.Errorf("meeting: summary not parseable")
	}
	return Summary{Title: "Transcript Summary", Body: body, GeneratedAt: time.Now()}, nil
}

type summaryPayload struct {
	TranscriptSummary *struct {
		Title   string `json:"title"`
		Body    string `json:"body"`
		Content string `json:"content"`
	} `json:"transcript_summary"`
	Title     string   `json:"title"`
	Body      string   `json:"body"`
	Content   string   `json:"content"`
	TodoList  []string `json:"todo_list"`
	TodoItems []string `json:"todoItems"`
	Todos     []string `json:"todos"`
}

func summaryFromPayload(p summaryPayload) Summary {
	block := p.TranscriptSummary
	title := pickText(blockTitle(block), p.Title)
	body := pickText(blockBody(block), p.Body, p.Content)
	todos := p.TodoList
	if len(todos) == 0 {
		todos = p.TodoItems
	}
	if len(todos) == 0 {
		todos = p.Todos
	}
	return Summary{Title: title, Body: body, Todos: cleanTodos(todos)}
}

func blockTitle(b *struct {
	Title   string `json:"title"`
	Body    string `json:"body"`
	Content string `json:"content"`
}) string {
	if b == nil {
		return ""
	}
	return strings.TrimSpace(b.Title)
}

func blockBody(b *struct {
	Title   string `json:"title"`
	Body    string `json:"body"`
	Content string `json:"content"`
}) string {
	if b == nil {
		return ""
	}
	return pickText(b.Content, b.Body)
}

func pickText(parts ...string) string {
	for _, p := range parts {
		if s := strings.TrimSpace(p); s != "" {
			return s
		}
	}
	return ""
}

func cleanTodos(in []string) []string {
	var out []string
	for _, t := range in {
		if s := strings.TrimSpace(t); s != "" {
			out = append(out, s)
		}
	}
	return out
}

func normalizeSummaryPayload(text string) string {
	t := strings.TrimSpace(text)
	if strings.HasPrefix(t, "```") {
		lines := strings.Split(t, "\n")
		if len(lines) >= 2 {
			lines = lines[1:]
			if strings.TrimSpace(lines[len(lines)-1]) == "```" {
				lines = lines[:len(lines)-1]
			}
			t = strings.TrimSpace(strings.Join(lines, "\n"))
		}
	}
	if j := extractJSONObject(t); j != "" {
		return j
	}
	return t
}

func extractJSONObject(text string) string {
	start := strings.IndexByte(text, '{')
	if start < 0 {
		return ""
	}
	depth := 0
	inString := false
	escaped := false
	for i := start; i < len(text); i++ {
		c := text[i]
		if inString {
			if escaped {
				escaped = false
				continue
			}
			if c == '\\' {
				escaped = true
				continue
			}
			if c == '"' {
				inString = false
			}
			continue
		}
		if c == '"' {
			inString = true
			continue
		}
		if c == '{' {
			depth++
		} else if c == '}' {
			depth--
			if depth == 0 {
				return text[start : i+1]
			}
		}
	}
	return ""
}

// SummaryChat is the minimal LLM surface summary generation needs.
type SummaryChat interface {
	Chat(ctx context.Context, req llm.Request) (llm.Response, error)
}

// GenerateSummary runs a hierarchical summary over transcriptRecord (Voxt 1.x flow).
func GenerateSummary(ctx context.Context, chat SummaryChat, template, mainLanguage, transcriptRecord string) (Summary, error) {
	if chat == nil {
		return Summary{}, fmt.Errorf("meeting: no language model")
	}
	transcriptRecord = strings.TrimSpace(transcriptRecord)
	if transcriptRecord == "" {
		return Summary{}, fmt.Errorf("meeting: empty transcript")
	}
	units := SummaryWindows(transcriptRecord, summaryWindowMax)
	if len(units) == 0 {
		return Summary{}, fmt.Errorf("meeting: empty transcript")
	}
	for {
		outputs, err := summarizeWindowGroup(ctx, chat, template, mainLanguage, units)
		if err != nil {
			return Summary{}, err
		}
		if len(outputs) == 1 {
			return ParseSummary(outputs[0])
		}
		units = make([]string, len(outputs))
		for i, o := range outputs {
			r := []rune(strings.TrimSpace(o))
			if len(r) > 4000 {
				r = r[:4000]
			}
			units[i] = string(r)
		}
	}
}

func summarizeWindowGroup(ctx context.Context, chat SummaryChat, template, mainLanguage string, units []string) ([]string, error) {
	groups := groupSummaryUnits(units, summaryWindowMax)
	out := make([]string, 0, len(groups))
	for _, group := range groups {
		chunk := strings.Join(group, "\n\n")
		prompt := RenderSummaryPrompt(template, mainLanguage, chunk)
		resp, err := chat.Chat(ctx, llm.Request{Messages: []llm.Message{{Role: llm.User, Content: prompt}}})
		if err != nil {
			return nil, err
		}
		out = append(out, strings.TrimSpace(resp.Text))
	}
	return out, nil
}

func groupSummaryUnits(units []string, limit int) [][]string {
	var groups [][]string
	var cur []string
	curCount := 0
	for _, u := range units {
		bounded := u
		if len([]rune(bounded)) > limit {
			bounded = string([]rune(bounded)[:limit])
		}
		added := len([]rune(bounded))
		if len(cur) > 0 {
			added += 2
		}
		if len(cur) > 0 && curCount+added > limit {
			groups = append(groups, cur)
			cur = nil
			curCount = 0
		}
		cur = append(cur, bounded)
		curCount += added
	}
	if len(cur) > 0 {
		groups = append(groups, cur)
	}
	return groups
}
