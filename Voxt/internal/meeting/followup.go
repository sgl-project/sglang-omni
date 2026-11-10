package meeting

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"unicode"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
)

const (
	// FollowUpWindowMax is the transcript context size for meeting follow-up (Voxt 1.x).
	FollowUpWindowMax  = 6000
	followUpMaxWindows = 3
)

// DefaultFollowUpPrompt is the meeting follow-up system template (Voxt 1.x transcript-follow-up.txt).
const DefaultFollowUpPrompt = `You are Voxt's transcript follow-up assistant.

Answer the user's follow-up question about the transcript.

Rules:
1. Use the transcript as the primary source of truth.
2. Use the summary and previous chat only as compressed context; if they conflict with the transcript, trust the transcript.
3. Reply in the user's main language.
4. Answer directly and concisely.
5. Do not invent facts. If the transcript does not contain the answer, clearly say that the transcript does not provide enough information.
6. Return plain text only. Do not use JSON or markdown code fences.

User main language: {{USER_MAIN_LANGUAGE}}

Current summary:
{{CURRENT_SUMMARY}}

Transcript:
{{TRANSCRIPT}}`

// ChatMessage is one stored follow-up turn in meetings.chat (Voxt 1.x TranscriptSummaryChatMessage).
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ParseChatMessages decodes meetings.chat JSON.
func ParseChatMessages(raw string) ([]ChatMessage, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" || raw == "[]" {
		return nil, nil
	}
	var out []ChatMessage
	if err := json.Unmarshal([]byte(raw), &out); err != nil {
		return nil, err
	}
	return out, nil
}

// EncodeChatMessages stores follow-up turns on the meeting row.
func EncodeChatMessages(msgs []ChatMessage) (string, error) {
	if len(msgs) == 0 {
		return "[]", nil
	}
	b, err := json.Marshal(msgs)
	return string(b), err
}

func trimFollowUpContext(text string) string {
	text = strings.TrimSpace(text)
	if len([]rune(text)) <= FollowUpWindowMax {
		return text
	}
	return string([]rune(text)[:FollowUpWindowMax])
}

type followUpScoredWindow struct {
	idx    int
	score  int
	window string
}

// RelevantFollowUpContext picks transcript windows for a question (Voxt 1.x MeetingSummaryContextPlanning).
func RelevantFollowUpContext(transcript, question string) string {
	windows := SummaryWindows(transcript, FollowUpWindowMax)
	if len(windows) == 0 {
		return ""
	}
	joined := strings.Join(windows, "\n\n")
	if len(windows) <= followUpMaxWindows && len([]rune(joined)) <= FollowUpWindowMax {
		return joined
	}
	terms := followUpSearchTerms(question)
	if len(terms) == 0 || followUpGlobalQuestion(question) {
		parts := []string{windows[0], windows[len(windows)-1]}
		return trimFollowUpContext(strings.Join(parts, "\n\n"))
	}
	ranked := make([]followUpScoredWindow, len(windows))
	for i, w := range windows {
		lower := strings.ToLower(w)
		score := 0
		for term := range terms {
			score += strings.Count(lower, term)
		}
		ranked[i] = followUpScoredWindow{idx: i, score: score, window: w}
	}
	for i := 0; i < len(ranked); i++ {
		for j := i + 1; j < len(ranked); j++ {
			a, b := ranked[i], ranked[j]
			swap := false
			if a.score != b.score {
				swap = a.score < b.score
			} else {
				swap = a.idx > b.idx
			}
			if swap {
				ranked[i], ranked[j] = ranked[j], ranked[i]
			}
		}
	}
	limit := followUpMaxWindows
	if limit < 1 {
		limit = 1
	}
	if limit > len(ranked) {
		limit = len(ranked)
	}
	selected := ranked[:limit]
	return joinFollowUpSelectedWindows(selected, FollowUpWindowMax)
}

// joinFollowUpSelectedWindows merges ranked windows under max runes, preferring high-score text when over budget.
func joinFollowUpSelectedWindows(selected []followUpScoredWindow, max int) string {
	if len(selected) == 0 {
		return ""
	}
	order := append([]followUpScoredWindow(nil), selected...)
	for i := 0; i < len(order); i++ {
		for j := i + 1; j < len(order); j++ {
			a, b := order[i], order[j]
			swap := false
			if a.score != b.score {
				swap = a.score < b.score
			} else if a.idx != b.idx {
				swap = a.idx > b.idx
			}
			if swap {
				order[i], order[j] = order[j], order[i]
			}
		}
	}
	var parts []string
	total := 0
	for _, s := range order {
		w := s.window
		runes := []rune(w)
		sep := 0
		if len(parts) > 0 {
			sep = 2
		}
		if total+sep+len(runes) <= max {
			parts = append(parts, w)
			total += sep + len(runes)
			continue
		}
		remaining := max - total - sep
		if remaining <= 0 {
			break
		}
		if s.score > 0 || len(parts) == 0 {
			parts = append(parts, string(runes[:remaining]))
		}
		break
	}
	return strings.TrimSpace(strings.Join(parts, "\n\n"))
}

func followUpSearchTerms(text string) map[string]struct{} {
	lower := strings.ToLower(text)
	terms := map[string]struct{}{}
	var word []rune
	flush := func() {
		if len(word) >= 2 {
			terms[string(word)] = struct{}{}
		}
		word = word[:0]
	}
	for _, r := range lower {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			word = append(word, r)
			continue
		}
		flush()
	}
	flush()
	var cjk []rune
	for _, r := range []rune(lower) {
		if r >= 0x3400 && r <= 0x9FFF {
			cjk = append(cjk, r)
		}
	}
	if len(cjk) >= 2 {
		for i := 0; i < len(cjk)-1; i++ {
			terms[string(cjk[i:i+2])] = struct{}{}
		}
	}
	return terms
}

func followUpGlobalQuestion(question string) bool {
	normalized := strings.ToLower(strings.TrimSpace(question))
	for _, hint := range []string{
		"overall", "entire", "whole meeting", "summary",
		"总结", "整体", "全程", "整个会议",
	} {
		if strings.Contains(normalized, hint) {
			return true
		}
	}
	return false
}

// RenderFollowUpSystem fills the follow-up system prompt.
func RenderFollowUpSystem(mainLanguage string, summary *Summary, transcriptContext string) string {
	out := DefaultFollowUpPrompt
	lang := strings.TrimSpace(mainLanguage)
	if lang == "" {
		lang = "English"
	}
	out = strings.ReplaceAll(out, asr.VarUserMainLanguage, lang)
	out = strings.ReplaceAll(out, "{{CURRENT_SUMMARY}}", formatFollowUpSummary(summary))
	out = strings.ReplaceAll(out, "{{TRANSCRIPT}}", strings.TrimSpace(transcriptContext))
	return strings.TrimSpace(out)
}

func formatFollowUpSummary(summary *Summary) string {
	if summary == nil || (strings.TrimSpace(summary.Body) == "" && len(summary.Todos) == 0 && strings.TrimSpace(summary.Title) == "") {
		return "No generated summary is available yet."
	}
	var b strings.Builder
	fmt.Fprintf(&b, "Title: %s\nBody:\n%s\n\nTODO:\n", summary.Title, summary.Body)
	if len(summary.Todos) == 0 {
		b.WriteString("None")
	} else {
		for i, todo := range summary.Todos {
			if i > 0 {
				b.WriteByte('\n')
			}
			fmt.Fprintf(&b, "%d. %s", i+1, todo)
		}
	}
	return b.String()
}

// BuildFollowUpRequest builds a multi-turn chat request with history in order.
func BuildFollowUpRequest(mainLanguage string, summary *Summary, transcriptContext string, history []ChatMessage, question string) llm.Request {
	msgs := []llm.Message{{Role: llm.System, Content: RenderFollowUpSystem(mainLanguage, summary, transcriptContext)}}
	for _, h := range history {
		role := llm.User
		if h.Role == "assistant" {
			role = llm.Assistant
		}
		msgs = append(msgs, llm.Message{Role: role, Content: h.Content})
	}
	msgs = append(msgs, llm.Message{Role: llm.User, Content: strings.TrimSpace(question)})
	return llm.Request{Messages: msgs}
}

// AnswerFollowUp sends question against transcript context and returns the assistant reply text.
func AnswerFollowUp(ctx context.Context, chat SummaryChat, mainLanguage, transcript string, summary *Summary, history []ChatMessage, question string) (string, error) {
	if chat == nil {
		return "", fmt.Errorf("meeting: no language model")
	}
	transcript = strings.TrimSpace(transcript)
	question = strings.TrimSpace(question)
	if transcript == "" {
		return "", fmt.Errorf("meeting: no transcript")
	}
	if question == "" {
		return "", fmt.Errorf("meeting: empty question")
	}
	contextText := RelevantFollowUpContext(transcript, question)
	req := BuildFollowUpRequest(mainLanguage, summary, contextText, history, question)
	resp, err := chat.Chat(ctx, req)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(resp.Text), nil
}
