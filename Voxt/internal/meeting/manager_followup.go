package meeting

import (
	"context"
	"fmt"
	"strings"
)

// AnswerMeetingFollowUp runs one follow-up turn and persists Q&A on the meeting row.
func (m *Manager) AnswerMeetingFollowUp(ctx context.Context, meetingID, question string) error {
	if meetingID == "" {
		return fmt.Errorf("meeting: missing id")
	}
	question = strings.TrimSpace(question)
	if question == "" {
		return fmt.Errorf("meeting: empty question")
	}
	mt, segs, err := m.cfg.Store.Meeting(ctx, meetingID)
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
	names := speakerNames(mt.Speakers)
	live := make([]Segment, 0, len(segs))
	for _, row := range segs {
		live = append(live, storeSegmentToMeeting(row))
	}
	record := TranscriptRecord(live, names)
	if strings.TrimSpace(record) == "" {
		return fmt.Errorf("meeting: no transcript")
	}
	history, err := ParseChatMessages(mt.Chat)
	if err != nil {
		return err
	}
	var sumPtr *Summary
	if s, ok := ParseStoredSummary(mt.Summary); ok {
		sumPtr = &s
	}
	mainLang := "English"
	if m.cfg.MainLanguage != nil {
		if v := strings.TrimSpace(m.cfg.MainLanguage()); v != "" {
			mainLang = v
		}
	}
	answer, err := AnswerFollowUp(ctx, chat, mainLang, record, sumPtr, history, question)
	if err != nil {
		return err
	}
	history = append(history,
		ChatMessage{Role: "user", Content: question},
		ChatMessage{Role: "assistant", Content: answer},
	)
	blob, err := EncodeChatMessages(history)
	if err != nil {
		return err
	}
	mt.Chat = blob
	return m.cfg.Store.UpdateMeeting(ctx, mt)
}
