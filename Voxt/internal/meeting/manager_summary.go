package meeting

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// GenerateMeetingSummary loads segments and writes a summary on the meeting row.
func (m *Manager) GenerateMeetingSummary(ctx context.Context, meetingID string) error {
	m.mu.Lock()
	activeID := m.id
	m.mu.Unlock()
	if meetingID == "" {
		return fmt.Errorf("meeting: missing id")
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
	tpl := m.cfg.Settings().Prompt
	mainLang := "English"
	if m.cfg.MainLanguage != nil {
		if v := strings.TrimSpace(m.cfg.MainLanguage()); v != "" {
			mainLang = v
		}
	}
	if activeID == meetingID {
		m.mu.Lock()
		m.state = StateFinishing
		m.snap.State = StateFinishing
		m.snap.Stage = StageSummary
		s := m.snap
		m.mu.Unlock()
		m.notify(s)
	}
	sum, err := GenerateSummary(ctx, chat, tpl, mainLang, record)
	if err != nil {
		if activeID == meetingID {
			m.mu.Lock()
			m.state = StateDone
			m.snap.State = StateDone
			m.snap.Stage = ""
			m.mu.Unlock()
			m.notify(m.Current())
		}
		return err
	}
	blob, err := sum.EncodeJSON()
	if err != nil {
		return err
	}
	mt.Title = sum.Title
	mt.Summary = blob
	if err := m.cfg.Store.UpdateMeeting(ctx, mt); err != nil {
		return err
	}
	if activeID == meetingID {
		m.mu.Lock()
		m.state = StateDone
		m.snap.State = StateDone
		m.snap.Stage = ""
		m.mu.Unlock()
		m.notify(m.Current())
	}
	return nil
}

func speakerNames(raw string) map[string]string {
	out := map[string]string{}
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return out
	}
	_ = json.Unmarshal([]byte(raw), &out)
	return out
}

func storeSegmentToMeeting(row store.MeetingSegment) Segment {
	return Segment{
		Source:      segmentSourceToMeeting(row.Source),
		Speaker:     row.Speaker,
		Start:       time.Duration(row.StartMS) * time.Millisecond,
		End:         time.Duration(row.EndMS) * time.Millisecond,
		Text:        row.Text,
		Translation: row.Translation,
		Final:       row.Final,
		StoreIdx:    row.Idx,
	}
}

func segmentSourceToMeeting(src store.SegmentSource) Source {
	if src == store.SegmentSystem {
		return SourceSystem
	}
	return SourceMic
}
