package session

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// memNotes is a store.Notes in memory; err fails every Put.
type memNotes struct {
	mu    sync.Mutex
	notes []store.Note
	err   error
}

func (n *memNotes) Put(_ context.Context, x store.Note) error {
	n.mu.Lock()
	defer n.mu.Unlock()
	if n.err != nil {
		return n.err
	}
	n.notes = append(n.notes, x)
	return nil
}

func (n *memNotes) List(context.Context) ([]store.Note, error) {
	n.mu.Lock()
	defer n.mu.Unlock()
	return append([]store.Note(nil), n.notes...), nil
}

func (n *memNotes) Delete(context.Context, string) error { return nil }

// TestNote: a note session saves what was said as a note, inserts
// nothing and keeps no history entry (nor its recording); when saving
// fails, the card shows the text.
func TestNote(t *testing.T) {
	for _, tc := range []struct {
		name  string
		notes *memNotes // nil: no Config.Notes
		saved bool
	}{
		{"saved", &memNotes{}, true},
		{"save fails", &memNotes{err: errors.New("disk full")}, false},
		{"no notes store", nil, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newHarness(t, &fakeFile{text: " buy milk "}, nil, func(c *Config, s *settings.Settings) {
				s.Privacy.KeepAudio = true
				if tc.notes != nil {
					c.Notes = tc.notes
				}
			})
			id, err := h.m.Start(StartRequest{Kind: store.KindTranscribe, Target: slack, Note: true})
			if err != nil {
				t.Fatal(err)
			}
			h.speak(id, 5)
			end := h.rec.wait(t)
			if end.State != Done || end.Result != "buy milk" || !end.Note || end.Inserted || end.Recording ||
				len(h.text.pastes()) != 0 || len(h.hist.all()) != 0 || h.m.Last() != "" {
				t.Errorf("end %+v, %d pastes, %d entries, last %q", end, len(h.text.pastes()), len(h.hist.all()), h.m.Last())
			}
			if tc.saved {
				got, _ := tc.notes.List(context.Background())
				if len(got) != 1 || got[0].ID != id || got[0].Text != "buy milk" || !got[0].CreatedAt.Equal(end.StartedAt) {
					t.Errorf("notes %+v", got)
				}
				if end.Card || end.Problem != nil {
					t.Errorf("saved, but card %v problem %+v", end.Card, end.Problem)
				}
				return
			}
			if !end.Card || end.Problem == nil || end.Problem.Code != ProblemInternal || !end.Problem.Degraded {
				t.Errorf("not saved: card %v problem %+v", end.Card, end.Problem)
			}
		})
	}
}
