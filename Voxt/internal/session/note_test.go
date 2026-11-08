package session

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
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

// TestNoteKeepsItsRecording: a note whose recognition failed, or that was
// cancelled, keeps its history entry and recording for Retry, as a
// dictation does. A note saved from the live text of a failed Final keeps
// them too, since the recognition failed.
func TestNoteKeepsItsRecording(t *testing.T) {
	for _, tc := range []struct {
		name   string
		prov   asr.Provider
		cancel bool
		ended  State
		status store.Status
		saved  bool
	}{
		{"recognition fails", &fakeFile{err: &asr.Error{Kind: asr.KindServer}}, false, Failed, store.StatusFailed, false},
		{"cancelled while recognising", &fakeFile{hang: true}, true, Cancelled, store.StatusCancelled, false},
		{"the Final fails, the live text saved", &fakeStreamer{finalErr: &asr.Error{Kind: asr.KindServer}}, false, Done, store.StatusDone, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			notes := &memNotes{}
			h := newHarness(t, tc.prov, nil, func(c *Config, _ *settings.Settings) { c.Notes = notes })
			id, err := h.m.Start(StartRequest{Kind: store.KindTranscribe, Target: slack, Note: true})
			if err != nil {
				t.Fatal(err)
			}
			h.speak(id, 5)
			if tc.cancel {
				h.rec.waitFor(t, "recognising", func(s Snapshot) bool { return s.State == Recognizing })
				h.m.Cancel(id)
			}
			end := h.rec.wait(t)
			if end.State != tc.ended || !end.Recording || !end.Note {
				t.Fatalf("end %s, recording %v", end.State, end.Recording)
			}
			es := h.hist.all()
			if len(es) != 1 || es[0].Status != tc.status || es[0].Audio != id+".wav" || !writeFileForTest(t, h.hist.dir, id+".wav") {
				t.Fatalf("entries %+v", es)
			}
			wantNotes := 0
			if tc.saved {
				wantNotes = 1
			}
			if got, _ := notes.List(context.Background()); len(got) != wantNotes {
				t.Errorf("notes %+v", got)
			}
		})
	}
}
