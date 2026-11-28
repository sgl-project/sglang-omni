package meeting

import (
	"encoding/json"
	"maps"
	"time"
)

const speakerFresh = 1500 * time.Millisecond

type speakerSeen struct {
	at time.Duration
	id string
}

type speakerTracker struct {
	seen      []speakerSeen
	names     map[string]string
	ambiguous map[string]bool
}

// ObserveSpeakers accepts UI evidence only for the recording it was requested
// for. It never assigns a stale response to a resumed track or another meeting.
func (m *Manager) ObserveSpeakers(id string, observation SpeakerObservation) {
	m.mu.Lock()
	if m.id != id || m.state != StateRecording || m.snap.State != StateRecording || m.timeline == nil {
		m.mu.Unlock()
		return
	}
	at := m.timeline.Elapsed()
	if observation.At != nil {
		at = min(max(*observation.At, 0), at)
	}
	if m.speakers.names == nil {
		m.speakers.names = map[string]string{}
	}
	if name := cleanSpeakerName(observation.Self); name != "" {
		m.speakers.names["me"] = name
	}
	name := ""
	if len(observation.Active) == 1 {
		name = cleanSpeakerName(observation.Active[0])
	}
	if name == m.speakers.names["me"] {
		name = ""
	}
	m.speakers.seen = append(m.speakers.seen, speakerSeen{at: at, id: name})
	// Live recognition commits within seconds. Keep a bounded observation tail.
	if len(m.speakers.seen) > 1200 {
		m.speakers.seen = append([]speakerSeen(nil), m.speakers.seen[len(m.speakers.seen)-1200:]...)
	}
	for _, segment := range m.snap.Segments {
		m.speakers.nameSegment(&segment)
	}
	m.snap.Speakers = maps.Clone(m.speakers.names)
	// Screens can name an acoustic identity, but a hidden window must
	// never clear it or substitute a visual identity for an audio one.
	snapshot := m.snap
	m.mu.Unlock()
	m.notify(snapshot)
}

// forRange needs consistent evidence over most of a sentence; a last-known
// active name must not leak across a switch, pause, hidden window or failed read.
func (s *speakerTracker) forRange(start, end time.Duration) string {
	if end <= start {
		return ""
	}
	totals := map[string]time.Duration{}
	for i, seen := range s.seen {
		until := seen.at + speakerFresh
		if i+1 < len(s.seen) {
			until = min(until, s.seen[i+1].at)
		}
		if seen.id != "" && until > start && seen.at < end {
			totals[seen.id] += max(min(until, end)-max(seen.at, start), 0)
		}
	}
	winner := ""
	for id, duration := range totals {
		if duration*5 >= (end-start)*4 {
			winner = id
			break
		}
	}
	return winner
}

func speakerNamesJSON(names map[string]string) string {
	if len(names) == 0 {
		return ""
	}
	b, _ := json.Marshal(names) // a string-to-string map is always encodable
	return string(b)
}

func (s *speakerTracker) nameSegment(segment *Segment) {
	name := s.forRange(segment.Start, segment.End)
	if name == "" || segment.Speaker == "" {
		return
	}
	if s.ambiguous[segment.Speaker] {
		return
	}
	if s.names == nil {
		s.names = map[string]string{}
	}
	// One visible name cannot confidently identify two different acoustic voices.
	for id, known := range s.names {
		if id != segment.Speaker && known == name {
			return
		}
	}
	if before := s.names[segment.Speaker]; before != "" && before != name {
		// Conflicting visible names may expose a merged acoustic voice.
		// Keep its number and colour, but do not call both people one name.
		delete(s.names, segment.Speaker)
		if s.ambiguous == nil {
			s.ambiguous = map[string]bool{}
		}
		s.ambiguous[segment.Speaker] = true
		return
	}
	s.names[segment.Speaker] = name
}
