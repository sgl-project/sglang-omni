package meeting

import (
	"context"
	"maps"
	"slices"
	"sort"
	"strconv"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/speaker"
)

type acousticSpeakers struct {
	frames []speaker.Frame
	slots  map[int]string
	until  time.Duration
}

func voiceSlot(scores [8]float32) int {
	best, next := 0, float32(0)
	for i := 1; i < len(scores); i++ {
		if scores[i] > scores[best] {
			next = scores[best]
			best = i
		} else {
			next = max(next, scores[i])
		}
	}
	// Do not attach a single person's colour to overlapping or uncertain speech.
	if scores[best] < 0.7 || next >= 0.35 {
		return -1
	}
	return best
}

func (v *acousticSpeakers) add(frames []speaker.Frame) {
	if v.slots == nil {
		v.slots = map[int]string{}
	}
	for _, f := range frames {
		if f.End <= f.Start || f.Start < v.until {
			continue
		}
		v.frames = append(v.frames, f)
		v.until = f.End
		if slot := voiceSlot(f.Scores); slot >= 0 && v.slots[slot] == "" {
			v.slots[slot] = "s" + strconv.Itoa(len(v.slots)+1)
		}
	}
	// Trim in batches, not by copying a ten-minute history every update.
	if len(v.frames) == 0 || v.frames[0].End >= v.until-11*time.Minute {
		return
	}
	first := 0
	for first < len(v.frames) && v.frames[first].End < v.until-10*time.Minute {
		first++
	}
	if first > 0 {
		v.frames = slices.Clone(v.frames[first:])
	}
}

func (v *acousticSpeakers) forRange(start, end time.Duration) string {
	if end <= start || end > v.until {
		return ""
	}
	totals := map[int]time.Duration{}
	var voiced time.Duration
	first := sort.Search(len(v.frames), func(i int) bool { return v.frames[i].End > start })
	for _, f := range v.frames[first:] {
		if f.Start >= end {
			break
		}
		dur := min(f.End, end) - max(f.Start, start)
		if dur <= 0 {
			continue
		}
		peak := slices.Max(f.Scores[:])
		if peak < 0.5 {
			continue
		} // silence is not a competing identity
		voiced += dur
		if slot := voiceSlot(f.Scores); slot >= 0 {
			totals[slot] += dur
		}
	}
	if voiced < 240*time.Millisecond {
		return ""
	}
	for slot, duration := range totals {
		if duration*5 >= voiced*4 {
			return v.slots[slot]
		}
	}
	return ""
}

// segmentSpeakerWriter changes only the speaker column; text corrections and
// translations cannot be overwritten by late audio classification.
type segmentSpeakerWriter interface {
	UpdateSegmentSpeakers(context.Context, int64, map[int]string) error
}

func (m *Manager) observeVoiceFrames(id string, frames []speaker.Frame) {
	m.mu.Lock()
	if m.id != id || m.voice == nil {
		m.mu.Unlock()
		return
	}
	m.acoustic.add(frames)
	labels := map[int]string{}
	segments := slices.Clone(m.snap.Segments)
	for i := range segments {
		seg := &segments[i]
		if seg.Source != SourceSystem || seg.Speaker != "" {
			continue
		}
		if id := m.acoustic.forRange(seg.Start, seg.End); id != "" {
			seg.Speaker = id
			labels[seg.StoreIdx] = id
			m.speakers.nameSegment(seg)
		}
	}
	m.snap.Segments = segments
	m.snap.Speakers = maps.Clone(m.speakers.names)
	if m.state == StateRecording && m.snap.State == StateRecording && len(frames) > 0 {
		last := frames[len(frames)-1]
		m.snap.LiveSpeakers = map[Source]string{SourceMic: "me", SourceSystem: m.acoustic.slots[voiceSlot(last.Scores)]}
		m.snap.SpeakerAt = last.End
	}
	seq := m.seq
	snapshot := m.snap
	m.mu.Unlock()
	m.notify(snapshot)
	m.writeSpeakerLabels(seq, labels)
}

func (m *Manager) writeSpeakerLabels(seq int64, labels map[int]string) {
	if len(labels) == 0 {
		return
	}
	if writer, ok := m.cfg.Store.(segmentSpeakerWriter); ok {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		if err := writer.UpdateSegmentSpeakers(ctx, seq, labels); err != nil && m.cfg.Warn != nil {
			m.cfg.Warn("saving live speaker labels: " + err.Error())
		}
	}
}

func (m *Manager) speakerRun() *voiceRun {
	m.mu.Lock()
	defer m.mu.Unlock()
	accepting := m.snap.State == StateStarting || m.snap.State == StateRecording
	if accepting && m.voice == nil && m.cfg.SpeakerModel != nil && m.cfg.Settings().Speakers && m.mode.Diarize() {
		id := m.id
		m.voice = newVoiceRun(m.cfg.SpeakerModel, func(frames []speaker.Frame) { m.observeVoiceFrames(id, frames) }, m.cfg.Warn)
	}
	return m.voice
}

// stopVoiceLocked releases the classifier before meeting metadata is saved.
// The manager lock is released while its bounded worker shuts down.
func (m *Manager) stopVoiceLocked() {
	if m.voice == nil {
		return
	}
	r := m.voice
	m.voice = nil
	m.mu.Unlock()
	r.close()
	m.mu.Lock()
}
