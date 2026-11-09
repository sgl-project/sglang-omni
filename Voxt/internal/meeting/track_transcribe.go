package meeting

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/vad"
)

const maxSnapshotSegments = 200

// TrackASR is one track's live recognition backend (stream and/or file).
type TrackASR struct {
	Caps      asr.Caps
	Streamer  asr.Streamer
	File      asr.FileTranscriber
	StreamReq asr.StreamRequest
	Lang      string
	Prompt    string
	Hints     []string
	VAD       vad.ModelStream // optional Silero for Speech
}

type trackTranscriber struct {
	mgr      *Manager
	src      Source
	job      string
	path     LivePath
	live     *Live
	cutter   Cutter
	file     asr.FileTranscriber
	fileReq  asr.FileRequest
	pcmBytes int64
	timeBase time.Duration

	utterCh  chan Utterance
	stop     chan struct{}
	wg       sync.WaitGroup
	queueDur time.Duration
}

func (m *Manager) newTrackTranscriber(ctx context.Context, src Source, job string) (*trackTranscriber, error) {
	if m.cfg.ASR == nil {
		return nil, nil
	}
	ta, err := m.cfg.ASR(ctx, src)
	if err != nil {
		return nil, err
	}
	tr := &trackTranscriber{mgr: m, src: src, job: job}
	path := PickLivePath(ta.Caps)
	tr.path = path
	switch path {
	case LivePathStream:
		if ta.Streamer == nil {
			return nil, fmt.Errorf("meeting: stream ASR missing for %s", src)
		}
		live := &Live{
			Clock:    m.cfg.Now,
			Streamer: ta.Streamer,
			Req:      ta.StreamReq,
			Source:   src,
			Speech:   NewSpeech(asr.SampleRate, ta.VAD),
			OnCommit: tr.emit,
		}
		if err := live.Open(ctx); err != nil {
			return nil, err
		}
		tr.live = live
	default:
		if ta.File == nil {
			return nil, fmt.Errorf("meeting: file ASR missing for %s", src)
		}
		tr.file = ta.File
		tr.fileReq = asr.FileRequest{
			SampleRate: asr.SampleRate,
			Lang:       ta.Lang,
			Prompt:     ta.Prompt,
			Hints:      ta.Hints,
			Format:     "pcm",
		}
		tr.utterCh = make(chan Utterance, 8)
		tr.stop = make(chan struct{})
		tr.wg.Add(1)
		go tr.fileWorker(ctx)
	}
	tr.refreshTimeBase()
	return tr, nil
}

func (tr *trackTranscriber) refreshTimeBase() {
	tr.timeBase = tr.mgr.timeline.At(tr.src, tr.job, tr.pcmBytes)
}

func (tr *trackTranscriber) onPCM(ctx context.Context, pcm []byte) error {
	tr.pcmBytes += int64(len(pcm))
	if tr.live != nil {
		if err := tr.live.Push(ctx, pcm); err != nil {
			return err
		}
		tr.mgr.updateLive(tr.src, tr.live.LiveText())
		return nil
	}
	for _, u := range tr.cutter.Push(pcm) {
		dur := u.End - u.Start
		tr.queueDur += dur
		tr.mgr.noteBacklog(tr.queueDur)
		select {
		case tr.utterCh <- u:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return nil
}

func (tr *trackTranscriber) finish(ctx context.Context) error {
	if tr.live != nil {
		tr.mgr.updateLive(tr.src, "")
		return tr.live.Finish(ctx)
	}
	for _, u := range tr.cutter.Flush() {
		select {
		case tr.utterCh <- u:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	close(tr.stop)
	tr.wg.Wait()
	return nil
}

func (tr *trackTranscriber) emit(seg Segment) {
	seg.Start += tr.timeBase
	seg.End += tr.timeBase
	if seg.Source == SourceMic {
		seg.Speaker = "me"
	}
	tr.mgr.commitSegment(seg)
}

func (tr *trackTranscriber) fileWorker(ctx context.Context) {
	defer tr.wg.Done()
	for {
		select {
		case <-tr.stop:
			for {
				select {
				case u := <-tr.utterCh:
					tr.transcribeFile(ctx, u)
				default:
					return
				}
			}
		case u := <-tr.utterCh:
			tr.queueDur -= u.End - u.Start
			if tr.queueDur < 0 {
				tr.queueDur = 0
			}
			tr.mgr.noteBacklog(tr.queueDur)
			tr.transcribeFile(ctx, u)
		}
	}
}

func (tr *trackTranscriber) transcribeFile(ctx context.Context, u Utterance) {
	req := tr.fileReq
	req.Audio = u.PCM
	res, err := tr.file.Transcribe(ctx, req)
	if err != nil {
		return
	}
	text := res.Text
	if text == "" {
		return
	}
	seg := Segment{
		ID:      core.NewID(tr.mgr.cfg.Now()),
		Source:  tr.src,
		Start:   tr.timeBase + u.Start,
		End:     tr.timeBase + u.End,
		Text:    text,
		Final:   false,
		Speaker: "",
	}
	if tr.src == SourceMic {
		seg.Speaker = "me"
	}
	tr.mgr.commitSegment(seg)
}

func (m *Manager) updateLive(src Source, text string) {
	m.mu.Lock()
	if m.snap.Live == nil {
		m.snap.Live = map[Source]string{}
	}
	if text == "" {
		delete(m.snap.Live, src)
	} else {
		m.snap.Live[src] = text
	}
	s := m.snap
	m.mu.Unlock()
	m.notify(s)
}

func (m *Manager) commitSegment(seg Segment) {
	if seg.Text == "" {
		return
	}
	if seg.ID == "" {
		seg.ID = core.NewID(m.cfg.Now())
	}
	m.mu.Lock()
	if m.snap.Segments == nil {
		m.snap.Segments = []Segment{}
	}
	m.snap.Segments = append(m.snap.Segments, seg)
	if len(m.snap.Segments) > maxSnapshotSegments {
		m.snap.Segments = m.snap.Segments[len(m.snap.Segments)-maxSnapshotSegments:]
	}
	idx := m.segIdx
	m.segIdx++
	seg.StoreIdx = idx
	seq := m.seq
	target := m.snap.Translate
	tr := m.translator
	s := m.snap
	mic := seg.Source == SourceMic
	m.mu.Unlock()
	if mic {
		m.noteMicText()
	} else {
		m.notify(s)
	}

	go func() {
		ctx := context.Background()
		row := store.MeetingSegment{
			Idx:     idx,
			Source:  toSegmentSource(seg.Source),
			Speaker: seg.Speaker,
			StartMS: seg.Start.Milliseconds(),
			EndMS:   seg.End.Milliseconds(),
			Text:    seg.Text,
			Final:   seg.Final,
		}
		_ = m.cfg.Store.AppendSegments(ctx, seq, []store.MeetingSegment{row})
		if tr != nil && strings.TrimSpace(target) != "" {
			tr.Enqueue([]Segment{seg})
		}
	}()
}

func toSegmentSource(s Source) store.SegmentSource {
	switch s {
	case SourceSystem:
		return store.SegmentSystem
	default:
		return store.SegmentMic
	}
}
