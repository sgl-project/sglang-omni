package main

// Meeting detail audio playback (C1-8): seeks in saved track WAVs via the
// mygo/audio stack (same output path as interaction sounds on macOS).

import (
	"context"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/egoist/mygo/audio"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

type meetingDetailPlayback struct {
	pos       time.Duration
	playing   bool
	highlight int // meeting_segments.idx; -1 none
	seq       int
}

type meetingPlayer struct {
	mu      sync.Mutex
	players []*audio.Player
	cancel  context.CancelFunc
}

func (p *meetingPlayer) stop() {
	p.mu.Lock()
	cancel := p.cancel
	for _, pl := range p.players {
		pl.Close()
	}
	p.players = nil
	p.cancel = nil
	p.mu.Unlock()
	if cancel != nil {
		cancel()
	}
}

func (a *app) stopMeetingPlayback() {
	a.mtPlayer.stop()
	if d := a.mv.detail; d != nil {
		d.playback.playing = false
		d.playback.highlight = -1
		d.playback.seq++
	}
}

func (a *app) toggleMeetingPlayback(d *meetingDetail) {
	if d == nil || d.playback.playing {
		a.stopMeetingPlayback()
		return
	}
	a.seekMeetingPlayback(d, d.playback.pos)
}

func (a *app) seekMeetingPlayback(d *meetingDetail, at time.Duration) {
	if d == nil || d.meeting.Audio == "" {
		return
	}
	dir := filepath.Join(a.dataDir(), "meetings", d.meeting.Audio)
	if !meeting.AudioFolderReady(filepath.Join(a.dataDir(), "meetings"), d.meeting.Audio) {
		return
	}
	a.stopMeetingPlayback()
	segs := playbackSegments(d.segments)
	d.playback.seq++
	seq := d.playback.seq
	d.playback.pos = at
	d.playback.playing = true
	d.playback.highlight = meeting.SegmentIndexAt(segs, at)

	paths := meeting.TrackPaths(dir)
	players, err := openMeetingAudio(paths, at)
	if err != nil {
		d.playback.playing = false
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	a.mtPlayer.mu.Lock()
	a.mtPlayer.players = players
	a.mtPlayer.cancel = cancel
	a.mtPlayer.mu.Unlock()

	dur := meeting.MaxTrackDuration(dir)
	for _, p := range players {
		p.Play()
	}

	a.async(func() {
		defer cancel()
		tick := time.NewTicker(100 * time.Millisecond)
		defer tick.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-tick.C:
				pos, playing := meetingAudioPosition(players)
				done := !playing || (dur > 0 && pos >= dur)
				a.post(func() {
					det := a.mv.detail
					if det == nil || det.id != d.id || det.playback.seq != seq {
						return
					}
					det.playback.pos = pos
					det.playback.highlight = meeting.SegmentIndexAt(segs, pos)
					det.playback.playing = !done
				})
				if done {
					a.post(a.stopMeetingPlayback)
					return
				}
			}
		}
	})
}

func openMeetingAudio(paths []string, at time.Duration) ([]*audio.Player, error) {
	var out []*audio.Player
	for _, p := range paths {
		f, err := os.Open(p)
		if err != nil {
			for _, pl := range out {
				pl.Close()
			}
			return nil, err
		}
		pl, err := audio.OpenStream(f)
		if err != nil {
			f.Close()
			for _, q := range out {
				q.Close()
			}
			return nil, err
		}
		pl.Seek(at)
		out = append(out, pl)
	}
	return out, nil
}

func meetingAudioPosition(players []*audio.Player) (time.Duration, bool) {
	var pos time.Duration
	any := false
	for _, p := range players {
		if p.Playing() {
			any = true
		}
		if d := p.Position(); d > pos {
			pos = d
		}
	}
	return pos, any
}

func playbackSegments(rows []store.MeetingSegment) []meeting.PlaybackSegment {
	out := make([]meeting.PlaybackSegment, 0, len(rows))
	for _, r := range rows {
		out = append(out, meeting.PlaybackSegment{Idx: r.Idx, StartMS: r.StartMS, EndMS: r.EndMS})
	}
	return out
}

func (a *app) meetingPlaybackBar(c *ui.Context, d *meetingDetail) {
	if !meeting.AudioFolderReady(filepath.Join(a.dataDir(), "meetings"), d.meeting.Audio) {
		return
	}
	t := c.Theme()
	pb := &d.playback
	if pb.pos > 0 || pb.playing {
		ui.Text(c, meeting.FormatPlaybackTime(pb.pos)).
			FontSize(px(t, fsCaption)).TextColor(tok(t).Text3).FontFeatures("tnum")
	}
	ic, label := icons.Play, i18n.T("Play")
	if pb.playing {
		ic, label = icons.Pause, i18n.T("Pause")
	}
	if iconButton(c.Key("mt-playback"), ic, label).Clicked() {
		a.toggleMeetingPlayback(d)
	}
}

const (
	meetingTranscriptInline = 40  // segments laid out in the panel itself
	meetingTranscriptH      = 480 // the list's height past them
)

func (a *app) meetingTranscriptList(c *ui.Context, d *meetingDetail) {
	if len(d.segments) == 0 {
		return
	}
	t := c.Theme()
	names := speakerNameMap(d.meeting.Speakers)
	segs := d.segments
	pb := &d.playback
	k := tok(t)
	seg := func(row store.MeetingSegment) {
		start := time.Duration(row.StartMS) * time.Millisecond
		label := meeting.SpeakerLabel(row.Speaker, names)
		switch label { // SpeakerLabel's English defaults, which exports keep
		case "Me":
			label = i18n.T("Me")
		case "Them":
			label = i18n.T("Other")
		case "Speaker " + row.Speaker[1:]: // s1…s4, unnamed
			label = i18n.T("Speaker %s", row.Speaker[1:])
		}
		// The whole line seeks the audio to its start.
		b := ui.ButtonBase(c.Key("mt-seg-" + itoa(row.Idx))).Padding(px(t, sp2)).Gap(px(t, sp3)).AlignItems(ui.Start).Radius(px(t, radiusControl)).
			Label(meeting.FormatPlaybackTime(start) + " " + label)
		switch {
		case row.Idx == pb.highlight:
			b.Background(k.AccentSoft)
		case b.Pressed():
			b.Background(k.Pressed)
		case b.Hovered():
			b.Background(k.Hover)
		}
		if b.Clicked() {
			a.seekMeetingPlayback(d, start)
		}
		b.Children(func() {
			ui.Text(c, meeting.FormatPlaybackTime(start)).FontSize(px(t, fsCaption)).TextColor(k.Text3).
				FontFeatures("tnum").Width(px(t, 40)).Shrink(0)
			ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, 2)).Children(func() {
				who := ui.Text(c, label).FontSize(px(t, fsCaption)).FontWeight(600).TextColor(k.Text2)
				if row.Source == store.SegmentMic {
					who.TextColor(k.AccentInk)
				}
				ui.Text(c, row.Text)
				if tr := row.Translation; tr != "" {
					ui.Text(c, tr).TextColor(k.Text2).FontSize(px(t, fsCaption))
				}
			})
		})
	}
	// A long meeting lays out thousands of rows: past a screenful, only
	// those in view, in a list of its own inside the panel's scroll.
	if len(segs) <= meetingTranscriptInline {
		ui.Column(c.Key("mt-transcript")).Children(func() {
			for _, row := range segs {
				seg(row)
			}
		})
		return
	}
	ui.List(c.Key("mt-transcript"), &d.segList, len(segs)).Height(px(t, meetingTranscriptH)).Shrink(0).
		ItemKey(func(i int) any { return segs[i].Idx }).Rows(func(r ui.ListRow) {
		seg(segs[r.Index])
	})
}
