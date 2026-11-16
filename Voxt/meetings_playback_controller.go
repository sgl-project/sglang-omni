package main

import (
	"context"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/egoist/mygo/audio"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

type meetingDetailPlayback struct {
	pos       time.Duration
	playing   bool
	highlight int
	seq       int
}

type meetingPlayer struct {
	mu      sync.Mutex
	players []*audio.Player
	cancel  context.CancelFunc
}

func (player *meetingPlayer) stop() {
	player.mu.Lock()
	cancel := player.cancel
	for _, stream := range player.players {
		stream.Close()
	}
	player.players = nil
	player.cancel = nil
	player.mu.Unlock()
	if cancel != nil {
		cancel()
	}
}

func (a *app) stopMeetingPlayback() {
	a.mtPlayer.stop()
	if detail := a.mv.detail; detail != nil {
		detail.playback.playing = false
		detail.playback.highlight = -1
		detail.playback.seq++
		a.refreshMeetingPlaybackState()
	}
}

func (a *app) toggleMeetingPlayback(id string) {
	detail := a.mv.detail
	if detail == nil || detail.id != id || detail.playback.playing {
		a.stopMeetingPlayback()
		return
	}
	a.seekMeetingPlayback(id, detail.playback.pos.Milliseconds())
}

func (a *app) seekMeetingPlayback(id string, startMS int64) {
	detail := a.mv.detail
	if detail == nil || detail.id != id || detail.meeting.Audio == "" {
		return
	}
	at := time.Duration(startMS) * time.Millisecond
	directory := filepath.Join(a.dataDir(), "meetings", detail.meeting.Audio)
	a.stopMeetingPlayback()
	segments := playbackSegments(detail.segments)
	detail.playback.seq++
	seq := detail.playback.seq
	detail.playback.pos = at
	detail.playback.playing = true
	detail.playback.highlight = meeting.SegmentIndexAt(segments, at)
	a.refreshMeetingPlaybackState()
	a.async(func() {
		if !meeting.AudioFolderReady(filepath.Join(a.dataDir(), "meetings"), detail.meeting.Audio) {
			a.post(func() {
				if current := a.mv.detail; current != nil && current.id == id && current.playback.seq == seq {
					current.playback.playing = false
					a.refreshMeetingPlaybackState()
				}
			})
			return
		}
		players, err := openMeetingAudio(meeting.TrackPaths(directory), at)
		duration := meeting.MaxTrackDuration(directory)
		a.post(func() {
			current := a.mv.detail
			if current == nil || current.id != id || current.playback.seq != seq {
				for _, player := range players {
					player.Close()
				}
				return
			}
			if err != nil {
				current.playback.playing = false
				a.refreshMeetingPlaybackState()
				return
			}
			ctx, cancel := context.WithCancel(context.Background())
			a.mtPlayer.mu.Lock()
			a.mtPlayer.players = players
			a.mtPlayer.cancel = cancel
			a.mtPlayer.mu.Unlock()
			for _, player := range players {
				player.Play()
			}
			a.async(func() {
				defer cancel()
				ticker := time.NewTicker(100 * time.Millisecond)
				defer ticker.Stop()
				for {
					select {
					case <-ctx.Done():
						return
					case <-ticker.C:
						position, playing := meetingAudioPosition(players)
						done := !playing || duration > 0 && position >= duration
						a.post(func() {
							current := a.mv.detail
							if current == nil || current.id != id || current.playback.seq != seq {
								return
							}
							current.playback.pos = position
							current.playback.highlight = meeting.SegmentIndexAt(segments, position)
							current.playback.playing = !done
							a.refreshMeetingPlaybackState()
						})
						if done {
							a.post(a.stopMeetingPlayback)
							return
						}
					}
				}
			})
		})
	})
}

func openMeetingAudio(paths []string, at time.Duration) ([]*audio.Player, error) {
	var players []*audio.Player
	for _, path := range paths {
		file, err := os.Open(path)
		if err != nil {
			for _, player := range players {
				player.Close()
			}
			return nil, err
		}
		player, err := audio.OpenStream(file)
		if err != nil {
			file.Close()
			for _, other := range players {
				other.Close()
			}
			return nil, err
		}
		player.Seek(at)
		players = append(players, player)
	}
	return players, nil
}

func meetingAudioPosition(players []*audio.Player) (time.Duration, bool) {
	var position time.Duration
	anyPlaying := false
	for _, player := range players {
		if player.Playing() {
			anyPlaying = true
		}
		if current := player.Position(); current > position {
			position = current
		}
	}
	return position, anyPlaying
}

func playbackSegments(rows []store.MeetingSegment) []meeting.PlaybackSegment {
	segments := make([]meeting.PlaybackSegment, 0, len(rows))
	for _, row := range rows {
		segments = append(segments, meeting.PlaybackSegment{Idx: row.Idx, StartMS: row.StartMS, EndMS: row.EndMS})
	}
	return segments
}
