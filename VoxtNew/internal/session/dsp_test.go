package session

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/dsp"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/vad"
)

// quietVoice is 100 ms of a quiet voice: a 220 Hz tone at -40 dBFS.
var quietVoice = tone(asr.BytesPerSecond/10, 220, 0.014)

// quietSpeech: the quiet voice from 0.5 s to 2.5 s of a 3 s recording.
func quietSpeech(n int) bool { return n >= 5 && n < 25 }

// TestSoundProcessing: a quiet voice reaches a file transcriber and a
// stream raised by the most (+12 dB) and filtered; the snapshot and the
// history say so; the recording kept is the original. With the processing
// off, the speech model gets the recording as it came (trimmed, as
// before).
func TestSoundProcessing(t *testing.T) {
	for _, off := range []bool{false, true} {
		t.Run("file", func(t *testing.T) {
			file := &fakeFile{text: "quiet words"}
			h := newHarness(t, file, nil, func(_ *Config, s *settings.Settings) {
				s.Audio.Clean.Off, s.Privacy.KeepAudio = off, true
			})
			h.audio.voice, h.audio.speaks = quietVoice, quietSpeech
			id := h.start()
			h.speak(id, 30)
			end := h.rec.wait(t)
			if end.State != Done || end.Result != "quiet words" {
				t.Fatalf("off %v: end %+v", off, end)
			}
			kept, err := os.ReadFile(filepath.Join(h.hist.dir, id+".wav"))
			pcm, _, _, _ := asr.ReadWAV(kept)
			if err != nil || len(pcm) < 30*len(quietVoice) || !bytes.Equal(pcm[5*len(quietVoice):6*len(quietVoice)], quietVoice) {
				t.Fatalf("off %v: the kept recording is not the original (%v)", off, err)
			}
			e := h.hist.all()[0]
			up, orig := vad.LevelDB(file.audio), vad.LevelDB(pcm)
			if off {
				trimmed := vad.Compact(pcm, asr.SampleRate, vad.Analyze(pcm, asr.SampleRate, vad.Params{}), vad.Params{}).PCM
				if !bytes.Equal(file.audio, trimmed) || end.DSP != "" || e.DSP != "" || e.GainDB != 0 {
					t.Errorf("off: upload %.1f dBFS of %.1f, dsp %q, entry %q %v", up, orig, end.DSP, e.DSP, e.GainDB)
				}
				return
			}
			// What it got is the kept recording processed and trimmed, byte
			// for byte, whatever its length; its speech (the recording's
			// 0.5 s to 2.5 s) 12 dB louder.
			f := dsp.ProcessFile(pcm, asr.SampleRate, dsp.Params{}, vad.Params{})
			want := vad.Compact(f.PCM, asr.SampleRate, f.Analysis, vad.Params{}).PCM
			speech := func(b []byte) float64 { return vad.LevelDB(b[6*len(quietVoice) : 24*len(quietVoice)]) }
			gain := speech(f.PCM) - speech(pcm)
			if !bytes.Equal(file.audio, want) || gain < 11.8 || gain > 12.2 {
				t.Errorf("on: upload %.1f dBFS of %.1f (%d bytes, %d expected), speech %+.2f dB", up, orig, len(file.audio), len(want), gain)
			}
			if end.DSP != "highpass+level" || end.GainDB != 12 || e.DSP != end.DSP || e.GainDB != 12 {
				t.Errorf("on: dsp %q %+.1f, entry %q %+.1f", end.DSP, end.GainDB, e.DSP, e.GainDB)
			}
			if !h.logged("sound: highpass+level") {
				t.Errorf("logs %q", h.logs)
			}
		})
		t.Run("stream", func(t *testing.T) {
			st := &fakeStreamer{final: "quiet words"}
			h := newHarness(t, st, nil, func(_ *Config, s *settings.Settings) {
				s.Audio.Clean.Off, s.Privacy.KeepAudio = off, true
			})
			h.audio.voice, h.audio.speaks = quietVoice, quietSpeech
			id := h.start()
			h.speak(id, 30)
			end := h.rec.wait(t)
			if end.State != Done || end.Result != "quiet words" {
				t.Fatalf("off %v: end %+v", off, end)
			}
			s := st.streams[0]
			s.mu.Lock()
			sent := s.data
			s.mu.Unlock()
			kept, _ := os.ReadFile(filepath.Join(h.hist.dir, id+".wav"))
			pcm, _, _, _ := asr.ReadWAV(kept)
			// The speech's last second (the recording's from 1.5 s to
			// 2.5 s, whatever its length): the gain has settled. What was
			// sent is the recording's tail.
			at := len(sent) - len(pcm) + 25*len(quietVoice)
			if at < asr.BytesPerSecond {
				t.Fatalf("off %v: sent %d bytes of %d", off, len(sent), len(pcm))
			}
			level := vad.LevelDB(sent[at-asr.BytesPerSecond : at])
			e := h.hist.all()[0]
			if off {
				if len(pcm) < len(sent) || !bytes.Equal(sent, pcm[len(pcm)-len(sent):]) || end.DSP != "" || e.DSP != "" {
					t.Errorf("off: sent %.1f dBFS, dsp %q", level, end.DSP)
				}
				return
			}
			if end.DSP != "highpass+level" || end.GainDB < 11.5 || e.GainDB != end.GainDB || level < -29 || level > -27.5 {
				t.Errorf("on: sent %.1f dBFS, dsp %q %+.1f, entry %+.1f", level, end.DSP, end.GainDB, e.GainDB)
			}
		})
	}
}
