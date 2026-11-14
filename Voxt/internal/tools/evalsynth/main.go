// Command evalsynth synthesises VoxtNew's speech fixtures on a Mac: the
// eval set's clips (testdata/eval, clips.json) and the speech sample
// (testdata/speech). The voice is Qwen3-TTS (Apache-2.0, its MLX 8-bit
// CustomVoice conversion), run by AudioSmith's engine in its one-shot mode;
// nothing is played (the engine writes files).
//
//	go run ./internal/tools/evalsynth -engine ENGINE [-only ID,…] [-force] testdata/eval
//	go run ./internal/tools/evalsynth -engine ENGINE -voice Uncle_Fu -seed 5 \
//	    -gain 0.9 -length 10.5 -text "…" -o testdata/speech/sample-zh-en.wav
//
// Each take is post-processed the same way (the speech sample's recipe,
// in Go instead of SoX): silence under -60 dBFS trimmed at both ends, 0.3 s
// of silence put back before and after (a one-off take: more after, to
// -length), the peak normalised to -3 dBFS (then -gain dB more),
// resampled to 16 kHz (a windowed-sinc low-pass) and written as 16-bit
// mono PCM WAV with a fixed-seed triangular dither: the same take gives the
// same bytes.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

const defaultModel = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"

// clip is the part of a clips.json entry this command reads.
type clip struct {
	ID    string  `json:"id"`
	Voice string  `json:"voice"`
	Seed  int     `json:"seed"`
	Speed float64 `json:"speed"`
	Text  string  `json:"text"`
	// Speak is what the voice reads when it differs from Text (a
	// spelling the voice says better).
	Speak string `json:"speak"`
}

func main() {
	engine := flag.String("engine", os.Getenv("AUDIOSMITH_ENGINE"), "audiosmith-engine (or $AUDIOSMITH_ENGINE)")
	model := flag.String("model", defaultModel, "the TTS model")
	only := flag.String("only", "", "comma-separated clip ids (default: the clips without a WAV)")
	force := flag.Bool("force", false, "synthesise clips that have a WAV again")
	text := flag.String("text", "", "one take: the text")
	voice := flag.String("voice", "Uncle_Fu", "one take: the voice")
	seed := flag.Int("seed", 7, "one take: the seed")
	speed := flag.Float64("speed", 0, "one take: the speed (0: the voice's)")
	out := flag.String("o", "", "one take: the WAV to write")
	gain := flag.Float64("gain", 0, "one take: dB added after the peak is normalised (the speech sample's level)")
	length := flag.Float64("length", 0, "one take: seconds to pad the end to (0: 0.3 s after the speech)")
	flag.Parse()
	if *engine == "" {
		fail(fmt.Errorf("-engine: where audiosmith-engine is"))
	}
	s := synth{engine: *engine, model: *model}
	if *text != "" {
		if *out == "" || flag.NArg() != 0 {
			fail(fmt.Errorf("one take needs -o and no directory"))
		}
		s.gain, s.length = *gain, *length
		fail(s.take(clip{Voice: *voice, Seed: *seed, Speed: *speed, Text: *text}, *out))
		return
	}
	if flag.NArg() != 1 {
		fail(fmt.Errorf("usage: evalsynth -engine ENGINE [-only ID,…] [-force] DIR"))
	}
	dir := flag.Arg(0)
	raw, err := os.ReadFile(filepath.Join(dir, "clips.json"))
	if err != nil {
		fail(err)
	}
	var m struct {
		Clips []clip `json:"clips"`
	}
	if err := json.Unmarshal(raw, &m); err != nil {
		fail(err)
	}
	want := map[string]bool{}
	for _, id := range strings.Split(*only, ",") {
		if id = strings.TrimSpace(id); id != "" {
			want[id] = true
		}
	}
	for _, c := range m.Clips {
		path := filepath.Join(dir, c.ID+".wav")
		if len(want) > 0 && !want[c.ID] {
			continue
		}
		if _, err := os.Stat(path); err == nil && !*force && len(want) == 0 {
			continue
		}
		if err := s.take(c, path); err != nil {
			fail(fmt.Errorf("%s: %w", c.ID, err))
		}
	}
}

func fail(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "evalsynth:", err)
		os.Exit(1)
	}
}

type synth struct {
	engine, model string
	gain, length  float64
}

// take synthesises c and writes the post-processed WAV to path.
func (s synth) take(c clip, path string) error {
	tmp, err := os.MkdirTemp("", "evalsynth")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmp)
	raw := filepath.Join(tmp, "take.wav")
	speak := c.Speak
	if speak == "" {
		speak = c.Text
	}
	args := []string{"--model", s.model, "--voice", c.Voice, "--seed", fmt.Sprint(c.Seed), "--text", speak, "--output", raw}
	if c.Speed > 0 {
		args = append(args, "--speed", fmt.Sprint(c.Speed))
	}
	cmd := exec.Command(s.engine, args...)
	cmd.Stderr = os.Stderr
	if b, err := cmd.Output(); err != nil {
		return fmt.Errorf("the engine: %v: %s", err, b)
	}
	b, err := os.ReadFile(raw)
	if err != nil {
		return err
	}
	x, rate, err := readWAV(b)
	if err != nil {
		return err
	}
	y := process(x, rate, s.gain, s.length)
	fmt.Printf("%s: %.2f s\n", filepath.Base(path), float64(len(y))/outRate)
	return os.WriteFile(path, writeWAV(y, uint32(c.Seed)), 0o644)
}
