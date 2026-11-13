package meeting

import (
	"fmt"
	"os"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

// ClipWAV writes the PCM in src between start and end into dst as a 16-bit
// PCM mono WAV at the source file's sample rate. Start and end align to
// whole samples (16-bit frames).
func ClipWAV(src, dst string, start, end time.Duration) error {
	if end <= start {
		return fmt.Errorf("meeting: clip end before start")
	}
	raw, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	pcm, rate, ch, err := asr.ReadWAV(raw)
	if err != nil {
		return err
	}
	if ch != 1 {
		return fmt.Errorf("meeting: clip wants mono, got %d channels", ch)
	}
	a := bytesAt(start, rate)
	b := bytesAt(end, rate)
	if a > len(pcm) {
		a = len(pcm)
	}
	if b > len(pcm) {
		b = len(pcm)
	}
	if b < a {
		b = a
	}
	out := pcm[a:b]
	return os.WriteFile(dst, asr.WAV(out, rate), 0o644)
}

func bytesAt(d time.Duration, rate int) int {
	if d < 0 {
		d = 0
	}
	if rate <= 0 {
		rate = asr.SampleRate
	}
	frames := int(d * time.Duration(rate) / time.Second)
	return frames * 2
}
