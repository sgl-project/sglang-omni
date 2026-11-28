package meeting

import (
	"encoding/binary"
	"fmt"
	"os"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

// Frames with a capture time land where it says, so both files hold the
// same moment at the same position: the system's tap starts 150 ms after
// the microphone, the microphone has jitter, a gap and an overlap, a
// pause starts a new period on a clock that moved on, and the files end
// at one length.
func TestArchivePlacesFramesByCaptureTime(t *testing.T) {
	a := newArchive(t.TempDir())
	write := archiveWriter(t, a)
	write(SourceMic, 1, 100, 1000)
	write(SourceSystem, 5, 100, 1150) // 150 ms of silence first
	write(SourceMic, 2, 100, 1110)    // 10 ms late: jitter, appended
	write(SourceMic, 3, 100, 1400)    // after a 200 ms gap
	write(SourceMic, 4, 100, 1450)    // 50 ms over what is written: cut
	if err := a.NewPeriod(); err != nil {
		t.Fatal(err)
	}
	write(SourceMic, 6, 100, 90_000)
	write(SourceSystem, 7, 100, 90_050)
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}

	checkArchive(t, a, map[Source]string{
		SourceMic:    "1×100 2×100 0×200 3×100 4×50 6×100 0×50",
		SourceSystem: "0×150 5×100 0×350 7×100",
	})
}

// A track that starts after the longest gap one frame may pad still lines
// up (Recording switched to Meeting after a long while), and so does one
// that stops and starts again (Meeting to Recording and back); a frame far
// past every file, or far before its own file's end, is a clock that went
// wrong and is appended as it comes, not padded or cut away.
func TestArchiveLateTrackAndClockJumps(t *testing.T) {
	old := archiveMaxGap
	archiveMaxGap = asr.SampleRate // 1 s
	t.Cleanup(func() { archiveMaxGap = old })
	a := newArchive(t.TempDir())
	write := archiveWriter(t, a)
	for i := range 30 { // 3 s of microphone alone
		write(SourceMic, 1, 100, 1000+float64(100*i))
	}
	write(SourceSystem, 5, 100, 4000) // starts 3 s in: padded, past the 1 s limit
	for i := range 30 {               // 3 s more of microphone alone: the system track stopped
		write(SourceMic, 3, 100, 4000+float64(100*i))
	}
	write(SourceSystem, 7, 100, 7000) // back 2.9 s after it stopped: padded
	write(SourceMic, 2, 100, 13_000)  // 5.9 s past every file: appended
	write(SourceSystem, 6, 100, 100)  // 7 s before its own end: appended, not cut
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	checkArchive(t, a, map[Source]string{
		SourceMic:    "1×3000 3×3000 2×100 0×100",
		SourceSystem: "0×3000 5×100 0×2900 7×100 6×100",
	})
}

// A period starts at its first frame, whichever track gives it: what the
// other track captured before is cut. At the start the microphone loses
// what it heard before the system's first frame; after a pause that added
// the microphone (Subtitles switched to Meeting), its first frames do not
// run back into the period before.
func TestArchivePeriodStartsAtItsFirstFrame(t *testing.T) {
	a := newArchive(t.TempDir())
	write := archiveWriter(t, a)
	write(SourceSystem, 5, 100, 1060)
	write(SourceMic, 1, 100, 1000) // 60 ms before the period's start
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	checkArchive(t, a, map[Source]string{
		SourceMic:    "1×40 0×60",
		SourceSystem: "5×100",
	})

	a = newArchive(t.TempDir())
	write = archiveWriter(t, a)
	write(SourceSystem, 5, 100, 1000)
	if err := a.NewPeriod(); err != nil {
		t.Fatal(err)
	}
	write(SourceSystem, 6, 100, 90_200)
	write(SourceMic, 1, 100, 89_968) // all before the period: dropped
	write(SourceMic, 2, 100, 90_068)
	write(SourceMic, 3, 100, 90_168) // its first 32 ms are before
	write(SourceMic, 4, 100, 90_268)
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	checkArchive(t, a, map[Source]string{
		SourceMic:    "0×100 3×68 4×100",
		SourceSystem: "5×100 6×100 0×68",
	})
}

// A file that cannot be padded at a resume does not keep the period
// before: the next frames are not placed a whole pause later.
func TestArchiveNewPeriodAfterAFailedPad(t *testing.T) {
	a := newArchive(t.TempDir())
	write := archiveWriter(t, a)
	write(SourceMic, 1, 100, 1000)
	write(SourceSystem, 5, 50, 1000)
	a.w[SourceSystem].f.Close()
	if err := a.NewPeriod(); err == nil {
		t.Fatal("padding a closed file did not fail")
	}
	write(SourceMic, 6, 100, 90_000)
	_ = a.Close() // the closed file fails again
	checkArchive(t, a, map[Source]string{SourceMic: "1×100 6×100"})
}

// archiveWriter writes ms of the value v to src, captured at the host time at.
func archiveWriter(t *testing.T, a *Archive) func(src Source, v int16, ms int, at float64) {
	return func(src Source, v int16, ms int, at float64) {
		t.Helper()
		pcm := make([]byte, 2*ms*asr.SampleRate/1000)
		for i := 0; i < len(pcm); i += 2 {
			binary.LittleEndian.PutUint16(pcm[i:], uint16(v))
		}
		if err := a.WriteAt(src, pcm, at); err != nil {
			t.Fatal(err)
		}
	}
}

// checkArchive compares each file, as value×ms runs, with want.
func checkArchive(t *testing.T, a *Archive, want map[Source]string) {
	t.Helper()
	for src, want := range want {
		raw, err := os.ReadFile(a.path(src))
		if err != nil {
			t.Fatal(err)
		}
		var got string
		for i := 44; i < len(raw); {
			v, j := binary.LittleEndian.Uint16(raw[i:]), i
			for j < len(raw) && binary.LittleEndian.Uint16(raw[j:]) == v {
				j += 2
			}
			got += fmt.Sprintf(" %d×%d", int16(v), (j-i)/2*1000/asr.SampleRate)
			i = j
		}
		if got[1:] != want {
			t.Errorf("%s: %s, want %s", src, got[1:], want)
		}
	}
}
