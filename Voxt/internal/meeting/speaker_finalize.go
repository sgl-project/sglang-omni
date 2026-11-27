package meeting

import (
	"strconv"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// keepLiveSpeakers anchors a final pass's independently numbered voices to the
// live timeline. An uncertain match gets a fresh anonymous ID, never a stale
// name that happened to use the same numeric MOSS label.
func keepLiveSpeakers(segments []Segment, source Source, live []store.MeetingSegment) {
	if source != SourceSystem {
		return
	}
	votes := map[string]map[string]time.Duration{}
	totals := map[string]time.Duration{}
	highest := 0
	for _, row := range live {
		if n, err := strconv.Atoi(strings.TrimPrefix(row.Speaker, "s")); err == nil {
			highest = max(highest, n)
		}
	}
	for _, seg := range segments {
		if seg.End <= seg.Start || seg.Speaker == "" {
			continue
		}
		totals[seg.Speaker] += seg.End - seg.Start
		if votes[seg.Speaker] == nil {
			votes[seg.Speaker] = map[string]time.Duration{}
		}
		for _, row := range live {
			if row.Source != store.SegmentSystem || row.Speaker == "" {
				continue
			}
			overlap := min(seg.End, time.Duration(row.EndMS)*time.Millisecond) - max(seg.Start, time.Duration(row.StartMS)*time.Millisecond)
			if overlap > 0 {
				votes[seg.Speaker][row.Speaker] += overlap
			}
		}
	}
	aliases := map[string]string{}
	used := map[string]bool{}
	for id, total := range totals {
		for old, overlap := range votes[id] {
			if overlap >= 500*time.Millisecond && overlap*5 >= total*4 && !used[old] {
				aliases[id] = old
				used[old] = true
			}
		}
	}
	// Do not reinterpret legacy unnamed-only transcripts: no IDs to anchor.
	if highest == 0 {
		return
	}
	for i := range segments {
		id := segments[i].Speaker
		if id == "" {
			continue
		}
		if aliases[id] == "" {
			highest++
			aliases[id] = "s" + strconv.Itoa(highest)
		}
		segments[i].Speaker = aliases[id]
	}
}
