package meeting

import (
	"fmt"
	"strings"
	"time"
)

// ExportSegment is one line in a meeting export (store segment or live segment).
type ExportSegment struct {
	Start       time.Duration
	Speaker     string
	Text        string
	Translation string
}

// ExportTxt returns a plain-text export (Voxt 1.x joinedText with timestamps).
func ExportTxt(segs []ExportSegment, names map[string]string) string {
	var b strings.Builder
	for _, seg := range segs {
		line := exportLine(seg, names)
		if line == "" {
			continue
		}
		if b.Len() > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(line)
	}
	return strings.TrimSpace(b.String())
}

// ExportMarkdown returns a Markdown transcript export.
func ExportMarkdown(title string, segs []ExportSegment, names map[string]string) string {
	var b strings.Builder
	title = strings.TrimSpace(title)
	if title != "" {
		fmt.Fprintf(&b, "# %s\n\n", title)
	}
	for _, seg := range segs {
		text := strings.TrimSpace(seg.Text)
		if text == "" {
			continue
		}
		label := SpeakerLabel(seg.Speaker, names)
		fmt.Fprintf(&b, "**%s** %s %s\n\n", formatExportTime(seg.Start), label, text)
		if tr := strings.TrimSpace(seg.Translation); tr != "" {
			fmt.Fprintf(&b, "> %s\n\n", tr)
		}
	}
	return strings.TrimSpace(b.String())
}

func exportLine(seg ExportSegment, names map[string]string) string {
	text := strings.TrimSpace(seg.Text)
	if text == "" {
		return ""
	}
	lines := []string{fmt.Sprintf("%s %s %s", formatExportTime(seg.Start), SpeakerLabel(seg.Speaker, names), text)}
	if tr := strings.TrimSpace(seg.Translation); tr != "" {
		lines = append(lines, "   -> "+tr)
	}
	return strings.Join(lines, "\n")
}

func formatExportTime(d time.Duration) string {
	if d < 0 {
		d = 0
	}
	sec := int(d.Round(time.Second).Seconds())
	h := sec / 3600
	m := (sec % 3600) / 60
	s := sec % 60
	if h > 0 {
		return fmt.Sprintf("%02d:%02d:%02d", h, m, s)
	}
	return fmt.Sprintf("%02d:%02d", m, s)
}

// SegmentsForExport maps live segments to export rows.
func SegmentsForExport(segs []Segment) []ExportSegment {
	out := make([]ExportSegment, 0, len(segs))
	for _, s := range segs {
		out = append(out, ExportSegment{
			Start:       s.Start,
			Speaker:     s.Speaker,
			Text:        s.Text,
			Translation: s.Translation,
		})
	}
	return out
}
