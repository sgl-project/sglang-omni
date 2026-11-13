package meeting_test

import (
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
)

func TestExportTxtAndMarkdown(t *testing.T) {
	segs := []meeting.ExportSegment{
		{Start: 65 * time.Second, Speaker: "me", Text: "Hello", Translation: "Hi"},
		{Start: 130 * time.Second, Speaker: "s1", Text: "Reply", Translation: "Antwort"},
	}
	names := map[string]string{"s1": "Bob"}
	txt := meeting.ExportTxt(segs, names)
	if !strings.Contains(txt, "01:05 Me Hello") || !strings.Contains(txt, "-> Hi") {
		t.Fatalf("txt %q", txt)
	}
	if !strings.Contains(txt, "Bob Reply") || !strings.Contains(txt, "-> Antwort") {
		t.Fatalf("txt speaker %q", txt)
	}
	md := meeting.ExportMarkdown("Stand-up", segs, names)
	if !strings.Contains(md, "# Stand-up") || !strings.Contains(md, "> Hi") {
		t.Fatalf("md %q", md)
	}
}
