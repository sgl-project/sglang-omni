package main

// The logs' order (shell_logs_order.go): the newest line first, copies in
// the log's order, a list at its top follows new lines, one scrolled down
// keeps its place and says how many came.

import (
	"fmt"
	"testing"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
)

func appendLines(a *app, from, to int) {
	for i := from; i <= to; i++ {
		a.logs.Append(core.LogEntry{Source: "app", Message: fmt.Sprintf("line %d", i)})
	}
}

func lineY(tt *ui.Tester, i int) (float32, bool) {
	r, ok := tt.Find(fmt.Sprintf("line %d", i))
	return r.Y, ok
}

func TestLogsNewestFirst(t *testing.T) {
	a, tt := logsApp(t, 5)
	for _, panel := range []bool{false, true} {
		if panel {
			a.navigate("/history")
			a.logsPanel = true
			frames(tt)
		}
		var prev float32
		for i := 5; i >= 1; i-- {
			y, ok := lineY(tt, i)
			if !ok {
				t.Fatalf("panel %v: no line %d: %q", panel, i, firstTexts(tt))
			}
			if i < 5 && y <= prev {
				t.Errorf("panel %v: line %d (y %v) not under line %d (y %v)", panel, i, y, i+1, prev)
			}
			prev = y
		}
	}
	// The newest are kept when the filter selects more than the view shows.
	if lines := a.visibleLogs(); lines[0].Message != "line 5" || lines[len(lines)-1].Message != "line 1" {
		t.Errorf("order %q … %q", lines[0].Message, lines[len(lines)-1].Message)
	}
}

func TestLogsCopyInLogOrder(t *testing.T) {
	a, tt := logsApp(t, 6)
	// Selected from the newest down, copied oldest first.
	clickLine(t, tt, ui.Cmd, 5)
	clickLine(t, tt, ui.Cmd, 2)
	clickLine(t, tt, ui.Cmd, 4)
	tt.Key(ui.Cmd, ui.KeyC)
	frames(tt)
	if got, want := tt.Clipboard(), formatted(a, 2, 4, 5); got != want {
		t.Errorf("⌘C:\n%s\nwant\n%s", got, want)
	}
	// Shift from the newer to the older one: the same lines, oldest first.
	clickLine(t, tt, 0, 6)
	clickLine(t, tt, ui.Shift, 3)
	wantSelected(t, a, "line 3", "line 4", "line 5", "line 6")
	click(t, tt, "Copy")
	if got, want := tt.Clipboard(), formatted(a, 3, 4, 5, 6); got != want {
		t.Errorf("header Copy:\n%s\nwant\n%s", got, want)
	}
	click(t, tt, "Copy all lines shown")
	if got, want := tt.Clipboard(), formatted(a, 1, 2, 3, 4, 5, 6); got != want {
		t.Errorf("copy all:\n%s\nwant\n%s", got, want)
	}
	// One line as it is.
	clickLine(t, tt, 0, 4)
	if tt.Clipboard() != formatted(a, 4) {
		t.Errorf("one line %q", tt.Clipboard())
	}
}

func TestLogsFollowAtTop(t *testing.T) {
	a, tt := logsApp(t, 60)
	y60, _ := lineY(tt, 60)
	appendLines(a, 61, 63)
	frames(tt)
	y63, ok := lineY(tt, 63)
	if !ok || y63 != y60 {
		t.Errorf("the newest line at %v (%v), want the top %v", y63, ok, y60)
	}
	if st := a.logSel.views[logPage].scroll; st.Y != 0 {
		t.Errorf("left the top: %v", st.Y)
	}
	if hasExact(tt, "3 new lines") {
		t.Error("a pill at the top")
	}
}

func TestLogsKeepPlaceScrolledDown(t *testing.T) {
	a, tt := logsApp(t, 80)
	x, y := lineAt(t, tt, 80)
	tt.Scroll(x, y, 0, 300) // down to older lines
	frames(tt)
	o := &a.logSel.views[logPage]
	if o.scroll.Y < 200 {
		t.Fatalf("did not scroll: %v", o.scroll.Y)
	}
	// A line in view, and where it is.
	read := 0
	var at float32
	for i := 70; i > 0; i-- {
		if y, ok := lineY(tt, i); ok && y > 150 {
			read, at = i, y
			break
		}
	}
	if read == 0 {
		t.Fatalf("no line in view: %q", firstTexts(tt))
	}
	appendLines(a, 81, 85)
	frames(tt)
	if y, ok := lineY(tt, read); !ok || y != at {
		t.Errorf("line %d moved from %v to %v (%v)", read, at, y, ok)
	}
	if !hasExact(tt, "5 new lines") {
		t.Fatalf("no pill: %q", firstTexts(tt))
	}
	appendLines(a, 86, 86)
	frames(tt)
	if y, _ := lineY(tt, read); y != at || !hasExact(tt, "6 new lines") {
		t.Errorf("after one more: line %d at %v, pill %v", read, y, hasExact(tt, "6 new lines"))
	}
	// A selection made meanwhile stays on its lines.
	clickLine(t, tt, ui.Cmd, read)
	appendLines(a, 87, 87)
	frames(tt)
	wantSelected(t, a, fmt.Sprintf("line %d", read))
	// The pill goes back to the top, and goes.
	click(t, tt, "7 new lines")
	if y, ok := lineY(tt, 87); !ok || o.scroll.Y != 0 {
		t.Errorf("after the pill: newest at %v (%v), offset %v", y, ok, o.scroll.Y)
	}
	if hasExact(tt, "7 new lines") {
		t.Error("the pill stayed")
	}
	// At the top again it follows.
	appendLines(a, 88, 88)
	frames(tt)
	if _, ok := lineY(tt, 88); !ok || o.scroll.Y != 0 {
		t.Errorf("not following after the pill: %v", o.scroll.Y)
	}
	// Scrolling back up by hand clears the count too.
	tt.Scroll(x, y, 0, 300)
	frames(tt)
	appendLines(a, 89, 90)
	frames(tt)
	if !hasExact(tt, "2 new lines") {
		t.Fatalf("no pill: %q", firstTexts(tt))
	}
	tt.Scroll(x, y, 0, -2000)
	frames(tt)
	if hasExact(tt, "2 new lines") || o.scroll.Y != 0 {
		t.Errorf("at the top by hand: pill %v, offset %v", hasExact(tt, "2 new lines"), o.scroll.Y)
	}
	if _, ok := lineY(tt, 90); !ok {
		t.Error("the newest line not shown at the top")
	}
}

func TestLogsNewLinesChinese(t *testing.T) {
	a, tt := logsApp(t, 80)
	i18n.SetLocale("zh-Hans")
	x, y := lineAt(t, tt, 80)
	tt.Scroll(x, y, 0, 300)
	frames(tt)
	appendLines(a, 81, 83)
	if !has(tt, "3 条新日志") {
		t.Errorf("pill: %q", firstTexts(tt))
	}
}
