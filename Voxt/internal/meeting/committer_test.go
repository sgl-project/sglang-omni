package meeting

import (
	"testing"
	"time"
)

func TestUncommittedSuffix(t *testing.T) {
	tests := []struct {
		cumulative, committed, want string
	}{
		{"今天下午开会", "", "今天下午开会"},
		{"今天下午开会", "今天下午", "开会"},
		{"今天下午开会。好的", "今天下午开会", "好的"},
		{"今天下午开会", "今天下午开会", ""},
		{"今天下午开会吧", "今天下午开会", "吧"},
		{"好的", "今天下午开会", "好的"},
		// The model revised the committed tail: emit only what follows the shared prefix (#3431).
		{"今天下午开会吧", "今天下午开个", "会吧"},
		{"今天下午很好", "今天下午开会", "很好"},
	}
	for _, tc := range tests {
		got := uncommittedSuffix(tc.cumulative, tc.committed)
		if got != tc.want {
			t.Errorf("uncommittedSuffix(%q, %q) = %q, want %q", tc.cumulative, tc.committed, got, tc.want)
		}
	}
}

func TestCommitterSilenceFinalize(t *testing.T) {
	var c Committer
	base := 2 * time.Second
	c.BeginPhrase(0)
	c.Partial("今天")
	c.Partial("今天下午")
	c.Partial("今天下午开会")
	seg, ok := c.Tick(base, silenceFinalizeLive)
	if !ok {
		t.Fatal("expected commit after silence")
	}
	if seg.Text != "今天下午开会" {
		t.Fatalf("text %q", seg.Text)
	}
	if seg.End != base {
		t.Fatalf("end %v", seg.End)
	}

	c.Partial("今天下午开会。好的")
	seg, ok = c.Tick(base+time.Second, silenceFinalizeLive)
	if !ok || seg.Text != "好的" {
		t.Fatalf("second commit got %q ok=%v", seg.Text, ok)
	}
}

func TestCommitterMaxSegment(t *testing.T) {
	var c Committer
	c.BeginPhrase(0)
	c.Partial("还在说")
	seg, ok := c.Tick(maximumLiveSegment, 0)
	if !ok || seg.Text != "还在说" {
		t.Fatalf("max segment commit got %q ok=%v", seg.Text, ok)
	}
}

func TestCommitterFinal(t *testing.T) {
	var c Committer
	c.BeginPhrase(0)
	c.Partial("第一句")
	if _, ok := c.Tick(2*time.Second, silenceFinalizeLive); !ok {
		t.Fatal("first commit")
	}
	c.Partial("第一句第二句")
	seg, ok := c.Final("第一句第二句", 3*time.Second)
	if !ok {
		t.Fatal("expected final commit")
	}
	if seg.Text != "第二句" {
		t.Fatalf("final tail %q", seg.Text)
	}
	if !seg.Final {
		t.Fatal("want final flag")
	}
}

func TestCommitterLive(t *testing.T) {
	var c Committer
	c.Partial("你好世界")
	if g := c.Live(); g != "你好世界" {
		t.Fatalf("live %q", g)
	}
	c.committed = "你好"
	if g := c.Live(); g != "世界" {
		t.Fatalf("live after partial commit %q", g)
	}
}

// TestCommitterPauseRevisedTail replays #3431: a pause commits a provisional
// tail, the model then revises it, and the next bubble must not repeat the sentence.
func TestCommitterPauseRevisedTail(t *testing.T) {
	var c Committer
	c.BeginPhrase(0)
	c.Partial("Yeah,你看这个应该都可以。但是你选不同的语言，它就会倾向于去翻译个问题。")
	if _, ok := c.Tick(time.Second, silenceFinalizeLive); !ok {
		t.Fatal("expected commit after pause")
	}
	c.Partial("Yeah,你看这个应该都可以。但是你选不同的语言，它就会倾向于去翻译更加，呃，你选的那个语言。")
	if g := c.Live(); g != "更加，呃，你选的那个语言。" {
		t.Fatalf("live after revision %q", g)
	}
	seg, ok := c.Tick(2*time.Second, silenceFinalizeLive)
	if !ok || seg.Text != "更加，呃，你选的那个语言。" {
		t.Fatalf("second commit got %q ok=%v", seg.Text, ok)
	}
}

// TestCommitterForcedCut covers #3987 at the committer: a long phrase ends at
// a short pause, and only a cut with no pause drops the trailing full stop.
func TestCommitterForcedCut(t *testing.T) {
	tests := []struct {
		name, partial string
		phrase, quiet time.Duration
		want          string
		commit        bool
	}{
		{"short pause, short phrase", "然后", time.Second, shortPauseLive, "", false},
		{"short pause, long phrase", "它就是因为。", longPhraseLive, shortPauseLive, "它就是因为。", true},
		{"forced cut zh", "它就。", maximumLiveSegment, 0, "它就", true},
		{"forced cut en", "so I think the.", maximumLiveSegment, 0, "so I think the", true},
		{"forced cut keeps question mark", "还是说？", maximumLiveSegment, 0, "还是说？", true},
		{"forced cut, no punctuation", "还在说", maximumLiveSegment, 0, "还在说", true},
	}
	for _, tc := range tests {
		var c Committer
		c.BeginPhrase(0)
		c.Partial(tc.partial)
		seg, ok := c.Tick(tc.phrase, tc.quiet)
		if ok != tc.commit || seg.Text != tc.want {
			t.Errorf("%s: got %q ok=%v, want %q ok=%v", tc.name, seg.Text, ok, tc.want, tc.commit)
		}
	}

	// The next bubble continues after the cut, not after the dropped full stop.
	var c Committer
	c.BeginPhrase(0)
	c.Partial("它就。")
	c.Tick(maximumLiveSegment, 0)
	c.Partial("它就是因为我。")
	if g := c.Live(); g != "是因为我。" {
		t.Fatalf("live after forced cut %q", g)
	}
}
