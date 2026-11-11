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
