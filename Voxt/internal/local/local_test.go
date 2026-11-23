package local

import (
	"context"
	"errors"
	"testing"
)

func TestChoose(t *testing.T) {
	list := []Model{{Kind: ASR, ID: "a", Name: "A"}, {Kind: ASR, ID: "b", Name: "B"}}
	for _, c := range []struct {
		want, got string
	}{{"b", "b"}, {" b ", "b"}, {"", "a"}, {"gone", "a"}} {
		if m, ok := Choose(list, c.want); !ok || m.ID != c.got {
			t.Errorf("Choose(%q) = %+v %v, want %s", c.want, m, ok, c.got)
		}
	}
	if m, ok := Choose(nil, "a"); ok {
		t.Errorf("Choose of nothing = %+v", m)
	}
}

func TestNotInstalled(t *testing.T) {
	var s Services = NotInstalled{}
	if l := s.Installed(ASR); len(l) != 0 {
		t.Errorf("installed %v", l)
	}
	p, _, err := s.ASR(context.Background(), "")
	if p != nil || !errors.Is(err, ErrNotInstalled) || err.Error() != "local: model not installed: no asr model" {
		t.Errorf("ASR %v %v", p, err)
	}
	c, _, err := s.LLM(context.Background(), "qwen")
	if c != nil || !errors.Is(err, ErrNotInstalled) || err.Error() != `local: model not installed: llm model "qwen"` {
		t.Errorf("LLM %v %v", c, err)
	}
}
