package omni

import (
	"context"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/omni/omnitest"
)

func TestEchoTail(t *testing.T) {
	terms := []string{"Voxt", "Omni", "SGLang", "Hugging Face"}
	for _, c := range []struct {
		text, kept string
		found      []string
		certain    bool
	}{
		// The user's dictation of 10-19: the whole context after the last sentence.
		{"这个Bot的这个文本处理还是很拉的，嗯。Voxt，Omni，SGLang。", "这个Bot的这个文本处理还是很拉的，嗯。", []string{"Voxt", "Omni", "SGLang"}, true},
		{"Let's ship it. Voxt, Omni, SGLang.", "Let's ship it.", []string{"Voxt", "Omni", "SGLang"}, true},
		{"好的。Voxt Omni SGLang", "好的。", []string{"Voxt", "Omni", "SGLang"}, true},
		{"好的。Omni、SGLang、Hugging Face。", "好的。", []string{"Omni", "SGLang", "Hugging Face"}, true},
		{"好的！ｖｏｘｔ，ｏｍｎｉ", "好的！", []string{"Voxt", "Omni"}, true},
		// The context as sent, one term a line; only the echo: nothing was said.
		{"Voxt\nOmni\nSGLang", "", []string{"Voxt", "Omni", "SGLang"}, true},
		{"Voxt，Omni，SGLang。", "", []string{"Voxt", "Omni", "SGLang"}, true},
		// One term alone, or terms out of the context's order: maybe said.
		{"Omni。", "", []string{"Omni"}, false},
		{"好的。Omni。", "好的。", []string{"Omni"}, false},
		{"好的。SGLang，Voxt。", "好的。", []string{"SGLang", "Voxt"}, false},
		// Speech that ends with terms.
		{"我们用 Omni。", "我们用 Omni。", nil, false},
		{"我们用 Omni 和 SGLang。", "我们用 Omni 和 SGLang。", nil, false},
		{"我说三个词：Voxt，Omni，SGLang。", "我说三个词：Voxt，Omni，SGLang。", nil, false},
		{"Voxt、Omni 和 SGLang 这三个项目都要更新。", "Voxt、Omni 和 SGLang 这三个项目都要更新。", nil, false},
		{"先更新这两个，Voxt，Omni。", "先更新这两个，Voxt，Omni。", nil, false},
		{"好的。Voxt，XOmni。", "好的。Voxt，XOmni。", nil, false},
		{"", "", nil, false},
	} {
		cut, found, certain := echoTail(c.text, terms, false)
		kept := c.text
		if cut >= 0 {
			kept = c.text[:cut]
		}
		if kept != c.kept || !slices.Equal(found, c.found) || certain != c.certain {
			t.Errorf("echoTail(%q) = %q, %v, %v; want %q, %v, %v", c.text, kept, found, certain, c.kept, c.found, c.certain)
		}
	}
}

func TestEchoTailCutOff(t *testing.T) {
	terms := []string{"Voxt", "Omni", "SGLang", "Kubernetes"}
	text := "好的。Voxt\nOmni\nSGLang\nKubern"
	if cut, found, _ := echoTail(text, terms, true); cut < 0 || text[:cut] != "好的。" || len(found) != 3 {
		t.Errorf("cut off: %d %v", cut, found)
	}
	if cut, _, _ := echoTail(text, terms, false); cut >= 0 {
		t.Errorf("a part of a term in a text the budget did not end: %d", cut)
	}
	if cut, _, _ := echoTail("我们用 Kubern", terms, true); cut >= 0 {
		t.Errorf("speech cut off: %d", cut)
	}
}

func TestSpokenTokens(t *testing.T) {
	for text, want := range map[string]int{
		"嗯。": 0, "": 0, "Um, okay.": 1, "好的。S G Lang。": 5, "我们用 Omni 和 SGLang。": 6, "Hugging Face": 2, "3.5 GB": 3,
	} {
		if got := spokenTokens(text); got != want {
			t.Errorf("spokenTokens(%q) = %d, want %d", text, got, want)
		}
	}
}

// TestFinalEcho: the terms the model echoed after the speech go when the
// same audio without the context has none of them; terms the speech has
// stay.
func TestFinalEcho(t *testing.T) {
	transcribeWith := func(o omnitest.Opts, hints ...string) (string, []omnitest.Record) {
		f := newFake(t, o, nil)
		res, err := f.asr(t, ASRFast).Transcribe(context.Background(), asr.FileRequest{Audio: tone(2*time.Second, 0.3), Format: "pcm", Lang: "zh", Hints: hints})
		if err != nil {
			t.Fatal(err)
		}
		return res.Text, f.records("final")
	}
	text, finals := transcribeWith(omnitest.Opts{Text: "好的。", Echo: true}, "VoxtNew", "MLX")
	if text != "好的。" || len(finals) != 2 || finals[1].Fields["prompt"] != "" || finals[1].Samples != finals[0].Samples {
		t.Errorf("echo: %q, finals %+v", text, finals)
	}
	// One term echoed alone goes too, on the same evidence.
	if text, _ := transcribeWith(omnitest.Opts{Text: " ", Echo: true}, "VoxtNew"); text != "" {
		t.Errorf("a lone echo: %q", text)
	}
	// Said: the audio without the context has it too.
	if text, finals := transcribeWith(omnitest.Opts{Text: "好的。MLX。"}, "VoxtNew", "MLX"); text != "好的。MLX。" || len(finals) != 2 {
		t.Errorf("said: %q, %d finals", text, len(finals))
	}
	// No tail of terms: one request.
	if text, finals := transcribeWith(omnitest.Opts{Text: "我们用 MLX。"}, "VoxtNew", "MLX"); text != "我们用 MLX。" || len(finals) != 1 {
		t.Errorf("speech that ends with a term: %q, %d finals", text, len(finals))
	}
}

func TestHeardAgain(t *testing.T) {
	terms := []string{"Omni", "SGLang"}
	for _, c := range []struct {
		plain, kept string
		want        bool
	}{
		{"好的。Omni，S G Long。", "好的。", true},
		{"好的。", "好的。", false},
		{"在《西游记》中，孙悟空是齐天大圣。", "", false},
		{"我们用 omni 部署。嗯。", "我们用 Omni 部署。嗯。", false},
		{"我们用 omni 部署。嗯。Omni。", "我们用 Omni 部署。嗯。", true},
	} {
		if got := heardAgain(c.plain, c.kept, terms); got != c.want {
			t.Errorf("heardAgain(%q, %q) = %v", c.plain, c.kept, got)
		}
	}
}

// TestLivePiecesEcho: the pieces of a long live session and its rest go
// through the same guard.
func TestLivePiecesEcho(t *testing.T) {
	f := newFake(t, omnitest.Opts{Text: "p{n}。", Echo: true}, nil)
	a := piecesASR(t, f)
	st, err := a.Open(context.Background(), asr.StreamRequest{SampleRate: sampleRate, Lang: "zh", Hints: []string{"VoxtNew"}})
	if err != nil {
		t.Fatal(err)
	}
	sendAll(t, st, speechAndPauses(2))
	eventually(t, "two pieces", func() bool { return piecesDone(st) == 2 })
	if err := st.Finish(); err != nil {
		t.Fatal(err)
	}
	evs := drain(t, st)
	if last := evs[len(evs)-1]; last.Kind != asr.EventFinal || strings.Contains(last.Text, "VoxtNew") || !strings.Contains(last.Text, "p") {
		t.Fatalf("final %+v", last)
	}
}
