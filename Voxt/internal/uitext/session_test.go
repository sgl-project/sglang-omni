package uitext

import (
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func TestSessionTextTranslators(t *testing.T) {
	translator := i18n.NewTranslator("zh-Hans")
	if got := KindTitleFor(translator, store.KindTranslate); got != translator.T("Translate") {
		t.Errorf("kind title: %q", got)
	}
	recording := session.Snapshot{State: session.Recording, Waiting: session.WaitMic}
	if got := WaitTextFor(translator, recording); got != translator.T("Starting the microphone…") {
		t.Errorf("wait text: %q", got)
	}
	composing := session.Snapshot{Kind: store.KindRewrite, State: session.Composing, Selection: "x"}
	if got := WorkingTextFor(translator, composing); got != translator.T("Rewriting the selection…") {
		t.Errorf("working text: %q", got)
	}
}

func TestWorkingTexts(t *testing.T) {
	for _, tc := range []struct {
		s    session.Snapshot
		want string
	}{
		{session.Snapshot{Kind: store.KindTranscribe, State: session.Composing}, "Cleaning up…"},
		{session.Snapshot{Kind: store.KindTranslate, State: session.Composing}, "Translating…"},
		{session.Snapshot{Kind: store.KindTranslate, State: session.Composing, Selection: "x"}, "Translating the selection…"},
		{session.Snapshot{Kind: store.KindRewrite, State: session.Composing}, "Writing…"},
		{session.Snapshot{Kind: store.KindRewrite, State: session.Composing, Selection: "x"}, "Rewriting the selection…"},
		{session.Snapshot{Kind: store.KindTranslate, State: session.Reading}, "Reading the selection…"},
		{session.Snapshot{Kind: store.KindRewrite, State: session.Recognizing}, "Recognising…"},
		{session.Snapshot{Kind: store.KindRewrite, State: session.Delivering}, "Inserting…"},
	} {
		if got := WorkingText(tc.s); got != tc.want {
			t.Errorf("%s %s: %q", tc.s.Kind, tc.s.State, got)
		}
	}
}
