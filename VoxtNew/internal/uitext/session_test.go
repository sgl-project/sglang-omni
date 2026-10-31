package uitext

import (
	"testing"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

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
		{session.Snapshot{Kind: store.KindRewrite, State: session.Recognizing}, "Recognizing…"},
		{session.Snapshot{Kind: store.KindRewrite, State: session.Delivering}, "Inserting…"},
	} {
		if got := WorkingText(tc.s); got != tc.want {
			t.Errorf("%s %s: %q", tc.s.Kind, tc.s.State, got)
		}
	}
}
