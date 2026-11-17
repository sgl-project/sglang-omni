package main

import (
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

// TestCardFollowUpQuote: a follow-up's selection in its bubble (#3816),
// in English and in Chinese: a short one whole, a long one cut, Show all
// shows it whole, Show less cuts it again; Continue's tip says the text
// selected first goes with it.
func TestCardFollowUpQuote(t *testing.T) {
	for _, tc := range []struct{ locale, all, less, tip string }{
		{"en", "Show all", "Show less", "Text you select first goes with it."},
		{"zh-Hans", "展开", "收起", "先选中的文字会一起加进对话。"},
	} {
		t.Run(tc.locale, func(t *testing.T) {
			a, _, _ := newTestApp(t)
			shortScreen(t, a)
			i18n.SetLocale(tc.locale)
			defer i18n.SetLocale("en")
			steps := cardSteps()
			for _, st := range steps {
				st.apply(a)
				if st.name == "follow-up-selected-long" {
					break
				}
			}
			a.overlaySync()
			tt := cardTester(t, a)
			frames(tt)
			v := a.cardNow()
			short, long := v.bubbles[4], v.bubbles[6]
			if short.quote != cardQuote || short.cut || !long.cut || len([]rune(long.quote)) > quoteMax+1 || !strings.HasSuffix(long.quote, "…") {
				t.Fatalf("bubbles %+v / %+v", short, long)
			}
			if !has(tt, tc.all) || has(tt, tc.less) {
				t.Errorf("no %q: %q", tc.all, firstTexts(tt))
			}
			before := a.ov.h
			click(t, tt, tc.all)
			frames(tt)
			if v := a.cardNow(); !a.answer.quotesOpen[3] || v.bubbles[6].quote != strings.TrimSpace(cardQuoteLong) || !has(tt, tc.less) || a.ov.h < before {
				t.Errorf("after %q: open %v, %d high (was %d)", tc.all, a.answer.quotesOpen, a.ov.h, before)
			}
			click(t, tt, tc.less)
			frames(tt)
			if a.answer.quotesOpen[3] || !has(tt, tc.all) {
				t.Errorf("after %q: %v", tc.less, a.answer.quotesOpen)
			}
			if tip := continueTip(a.cardNow().follow); !strings.Contains(tip, tc.tip) || !strings.Contains(continueTip(""), tc.tip) {
				t.Errorf("Continue's tip %q", tip)
			}
		})
	}
}
