package settings

import "testing"

func TestCardBodyMax(t *testing.T) {
	f := Features{}
	f.Translate.CardSize = CardSizeLarge
	if got := f.CardBodyMax(answerCardTranslate); got != cardTextMaxLarge {
		t.Fatalf("large translate: %d", got)
	}
	f.Rewrite.CardSize = CardSizeLarge
	if got := f.CardBodyMax(answerCardRewrite); got != cardTextMaxLarge {
		t.Fatalf("large rewrite: %d", got)
	}
	if f.CardBodyMax(answerCardTranscribe) != cardTextMaxCompact {
		t.Fatalf("compact transcribe: %d", f.CardBodyMax(answerCardTranscribe))
	}
}

func TestShowCardFor(t *testing.T) {
	f := Features{Transcribe: Transcribe{ShowCard: true}}
	if !f.ShowCardFor(answerCardTranscribe) || f.ShowCardFor(answerCardTranslate) {
		t.Fatal("transcribe only")
	}
	if !f.ShowCardFor(answerCardRewrite) {
		t.Fatal("rewrite always in card path")
	}
}
