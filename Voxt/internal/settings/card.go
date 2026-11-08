package settings

import "encoding/json"

// Answer-card feature keys (match store.Kind values; settings stays a leaf).
const (
	answerCardTranscribe = "transcribe"
	answerCardTranslate  = "translate"
	answerCardRewrite    = "rewrite"
)

// Answer-card text height caps (DIP); keep in sync with layout.CardBodyMax and
// layout.CardBodyLarge.
const cardTextMaxCompact = 220
const cardTextMaxLarge = 400

const (
	CardSizeCompact = ""
	CardSizeLarge   = "large"
)

// migrateShowCard turns Translate's card switch, the only one before the
// switch was every feature's (features.translate.showCard), into
// Features.ShowCard: who showed translations in the card keeps the card.
func migrateShowCard(data []byte, f *Features) {
	var doc struct {
		Features struct {
			Translate struct {
				ShowCard bool `json:"showCard"`
			} `json:"translate"`
		} `json:"features"`
	}
	if json.Unmarshal(data, &doc) == nil && doc.Features.Translate.ShowCard {
		f.ShowCard = true
	}
}

// migratePerModeCard copies the old global card switch into each feature's
// ShowCard when that key is missing. Without a global key, Transcribe
// defaults off (Translate kept migrateShowCard's global from translate-only).
func migratePerModeCard(data []byte, f *Features) {
	migrateShowCard(data, f)
	global := f.ShowCard
	var doc struct {
		Features struct {
			ShowCard   *bool `json:"showCard"`
			Transcribe struct {
				ShowCard *bool `json:"showCard"`
			} `json:"transcribe"`
			Translate struct {
				ShowCard *bool `json:"showCard"`
			} `json:"translate"`
		} `json:"features"`
	}
	if json.Unmarshal(data, &doc) != nil {
		return
	}
	hadGlobal := doc.Features.ShowCard != nil
	if doc.Features.Transcribe.ShowCard == nil {
		if hadGlobal {
			f.Transcribe.ShowCard = global
		} else {
			f.Transcribe.ShowCard = false
		}
	}
	if doc.Features.Translate.ShowCard == nil && hadGlobal {
		f.Translate.ShowCard = global
	}
}

// ShowCardFor reports whether results for kind go to the answer card by
// preference (Rewrite always does; delivery checks may still force the card).
func (f Features) ShowCardFor(kind string) bool {
	switch kind {
	case answerCardTranscribe:
		return f.Transcribe.ShowCard
	case answerCardTranslate:
		return f.Translate.ShowCard
	default:
		return true
	}
}

// CardBodyMax is the answer card's text height cap for kind.
func (f Features) CardBodyMax(kind string) int {
	size := CardSizeCompact
	switch kind {
	case answerCardTranscribe:
		size = f.Transcribe.CardSize
	case answerCardTranslate:
		size = f.Translate.CardSize
	case answerCardRewrite:
		size = f.Rewrite.CardSize
	}
	if size == CardSizeLarge {
		return cardTextMaxLarge
	}
	return cardTextMaxCompact
}
