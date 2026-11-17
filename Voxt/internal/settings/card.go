package settings

import "encoding/json"

// migrateShowCard reads the answer card switch that was one for the three
// features (features.showCard, 2.0.3 betas): on, it turns each feature's
// own switch on, so who had the card keeps it everywhere. Off was not
// written, so the features keep theirs (Rewrite's on by default).
// Translate's own switch (features.translate.showCard) is read as it is.
func migrateShowCard(data []byte, f *Features) {
	var doc struct {
		Features struct {
			ShowCard bool `json:"showCard"`
		} `json:"features"`
	}
	if json.Unmarshal(data, &doc) == nil && doc.Features.ShowCard {
		f.Transcribe.ShowCard, f.Translate.ShowCard, f.Rewrite.ShowCard = true, true, true
	}
}
