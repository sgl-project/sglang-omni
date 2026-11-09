package settings

import "encoding/json"

// migrateShowCard turns Translate's card switch, the only one before the
// switch was every feature's (features.translate.showCard), into
// Features.ShowCard: who showed translations in the card keeps the card.
// A document saved since has no such key, so a choice made later stays.
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
