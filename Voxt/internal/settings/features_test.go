package settings

import (
	"encoding/json"
	"testing"
)

// TestCardSizeClamp: a saved card size keeps its meaning with the fill
// size added: 0 to 3 read back as stored, anything outside clamps.
func TestCardSizeClamp(t *testing.T) {
	f := Features{CardSizes: map[string]int{"rewrite": 3, "translate": 9, "transcribe": -1}}
	if got := f.CardSize("rewrite"); got != 3 {
		t.Errorf("the fill size reads as %d, want 3", got)
	}
	if got := f.CardSize("translate"); got != CardSizeSteps-1 {
		t.Errorf("a size past the last reads as %d, want %d", got, CardSizeSteps-1)
	}
	if got := f.CardSize("transcribe"); got != 0 {
		t.Errorf("a negative size reads as %d, want 0", got)
	}
}

// TestStoredCardSizesKeepTheirMeaning: settings written before Fill keep
// the same small, medium and large widths.
func TestStoredCardSizesKeepTheirMeaning(t *testing.T) {
	var features Features
	if err := json.Unmarshal([]byte(`{"cardSizes":{"rewrite":0,"translate":1,"transcribe":2}}`), &features); err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		kind string
		want int
	}{{"rewrite", 0}, {"translate", 1}, {"transcribe", 2}} {
		if got := features.CardSize(test.kind); got != test.want {
			t.Errorf("%s: size %d, want %d", test.kind, got, test.want)
		}
	}
}
