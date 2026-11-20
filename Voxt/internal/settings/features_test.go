package settings

import "testing"

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
