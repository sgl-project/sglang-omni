package local

import "testing"

func TestLibraryHoldNoOp(t *testing.T) {
	lib := &Library{}
	rel := lib.Hold()
	rel()
}

func TestLibraryHoldDelegates(t *testing.T) {
	var calls int
	lib := &Library{open: Openers{Hold: func() func() {
		calls++
		return func() { calls-- }
	}}}
	rel := lib.Hold()
	if calls != 1 {
		t.Fatalf("hold calls %d", calls)
	}
	rel()
	if calls != 0 {
		t.Fatalf("after release calls %d", calls)
	}
}
