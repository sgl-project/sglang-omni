package scene

import (
	"image"
	"testing"
)

func TestRepackOrdersEqualSizedRectanglesByPosition(t *testing.T) {
	atlas := NewAtlas(1, 6, 4)
	for x := 0; x < 3; x++ {
		atlas.Pix[x] = byte(x + 1)
	}
	rectangles := []image.Rectangle{
		image.Rect(2, 0, 3, 1),
		image.Rect(0, 0, 1, 1),
		image.Rect(1, 0, 2, 1),
	}
	positions, ok := atlas.Repack(rectangles)
	if !ok {
		t.Fatal("repack failed")
	}
	wantPositions := []image.Point{{X: 4, Y: 0}, {X: 0, Y: 0}, {X: 2, Y: 0}}
	for index, want := range wantPositions {
		if positions[index] != want {
			t.Errorf("rectangle %d moved to %v, want %v", rectangles[index], positions[index], want)
		}
	}
	for _, want := range []struct {
		x     int
		value byte
	}{{4, 3}, {0, 1}, {2, 2}} {
		if atlas.Pix[want.x] != want.value {
			t.Errorf("pixel %d = %d, want %d", want.x, atlas.Pix[want.x], want.value)
		}
	}
}
