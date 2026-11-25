package main

// The hat studio's controller (share_studio.go): the strokes and their
// undo steps, the pen's color and size, the eraser, the wardrobe, and Put on, which renders the board to a PNG in the library's
// "hats" folder and puts it on the card's Voxt.

import (
	"bytes"
	"fmt"
	"image/png"
	"log"
	"math"
	"os"
	"path/filepath"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

type studioKeep struct {
	base    string
	strokes []shareStroke
	// undo holds the board before each change: its base and strokes.
	undo    []shareBoard
	color   string
	brush   int
	erasing bool
	peek    bool
	closet  bool
}

type shareBoard struct {
	base    string
	strokes []shareStroke
}

func (s *shareKeep) studioView() *shareStudioViewState {
	st := s.studio
	if st == nil {
		return nil
	}
	view := &shareStudioViewState{Base: st.base, Strokes: st.strokes, Color: st.color, Brush: st.brush,
		Erasing: st.erasing, CanUndo: len(st.undo) > 0, Peek: st.peek, Closet: st.closet}
	if st.closet {
		for _, o := range shareOutfits {
			view.Wardrobe = append(view.Wardrobe, "outfit:"+o)
		}
		for _, h := range s.wardrobe {
			view.Wardrobe = append(view.Wardrobe, "hat:"+h)
		}
	}
	return view
}

// openStudio opens a blank board with the vivid red pen, medium.
func (a *app) openStudio() {
	a.share.studio = &studioKeep{color: shareStudioPalette[1][0], brush: 1}
}

func (a *app) studioCancel() { a.share.studio = nil }

// keep saves the board as an undo step before a change.
func (st *studioKeep) keep() {
	st.undo = append(st.undo, shareBoard{base: st.base, strokes: append([]shareStroke(nil), st.strokes...)})
	if len(st.undo) > 50 {
		st.undo = st.undo[1:]
	}
}

// studioPoint is the pen at (x, y), in the mascot's units: begin starts a
// stroke, the points after it continue it; with the eraser, it erases.
func (a *app) studioPoint(x, y float32, begin bool) {
	st := a.share.studio
	if st == nil || st.peek {
		return
	}
	if st.erasing {
		if begin {
			st.keep()
		}
		st.strokes = shareErase(st.strokes, x, y, shareBrushes[st.brush]*1.2)
		return
	}
	if begin {
		st.keep()
		// A point alone draws nothing: a click makes a dot.
		st.strokes = append(st.strokes, shareStroke{Color: st.color, Width: shareBrushes[st.brush], Points: [][2]float32{{x, y}, {x + .01, y}}})
		return
	}
	if len(st.strokes) == 0 {
		return
	}
	last := &st.strokes[len(st.strokes)-1]
	p := last.Points[len(last.Points)-1]
	if math.Hypot(float64(x-p[0]), float64(y-p[1])) < 1 {
		return // the pointer barely moved
	}
	last.Points = append(last.Points, [2]float32{x, y})
}

// studioEnd ends a stroke; an eraser that touched nothing leaves no undo
// step.
func (a *app) studioEnd() {
	st := a.share.studio
	if st == nil || !st.erasing || len(st.undo) == 0 {
		return
	}
	if prev := st.undo[len(st.undo)-1]; prev.base == st.base && shareStrokesEqual(prev.strokes, st.strokes) {
		st.undo = st.undo[:len(st.undo)-1]
	}
}

func (a *app) studioColor(color string) {
	if st := a.share.studio; st != nil {
		st.color, st.erasing = color, false
	}
}

func (a *app) studioBrush(brush int) {
	if st := a.share.studio; st != nil && brush >= 0 && brush < len(shareBrushes) {
		st.brush, st.erasing = brush, false
	}
}

func (a *app) studioEraser() {
	if st := a.share.studio; st != nil {
		st.erasing = !st.erasing
	}
}

func (a *app) studioUndo() {
	st := a.share.studio
	if st == nil || len(st.undo) == 0 {
		return
	}
	prev := st.undo[len(st.undo)-1]
	st.undo = st.undo[:len(st.undo)-1]
	st.base, st.strokes = prev.base, prev.strokes
}

func (a *app) studioClear() {
	st := a.share.studio
	if st == nil || (st.base == "" && len(st.strokes) == 0) {
		return
	}
	st.keep()
	st.base, st.strokes = "", nil
}

func (a *app) studioPeek() {
	if st := a.share.studio; st != nil {
		st.peek = !st.peek
	}
}

func (a *app) studioCloset(open bool) {
	if st := a.share.studio; st != nil {
		st.closet = open
		if open {
			a.share.wardrobe = readWardrobe(a.hatsDir())
		}
	}
}

// studioLoad puts a wardrobe hat on the board to change or wear.
func (a *app) studioLoad(item string) {
	st := a.share.studio
	if st == nil {
		return
	}
	st.keep()
	st.base, st.strokes, st.closet = item, nil, false
}

// studioWear puts the board's hat on the card's Voxt: a built-in or a
// wardrobe hat left as it is goes on as itself; anything drawn is saved
// as a new hat in the wardrobe first.
func (a *app) studioWear() {
	s := &a.share
	st := s.studio
	if st == nil {
		return
	}
	switch {
	case len(st.strokes) == 0:
		s.wearRef(st.base)
	default:
		name, err := a.saveHat(st.base, st.strokes)
		if err != nil {
			log.Printf("share: save the hat: %v", err)
			a.toast(i18n.T("Could not save the hat."))
			return
		}
		s.wardrobe = readWardrobe(a.hatsDir()) // the new hat first, without the dropped ones
		s.wearRef("hat:" + name)
	}
	s.studio = nil
}

// saveHat renders the board (its base and strokes, without the head's
// guide) at 2× into a PNG in the hats folder, and drops the oldest hats
// past shareWardrobeMax.
func (a *app) saveHat(base string, strokes []shareStroke) (string, error) {
	dir := a.hatsDir()
	if dir == "" {
		return "", fmt.Errorf("no library")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	a.shareBitmapNow(base) // the board's base, should it be still on its way
	var buf bytes.Buffer
	if err := png.Encode(&buf, renderHatPicture(base, strokes, a)); err != nil {
		return "", err
	}
	name := fmt.Sprintf("%019d.png", a.now().UnixNano())
	if err := os.WriteFile(filepath.Join(dir, name), buf.Bytes(), 0o644); err != nil {
		return "", err
	}
	if names := readWardrobe(dir); len(names) > shareWardrobeMax {
		for _, old := range names[shareWardrobeMax:] {
			_ = os.Remove(filepath.Join(dir, old))
		}
	}
	return name, nil
}
