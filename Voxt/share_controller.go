package main

// The share card's controller (share_card.go): what the card keeps while
// Voxt runs (its background, its hat, the last sentence said to it), the
// try that records that sentence, and Copy, which renders the card
// without its buttons and puts the picture on the clipboard. Drawn hats
// live in the library's "hats" folder (share_studio_controller.go).

import (
	"bytes"
	"fmt"
	"image"
	"image/png"
	"log"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/transfer"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

// shareCopiedFor is how long Copy shows its tick.
const shareCopiedFor = 1400 * time.Millisecond

// shareOutfits are the built-in hats the card and the wardrobe offer.
var shareOutfits = []string{outfitSpring, outfitChristmas, outfitNewYear}

type shareState struct {
	share shareKeep
}

type shareKeep struct {
	open       bool
	background int
	// outfit is a built-in hat, hat a drawn one (its file's name); dressed
	// counts the hats put on.
	outfit, hat string
	dressed     uint64
	// The last sentence said to the card: kept while Voxt runs.
	text  string
	audio time.Duration
	said  time.Time
	// tryID is the card's try; taken the try whose sentence it took.
	tryID, taken string
	block        *sessionNotice
	copiedAt     time.Time
	studio       *studioKeep
	// wardrobe is the drawn hats' names, newest first; bitmaps the
	// decoded hats and pictures, by reference.
	wardrobe []string
	bitmaps  map[string]*sharePicture
	loading  map[string]bool // read off the main thread now
	// insights is the insights card (insights_controller.go).
	insights insightsShare
}

// shareClipboard puts a PNG on the clipboard; tests replace it.
var shareClipboard = func(data []byte) error {
	return mygo.Clipboard.Write(transfer.New(transfer.NewItem(transfer.Bytes(transfer.PNG, data))))
}

// openShare opens the card: a random background, and today's outfit or
// a random hat when it wears none yet.
func (a *app) openShare() {
	s := &a.share
	if s.open {
		return
	}
	s.open = true
	s.background = rand.IntN(len(shareBackgrounds))
	s.wardrobe = readWardrobe(a.hatsDir())
	if s.outfit == "" && s.hat == "" {
		s.outfit = mascotOutfitOn(a.now())
		if s.outfit == "" {
			s.outfit = shareOutfits[rand.IntN(len(shareOutfits))]
		}
		s.dressed++
	}
}

func (a *app) closeShare() {
	s := &a.share
	if s.tryID != "" && a.sess.ID == s.tryID && a.sess.State.Live() {
		a.cancelSession() // the card's recording goes with it
	}
	s.open, s.studio, s.block = false, nil, nil
}

// shuffleShare picks another background and another hat, among the
// built-in ones and the drawn ones.
func (a *app) shuffleShare() {
	s := &a.share
	if n := len(shareBackgrounds); n > 1 {
		s.background = (s.background + 1 + rand.IntN(n-1)) % n
	}
	hats := make([]string, 0, len(shareOutfits)+len(s.wardrobe))
	for _, o := range shareOutfits {
		hats = append(hats, "outfit:"+o)
	}
	for _, h := range s.wardrobe {
		hats = append(hats, "hat:"+h)
	}
	current := "outfit:" + s.outfit
	if s.hat != "" {
		current = "hat:" + s.hat
	}
	if i := slices.Index(hats, current); i >= 0 && len(hats) > 1 {
		hats = slices.Delete(hats, i, i+1)
	}
	s.wearRef(hats[rand.IntN(len(hats))])
}

// wearRef puts on a hat by its reference ("outfit:<key>" or
// "hat:<name>"); "" takes the hat off.
func (s *shareKeep) wearRef(ref string) {
	s.outfit, s.hat = "", ""
	if key, ok := strings.CutPrefix(ref, "outfit:"); ok {
		s.outfit = key
	} else if name, ok := strings.CutPrefix(ref, "hat:"); ok {
		s.hat = name
	}
	s.dressed++
}

// startShareTry records a Transcribe try for the card.
func (a *app) startShareTry() {
	id, notice := a.startTry(featureKind(hotkey.Transcribe))
	a.shareTryStarted(id, notice)
}

// shareTryStarted keeps the try a click or a shortcut started for the
// card ("" when it did not: notice says why).
func (a *app) shareTryStarted(id string, notice *sessionNotice) {
	if id != "" {
		a.share.tryID = id
	}
	a.share.block = notice
}

func (a *app) stopShareTry() { a.stopSession() }

// shareModel is the card's view state; a finished try of the card's
// leaves its sentence first.
func (a *app) shareModel(now time.Time) shareCardViewState {
	s := &a.share
	mine := s.tryID != "" && a.sess.ID == s.tryID
	if mine && s.taken != a.sess.ID && a.sess.State.Ended() {
		s.taken = a.sess.ID
		if text, audio, said, ok := shareSentence(a.sess); ok {
			s.text, s.audio, s.said = text, audio, said
		}
	}
	notice := ""
	if s.block != nil {
		notice = s.block.text
	}
	return buildShareCardViewState(i18n.CurrentTranslator(), shareCardInput{
		Background: s.background, Outfit: s.outfit, Hat: s.hat, Dressed: s.dressed,
		Text: s.text, Audio: s.audio, Said: s.said, Mine: mine, Snapshot: a.sess, Now: now,
		Copied: !s.copiedAt.IsZero() && now.Sub(s.copiedAt) < shareCopiedFor, Notice: notice,
		Studio: s.studioView(),
	})
}

// shareDialog shows the card over the main window while it is open.
func (a *app) shareDialog(c *uiContext) {
	if !a.share.open {
		return
	}
	state := a.shareModel(a.now())
	shareCardDialog(c, &state, a)
}

// copyShare renders the card without its buttons and copies the picture.
func (a *app) copyShare() {
	state := a.shareModel(a.now())
	state.Studio = nil
	if a.share.hat != "" {
		a.shareBitmapNow("hat:" + a.share.hat)
	}
	if err := copyPicture(renderShareCard(&state, a)); err != nil {
		log.Printf("share: copy the card: %v", err)
		a.toast(i18n.T("Could not copy the picture."))
		return
	}
	a.share.copiedAt = a.now()
}

func copyPicture(img image.Image) error {
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return err
	}
	return shareClipboard(buf.Bytes())
}

// shareBitmap is a drawn hat, read and decoded off the main thread: nil
// until it is ready (the window redraws then), and nil for good when it
// cannot be read.
func (a *app) shareBitmap(ref string) *sharePicture {
	s := &a.share
	if bm, ok := s.bitmaps[ref]; ok {
		return bm
	}
	if s.loading[ref] {
		return nil
	}
	if s.loading == nil {
		s.loading = map[string]bool{}
	}
	s.loading[ref] = true
	path := a.hatPath(ref)
	a.async(func() {
		bm, _ := loadHat(path)
		a.post(func() {
			delete(s.loading, ref)
			s.keepBitmap(ref, bm)
		})
	})
	return s.bitmaps[ref] // tests: async and post ran at once
}

// shareBitmapNow is shareBitmap for Copy and Put on, which render at once.
func (a *app) shareBitmapNow(ref string) *sharePicture {
	if bm, ok := a.share.bitmaps[ref]; ok {
		return bm
	}
	path := a.hatPath(ref)
	if path == "" {
		return nil
	}
	bm, _ := loadHat(path)
	a.share.keepBitmap(ref, bm)
	return bm
}

func (s *shareKeep) keepBitmap(ref string, bm *sharePicture) {
	if s.bitmaps == nil {
		s.bitmaps = map[string]*sharePicture{}
	}
	s.bitmaps[ref] = bm
}

// hatPath is the file of a drawn hat ("hat:<name>"), "" for anything else.
func (a *app) hatPath(ref string) string {
	if name, ok := strings.CutPrefix(ref, "hat:"); ok && a.hatsDir() != "" {
		return filepath.Join(a.hatsDir(), filepath.Base(name))
	}
	return ""
}

// shareHatSideMax bounds a file in the hats folder: Put on saves the hat's
// box at 2× (660×540), so anything far larger is not a hat Voxt saved.
const shareHatSideMax = 2048

// loadHat reads and decodes a drawn hat. It runs off the main thread.
func loadHat(path string) (*sharePicture, error) {
	if path == "" {
		return nil, os.ErrNotExist
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	cfg, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	if cfg.Width > shareHatSideMax || cfg.Height > shareHatSideMax {
		return nil, fmt.Errorf("%s is %d×%d, not a drawn hat", filepath.Base(path), cfg.Width, cfg.Height)
	}
	return decodeSharePicture(data)
}

// hatsDir is where drawn hats are kept, "" without a library (tests).
func (a *app) hatsDir() string {
	if a.lib == nil || a.lib.store == nil {
		return ""
	}
	return filepath.Join(a.lib.store.Dir(), "hats")
}

// readWardrobe lists the drawn hats in dir, newest first: their names are
// the times they were drawn.
func readWardrobe(dir string) []string {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	var names []string
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".png") {
			names = append(names, e.Name())
		}
	}
	slices.Sort(names)
	slices.Reverse(names)
	return names
}
