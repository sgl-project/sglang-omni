//go:build linux && (amd64 || arm64)

package linux

import (
	"sync"

	"github.com/ebitengine/purego"

	"github.com/egoist/mygo/internal/platform"
)

// Input methods ask for the text around the caret, and delete some of it
// before they commit what replaces it, as Hangul and other input methods
// that edit what was typed do.

var (
	systemOnce                    sync.Once
	gtkIMContextSetSurrounding    func(im ptr, text *byte, length, cursor int32)
	gtkIMContextGetSurrounding    func(im ptr, text *ptr, cursor *int32) bool
	gtkIMContextDeleteSurrounding func(im ptr, offset, n int32) bool

	cbIMRetrieveSurrounding, cbIMDeleteSurrounding, cbSurfaceAreaDestroy ptr

	// surfaceAreas finds surfaces by their widget.
	surfaceAreas = map[ptr]*surface{}
)

// connectSystem connects the surface to the system's input methods, drag
// and drop, and assistive technology.
func (s *surface) connectSystem(data ptr) {
	systemOnce.Do(initSystemSurface)
	surfaceAreas[s.area] = s
	connect(s.area, "destroy", cbSurfaceAreaDestroy, data)
	connect(s.im, "retrieve-surrounding", cbIMRetrieveSurrounding, data)
	connect(s.im, "delete-surrounding", cbIMDeleteSurrounding, data)
	s.acceptFileDrops(data)
}

func initSystemSurface() {
	t := libGTK
	mustBind(t, &gtkIMContextSetSurrounding, "gtk_im_context_set_surrounding")
	mustBind(t, &gtkIMContextGetSurrounding, "gtk_im_context_get_surrounding")
	mustBind(t, &gtkIMContextDeleteSurrounding, "gtk_im_context_delete_surrounding")
	loadDrops()
	b := func() *Backend { return theBackend }
	cbIMRetrieveSurrounding = purego.NewCallback(func(im, data ptr) bool {
		s := b().surfaceOf(data)
		if s == nil || !s.input.Active {
			return false
		}
		text, cursor := s.surrounding()
		gtkIMContextSetSurrounding(im, cs(text), int32(len(text)), int32(cursor))
		return true
	})
	cbIMDeleteSurrounding = purego.NewCallback(func(im ptr, offset, n int32, data ptr) bool {
		s := b().surfaceOf(data)
		if s == nil || !s.input.Active || n < 0 {
			return false
		}
		if client := s.input.Client; client != nil {
			r := platform.ClientDeleteRange(client, int(offset), int(n))
			client.ReplaceText(&r, "")
			return true
		}
		from := s.input.End + int(offset)
		s.send(platform.SurfaceEvent{Kind: platform.TextInput, Replace: true, From: from, To: from + int(n)})
		return true
	})
	cbSurfaceAreaDestroy = purego.NewCallback(func(widget, data ptr) {
		if s := surfaceAreas[widget]; s != nil {
			delete(surfaceAreas, widget)
			s.destroyAccess()
		}
	})
}

// surrounding returns the text around the caret that input methods see,
// and the caret's byte offset in it: the end of the selection.
func (s *surface) surrounding() (string, int) {
	if c := s.input.Client; c != nil {
		context := platform.ClientTextContext(c)
		return context.Text, platform.UTF16ByteOffset(context.Text, context.Caret)
	}
	text, cursor := s.input.Text, 0
	for i := range text {
		if cursor == s.input.End {
			return text, i
		}
		cursor++
	}
	return text, len(text)
}
