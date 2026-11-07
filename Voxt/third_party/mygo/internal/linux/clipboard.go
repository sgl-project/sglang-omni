//go:build linux && (amd64 || arm64)

package linux

import (
	"bytes"
	"errors"
	"slices"
	"strings"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/egoist/mygo/internal/platform"
	"github.com/egoist/mygo/transfer"
)

var (
	gtkClipboardSetWithData                            func(ptr, unsafe.Pointer, uint32, ptr, ptr, ptr) bool
	gtkClipboardSetCanStore                            func(ptr, unsafe.Pointer, int32)
	gtkClipboardStore                                  func(ptr)
	gdkClipboardPersistence                            func(ptr) bool
	cbClipboardGet, cbClipboardClear, cbClipboardOwner ptr
	gtkClipboardSources                                = map[ptr]*gtkClipboardSource{}
	gtkClipboardCurrent                                *gtkClipboardSource
	clipboardID                                        ptr
	clipboardGeneration                                uint64
	clipboardConnected                                 bool
)

type gtkClipboardSource struct {
	id       ptr
	data     transfer.Data
	released func()
}

func initClipboardCallbacks() {
	// Clipboard-only apps may never create a surface or initialize its drag
	// adapter. Bind the shared selection primitives independently of it.
	mustBind(libGTK, &gtkSelectionDataSet, "gtk_selection_data_set")
	mustBind(libGTK, &gtkSelectionDataTarget, "gtk_selection_data_get_target")
	mustBind(libGTK, &gtkClipboardSetWithData, "gtk_clipboard_set_with_data")
	mustBind(libGTK, &gtkClipboardSetCanStore, "gtk_clipboard_set_can_store")
	mustBind(libGTK, &gtkClipboardStore, "gtk_clipboard_store")
	mustBind(libGDK, &gdkClipboardPersistence, "gdk_display_supports_clipboard_persistence")
	cbClipboardGet = purego.NewCallback(func(board, sel ptr, info uint32, data ptr) {
		s := gtkClipboardSources[data]
		if s == nil {
			return
		}
		atom := gtkSelectionDataTarget(sel)
		f := gtkFormat(takeStr(gdkAtomName(atom)))
		b, err := s.data.Read(f)
		if errors.Is(err, transfer.ErrFormat) && f == transfer.URIList {
			b, err = s.data.Read(transfer.FileList)
		}
		if err != nil {
			return
		}
		if len(b) == 0 {
			b = []byte{0}
			gtkSelectionDataSet(sel, atom, 8, &b[0], 0)
		} else {
			gtkSelectionDataSet(sel, atom, 8, &b[0], int32(len(b)))
		}
	})
	cbClipboardClear = purego.NewCallback(func(board, data ptr) {
		if s := gtkClipboardSources[data]; s != nil {
			delete(gtkClipboardSources, data)
			if gtkClipboardCurrent == s {
				gtkClipboardCurrent = nil
			}
			s.data = transfer.Data{}
			s.release()
		}
	})
	cbClipboardOwner = purego.NewCallback(func(board, event, data ptr) { clipboardGeneration++ })
}

func (s *gtkClipboardSource) release() {
	if fn := s.released; fn != nil {
		s.released = nil
		fn()
	}
}

func (c clipboard) WriteData(data transfer.Data, released func()) error {
	if len(data.Formats()) == 0 {
		c.Clear()
		if released != nil {
			released()
		}
		return nil
	}
	clipboardID++
	s := &gtkClipboardSource{id: clipboardID, data: data, released: released}
	gtkClipboardSources[s.id] = s
	targets := gtkTargets(data.Formats())
	if !gtkClipboardSetWithData(clip(), unsafe.Pointer(&targets[0]), uint32(len(targets)), cbClipboardGet, cbClipboardClear, s.id) {
		delete(gtkClipboardSources, s.id)
		return errors.New("mygo: cannot own the clipboard selection")
	}
	gtkClipboardCurrent = s
	gtkClipboardSetCanStore(clip(), nil, 0)
	return nil
}

func clipboardTargets() ([]transfer.Format, map[transfer.Format]ptr) {
	var atoms ptr
	var n int32
	out := []transfer.Format{}
	targets := map[transfer.Format]ptr{}
	if !gtkClipboardWaitForTargets(clip(), &atoms, &n) || atoms == 0 {
		return out, targets
	}
	defer gFree(atoms)
	if n < 0 || n > 4096 {
		return out, targets
	}
	for i := range int(n) {
		atom := field[ptr](atoms, uintptr(i)*unsafe.Sizeof(ptr(0)))
		name := takeStr(gdkAtomName(atom))
		if name == "" || strings.ContainsAny(name, "\x00\r\n") {
			continue
		}
		switch name {
		case "TARGETS", "MULTIPLE", "TIMESTAMP", "SAVE_TARGETS":
			continue
		}
		f := gtkFormat(name)
		if f == platform.DragSessionFormat {
			continue
		}
		if !slices.Contains(out, f) {
			out = append(out, f)
		}
		// Prefer canonical MIME targets over aliases where both are offered.
		if targets[f] == 0 || name == string(f) {
			targets[f] = atom
		}
		if f == transfer.URIList && targets[transfer.FileList] == 0 {
			targets[transfer.FileList] = atom
			if !slices.Contains(out, transfer.FileList) {
				out = append(out, transfer.FileList)
			}
		}
		if strings.HasPrefix(name, "image/") && !slices.Contains(out, transfer.PNG) {
			out = append(out, transfer.PNG)
		}
	}
	return out, targets
}

func (clipboard) Formats() []transfer.Format {
	formats, _ := clipboardTargets()
	if s := gtkClipboardCurrent; s != nil {
		return s.data.Formats()
	}
	return formats
}

func (c clipboard) ReadData(formats []transfer.Format) (transfer.Data, error) {
	generation := clipboardGeneration
	offered, targets := clipboardTargets()
	if generation != clipboardGeneration {
		return transfer.Data{}, platform.ErrClipboardChanged
	}
	if s := gtkClipboardCurrent; s != nil {
		if len(formats) == 0 {
			formats = s.data.Formats()
		}
		d, err := s.data.Materialize(formats)
		if generation != clipboardGeneration || gtkClipboardCurrent != s {
			return transfer.Data{}, platform.ErrClipboardChanged
		}
		return d, err
	}
	if len(formats) == 0 {
		formats = offered
		if len(formats) == 0 {
			return transfer.Data{}, nil
		}
	}
	var reps []transfer.Representation
	for n, f := range formats {
		if slices.Contains(formats[:n], f) || !slices.Contains(offered, f) {
			continue
		}
		var b []byte
		if f == transfer.PNG && targets[f] == 0 {
			b = c.ReadImage()
			if b == nil {
				return transfer.Data{}, errors.New("mygo: cannot convert clipboard image")
			}
		} else {
			sd := gtkClipboardWaitForContents(clip(), targets[f])
			if sd == 0 {
				return transfer.Data{}, transfer.ErrFormat
			}
			length, p := gtkSelectionDataGetLength(sd), gtkSelectionDataGetData(sd)
			if length < 0 || length > 64<<20 || (length != 0 && p == 0) {
				gtkSelectionDataFree(sd)
				return transfer.Data{}, errors.New("mygo: invalid or oversized clipboard representation")
			}
			b = bytes.Clone(unsafe.Slice(*(**byte)(unsafe.Pointer(&p)), int(length)))
			gtkSelectionDataFree(sd)
		}
		if generation != clipboardGeneration {
			return transfer.Data{}, platform.ErrClipboardChanged
		}
		reps = append(reps, transfer.Bytes(f, b))
	}
	if len(reps) == 0 {
		return transfer.Data{}, transfer.ErrFormat
	}
	return transfer.New(transfer.NewItem(reps...)), nil
}

func (clipboard) Flush() error {
	s := gtkClipboardCurrent
	if s == nil {
		return nil
	}
	generation := clipboardGeneration
	d, err := s.data.Materialize(s.data.Formats())
	if err != nil {
		return err
	}
	if gtkClipboardCurrent != s || generation != clipboardGeneration {
		return platform.ErrClipboardChanged
	}
	// GTK still serves eager bytes if no clipboard manager is available;
	// provider closures and their resources can be released in either case.
	s.data = d
	s.release()
	if !gdkClipboardPersistence(gdkDisplayGetDefault()) {
		return platform.ErrClipboardPersistence
	}
	gtkClipboardStore(clip())
	return nil
}

func (c clipboard) Close() {
	if gtkClipboardCurrent != nil {
		c.Clear()
	}
}
