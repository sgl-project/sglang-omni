//go:build linux && (amd64 || arm64)

package linux

import (
	"bytes"
	"errors"
	"image"
	"image/png"
	"slices"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/egoist/mygo/internal/platform"
	"github.com/egoist/mygo/transfer"
)

// Selection data arrives asynchronously. Only accepted formats are read
// after drop, before acknowledging the negotiated effect.
var (
	gtkDragDestSet                                                                func(ptr, int32, unsafe.Pointer, int32, int32)
	gtkDragGetData                                                                func(ptr, ptr, ptr, uint32)
	gtkDragFinish                                                                 func(ptr, bool, bool, uint32)
	gdkDragStatus                                                                 func(ptr, int32, uint32)
	gtkTargetListNew                                                              func(unsafe.Pointer, uint32) ptr
	gtkTargetListUnref                                                            func(ptr)
	gtkDragBeginCoordinates                                                       func(ptr, ptr, int32, int32, ptr, int32, int32) ptr
	gtkDragCancel                                                                 func(ptr)
	gtkDragSetIconPixbuf                                                          func(ptr, ptr, int32, int32)
	gtkDragGetSourceWidget                                                        func(ptr) ptr
	gdkDragContextTargets                                                         func(ptr) ptr
	gdkDragContextActions                                                         func(ptr) int32
	gdkDragContextSuggested                                                       func(ptr) int32
	gdkDragContextSelected                                                        func(ptr) int32
	gtkSelectionDataSet                                                           func(ptr, ptr, int32, *byte, int32)
	gtkSelectionDataTarget                                                        func(ptr) ptr
	cbSurfaceDragMotion, cbSurfaceDragLeave, cbSurfaceDragDrop, cbSurfaceDragData ptr
	cbSurfaceDragGet, cbSurfaceDragEnd, cbSurfaceDragFailed, cbSurfaceDragDelete  ptr
	gtkDataSources                                                                = map[ptr]*surface{}
)

const (
	gdkActionCopy = 1 << 1
	gdkActionMove = 1 << 2
)

type gtkTargetEntry struct {
	target      *byte
	flags, info uint32
}
type gtkDataDrop struct {
	context ptr
	event   *platform.DataDragEvent
	x, y    float64
	time    uint32
	pending map[ptr][]transfer.Format
	reps    []transfer.Representation
	failed  bool
}

func loadDrops() {
	mustBind(libGTK, &gtkDragDestSet, "gtk_drag_dest_set")
	mustBind(libGTK, &gtkDragGetData, "gtk_drag_get_data")
	mustBind(libGTK, &gtkDragFinish, "gtk_drag_finish")
	mustBind(libGDK, &gdkDragStatus, "gdk_drag_status")
	mustBind(libGTK, &gtkTargetListNew, "gtk_target_list_new")
	mustBind(libGTK, &gtkTargetListUnref, "gtk_target_list_unref")
	mustBind(libGTK, &gtkDragBeginCoordinates, "gtk_drag_begin_with_coordinates")
	mustBind(libGTK, &gtkDragCancel, "gtk_drag_cancel")
	mustBind(libGTK, &gtkDragSetIconPixbuf, "gtk_drag_set_icon_pixbuf")
	mustBind(libGTK, &gtkDragGetSourceWidget, "gtk_drag_get_source_widget")
	mustBind(libGDK, &gdkDragContextTargets, "gdk_drag_context_list_targets")
	mustBind(libGDK, &gdkDragContextActions, "gdk_drag_context_get_actions")
	mustBind(libGDK, &gdkDragContextSuggested, "gdk_drag_context_get_suggested_action")
	mustBind(libGDK, &gdkDragContextSelected, "gdk_drag_context_get_selected_action")
	mustBind(libGTK, &gtkSelectionDataSet, "gtk_selection_data_set")
	mustBind(libGTK, &gtkSelectionDataTarget, "gtk_selection_data_get_target")
	b := func() *Backend { return theBackend }
	cbSurfaceDragMotion = purego.NewCallback(func(widget, context ptr, x, y int32, time uint32, data ptr) bool {
		s := b().surfaceOf(data)
		if s == nil {
			return false
		}
		d, _ := gtkOffer(context)
		taken := s.send(platform.SurfaceEvent{Kind: platform.DataDragOver, X: float64(x), Y: float64(y), Drag: d})
		gdkDragStatus(context, gtkOperations(d.Operation), time)
		return taken
	})
	cbSurfaceDragLeave = purego.NewCallback(func(widget, context ptr, time uint32, data ptr) {
		if s := b().surfaceOf(data); s != nil {
			s.send(platform.SurfaceEvent{Kind: platform.DataDragLeave, Drag: &platform.DataDragEvent{}})
		}
	})
	cbSurfaceDragDrop = purego.NewCallback(func(widget, context ptr, x, y int32, time uint32, data ptr) bool {
		s := b().surfaceOf(data)
		if s == nil {
			return false
		}
		d, targets := gtkOffer(context)
		if !s.send(platform.SurfaceEvent{Kind: platform.DataDragOver, X: float64(x), Y: float64(y), Drag: d}) {
			return false
		}
		if len(d.Data.Formats()) > 0 {
			// A recognized local source preserves item boundaries and typed values.
			taken := s.send(platform.SurfaceEvent{Kind: platform.DataDrop, X: float64(x), Y: float64(y), Drag: d})
			gtkDragFinish(context, taken, taken && d.Operation == transfer.Move, time)
			return true
		}
		s.cancelPendingDrop()
		drop := &gtkDataDrop{context: gObjectRef(context), event: d, x: float64(x), y: float64(y), time: time, pending: map[ptr][]transfer.Format{}}
		for _, f := range d.Formats {
			if atom := targets[f]; atom != 0 {
				drop.pending[atom] = append(drop.pending[atom], f)
			}
		}
		if len(drop.pending) == 0 {
			gtkDragFinish(context, false, false, time)
			gObjectUnref(drop.context)
			return true
		}
		s.dataDrop = drop
		for atom := range drop.pending {
			gtkDragGetData(widget, context, atom, time)
		}
		return true
	})
	cbSurfaceDragData = purego.NewCallback(func(widget, context ptr, x, y int32, sel ptr, info, time uint32, data ptr) {
		s := b().surfaceOf(data)
		if s == nil || s.dataDrop == nil || s.dataDrop.context != context {
			return
		}
		drop := s.dataDrop
		atom := gtkSelectionDataTarget(sel)
		formats, exists := drop.pending[atom]
		if !exists {
			return
		}
		delete(drop.pending, atom)
		n, p := gtkSelectionDataGetLength(sel), gtkSelectionDataGetData(sel)
		if n < 0 || n > 64<<20 || (n > 0 && p == 0) {
			drop.failed = true
		} else {
			var buf []byte
			if n > 0 {
				buf = bytes.Clone(unsafe.Slice(*(**byte)(unsafe.Pointer(&p)), n))
			}
			for _, f := range formats {
				if f == transfer.FileList {
					files, _ := transfer.New(transfer.NewItem(transfer.Bytes(transfer.URIList, buf))).Files()
					if len(files) == 0 {
						continue
					}
				}
				drop.reps = append(drop.reps, transfer.Bytes(f, buf))
			}
		}
		if len(drop.pending) != 0 {
			return
		}
		s.dataDrop = nil
		taken := false
		if !drop.failed {
			drop.event.Data = transfer.New(transfer.NewItem(drop.reps...))
			taken = s.send(platform.SurfaceEvent{Kind: platform.DataDrop, X: drop.x, Y: drop.y, Drag: drop.event})
		}
		gtkDragFinish(context, taken, taken && drop.event.Operation == transfer.Move, drop.time)
		gObjectUnref(drop.context)
	})
	cbSurfaceDragGet = purego.NewCallback(func(widget, context, sel ptr, info, time uint32, data ptr) {
		s := b().surfaceOf(data)
		if s == nil || s.dragRequest == nil {
			return
		}
		r, atom := s.dragRequest, gtkSelectionDataTarget(sel)
		f := gtkFormat(takeStr(gdkAtomName(atom)))
		var buf []byte
		var err error
		if f == platform.DragSessionFormat {
			buf = []byte(r.Session)
		} else {
			buf, err = r.Data.Read(f)
		}
		if err != nil {
			s.dragError = err
			return
		}
		if len(buf) == 0 {
			buf = []byte{0}
			gtkSelectionDataSet(sel, atom, 8, &buf[0], 0)
		} else {
			gtkSelectionDataSet(sel, atom, 8, &buf[0], int32(len(buf)))
		}
	})
	cbSurfaceDragFailed = purego.NewCallback(func(widget, context ptr, result int32, data ptr) bool {
		s := b().surfaceOf(data)
		if s == nil {
			s = gtkDataSources[widget]
		}
		if s != nil && s.dragRequest != nil {
			// GTK_NO_TARGET=1 and USER_CANCELLED=2 are normal no-drop
			// results. gtk_drag_cancel can also abort with a generic error;
			// our explicit cancellation must remain a cancellation.
			explicit := s.dragCanceled
			canceled := explicit || result == 1 || result == 2
			s.dragCanceled = true
			if !canceled && result > 2 {
				s.dragError = errors.New("mygo: GTK drag failed")
			}
			// An explicit cancel (including destruction) must release the
			// tracker immediately, without a snap-back animation holding
			// the input grab after the app reports cancellation.
			return explicit
		}
		return false
	})
	cbSurfaceDragEnd = purego.NewCallback(func(widget, context, data ptr) {
		s := b().surfaceOf(data)
		if s == nil {
			s = gtkDataSources[widget]
		}
		if s == nil || s.dragRequest == nil {
			return
		}
		r := s.dragRequest
		op := goGTKOperations(gdkDragContextSelected(context))
		if s.dragCanceled || s.dragError != nil {
			op = transfer.None
		}
		delete(gtkDataSources, widget)
		s.dragRequest, s.dragContext = nil, 0
		if s.lastPointer != 0 {
			gdkEventFree(s.lastPointer)
			s.lastPointer = 0
		}
		r.Done(transfer.Result{Operation: op, Canceled: op == transfer.None && s.dragError == nil, Err: s.dragError})
	})
	// The app's Done callback owns removal after a successful move.
	cbSurfaceDragDelete = purego.NewCallback(func(widget, context, data ptr) {})
}
func gtkOperations(o transfer.Operation) int32 {
	var n int32
	if o&transfer.Copy != 0 {
		n |= gdkActionCopy
	}
	if o&transfer.Move != 0 {
		n |= gdkActionMove
	}
	return n
}
func goGTKOperations(n int32) transfer.Operation {
	var o transfer.Operation
	if n&gdkActionCopy != 0 {
		o |= transfer.Copy
	}
	if n&gdkActionMove != 0 {
		o |= transfer.Move
	}
	return o
}
func gtkFormat(t string) transfer.Format {
	if t == "UTF8_STRING" || t == "text/plain;charset=utf-8" {
		return transfer.Text
	}
	return transfer.Format(t)
}
func gtkOffer(context ptr) (*platform.DataDragEvent, map[transfer.Format]ptr) {
	d := &platform.DataDragEvent{Offer: transfer.Offer{Operations: goGTKOperations(gdkDragContextActions(context)), Suggested: goGTKOperations(gdkDragContextSuggested(context))}}
	targets := map[transfer.Format]ptr{}
	for l := gdkDragContextTargets(context); l != 0; l = field[ptr](l, 8) {
		atom := field[ptr](l, 0)
		f := gtkFormat(takeStr(gdkAtomName(atom)))
		if f == platform.DragSessionFormat {
			continue
		}
		if !slices.Contains(d.Offer.Formats, f) {
			d.Offer.Formats = append(d.Offer.Formats, f)
		}
		targets[f] = atom
		if f == transfer.URIList {
			d.HasFiles = true // Existing GTK file drops also advertise URI lists.
			if !slices.Contains(d.Offer.Formats, transfer.FileList) {
				d.Offer.Formats = append(d.Offer.Formats, transfer.FileList)
			}
			targets[transfer.FileList] = atom
		}
	}
	if src := gtkDataSources[gtkDragGetSourceWidget(context)]; src != nil && src.dragRequest != nil {
		d.Session = src.dragRequest.Session
	}
	return d, targets
}
func gtkTargets(formats []transfer.Format) []gtkTargetEntry {
	var names []string
	for _, f := range formats {
		name := string(f)
		if f == transfer.FileList {
			name = string(transfer.URIList)
		}
		if !slices.Contains(names, name) {
			names = append(names, name)
		}
		if f == transfer.Text {
			names = append(names, "UTF8_STRING", "text/plain;charset=utf-8")
		}
	}
	targets := make([]gtkTargetEntry, len(names))
	for n, name := range names {
		targets[n] = gtkTargetEntry{target: cs(name), info: uint32(n)}
	}
	return targets
}
func (s *surface) SetDropFormats(formats []transfer.Format) {
	targets := gtkTargets(formats)
	if len(targets) != 0 {
		gtkDragDestSet(s.area, 0, unsafe.Pointer(&targets[0]), int32(len(targets)), gdkActionCopy|gdkActionMove)
	}
}
func (s *surface) acceptFileDrops(data ptr) {
	s.SetDropFormats([]transfer.Format{transfer.URIList})
	connect(s.area, "drag-motion", cbSurfaceDragMotion, data)
	connect(s.area, "drag-leave", cbSurfaceDragLeave, data)
	connect(s.area, "drag-drop", cbSurfaceDragDrop, data)
	connect(s.area, "drag-data-received", cbSurfaceDragData, data)
	connect(s.area, "drag-data-get", cbSurfaceDragGet, data)
	connect(s.area, "drag-end", cbSurfaceDragEnd, data)
	connect(s.area, "drag-failed", cbSurfaceDragFailed, data)
	connect(s.area, "drag-data-delete", cbSurfaceDragDelete, data)
}
func (s *surface) StartDataDrag(r platform.DragRequest) {
	if s.dragRequest != nil || s.lastPointer == 0 {
		r.Done(transfer.Result{Err: errors.New("mygo: GTK drag needs a pointer gesture")})
		return
	}
	targets := gtkTargets(append(r.Data.Formats(), platform.DragSessionFormat))
	list := gtkTargetListNew(unsafe.Pointer(&targets[0]), uint32(len(targets)))
	defer gtkTargetListUnref(list)
	s.dragRequest, s.dragError, s.dragCanceled = &r, nil, false
	gtkDataSources[s.area] = s
	s.dragContext = gtkDragBeginCoordinates(s.area, list, gtkOperations(r.Operations), 1, s.lastPointer, int32(r.X), int32(r.Y))
	if s.dragContext == 0 {
		delete(gtkDataSources, s.area)
		s.dragRequest = nil
		r.Done(transfer.Result{Err: errors.New("mygo: GTK failed to start drag")})
		return
	}
	if len(r.Preview) > 0 {
		pix := bytes.Clone(r.Preview)
		for i := 0; i < len(pix); i += 4 {
			pix[i], pix[i+2] = pix[i+2], pix[i]
		}
		var buf bytes.Buffer
		png.Encode(&buf, &image.RGBA{Pix: pix, Stride: r.Width * 4, Rect: image.Rect(0, 0, r.Width, r.Height)})
		if pb, err := pixbufFromPNG(buf.Bytes()); err == nil {
			gtkDragSetIconPixbuf(s.dragContext, pb, int32(r.HotX), int32(r.HotY))
			gObjectUnref(pb)
		}
	}
}
func (s *surface) CancelDataDrag() {
	if s.dragContext != 0 {
		s.dragCanceled = true
		gtkDragCancel(s.dragContext)
	}
}
func (s *surface) cancelPendingDrop() {
	if d := s.dataDrop; d != nil {
		s.dataDrop = nil
		gtkDragFinish(d.context, false, false, d.time)
		gObjectUnref(d.context)
	}
}
func (s *surface) fileDragOver(x, y float64) bool {
	return s.send(platform.SurfaceEvent{Kind: platform.FileDragOver, X: x, Y: y})
}
func (s *surface) dropFiles(x, y float64, paths []string) bool {
	return s.send(platform.SurfaceEvent{Kind: platform.FileDrop, X: x, Y: y, Files: paths})
}
func selectionPaths(sel ptr) []string {
	uris := gtkSelectionDataGetUris(sel)
	if uris == 0 {
		return nil
	}
	defer gStrfreev(uris)
	var paths []string
	for i := uintptr(0); ; i++ {
		uri := field[ptr](uris, i*8)
		if uri == 0 {
			break
		}
		if p := takeStr(gFilenameFromURI(goStr(uri), 0, 0)); p != "" {
			paths = append(paths, p)
		}
	}
	return paths
}
