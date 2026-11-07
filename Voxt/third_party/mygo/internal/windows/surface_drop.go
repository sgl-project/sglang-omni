//go:build windows && (amd64 || arm64)

package windows

import (
	"sync"
	"syscall"
	"unsafe"

	"github.com/egoist/mygo/internal/platform"
	"github.com/egoist/mygo/transfer"
)

// Files dragged over native UI come through OLE drag and drop: the
// surface's window is a drop target, which tells the source whether the
// content takes the files where they are. Over the page of a web view
// that nothing the content painted covers, the page takes the drag
// instead, through its composition controller, as it would its own.

var (
	procOleInitialize    = ole32.NewProc("OleInitialize")
	procRegisterDragDrop = ole32.NewProc("RegisterDragDrop")
	procRevokeDragDrop   = ole32.NewProc("RevokeDragDrop")
	procReleaseStgMedium = ole32.NewProc("ReleaseStgMedium")
	procDragQueryFileW   = shell32.NewProc("DragQueryFileW")

	dropOnce       sync.Once
	dropTargetVtbl [7]uintptr
	dropTargets    = map[uintptr]*dropTarget{}
	iidIDropTarget = guid("00000122-0000-0000-c000-000000000046")
	iidIUnknown    = guid("00000000-0000-0000-c000-000000000046")
)

const (
	cfHDrop          = 15
	dvaspectContent  = 1
	tymedHGlobal     = 1
	dropEffectNone   = 0
	dropEffectCopy   = 1
	dropEffectMove   = 2
	dropEffectLink   = 4
	dataGetData      = 3 // IDataObject::GetData
	dataQueryGetData = 5 // IDataObject::QueryGetData
)

// dropTarget is the IDropTarget of a surface.
type dropTarget struct {
	vtbl       *[7]uintptr
	refs       int32
	s          *surface
	data       uintptr
	helper     uintptr
	drag       *platform.DataDragEvent
	formats    map[transfer.Format]uint16
	legacyLink bool
	// page is the web view whose page has the drag.
	page *window
}

// formatEtc is FORMATETC.
type formatEtc struct {
	Format uint16
	Device uintptr
	Aspect uint32
	Index  int32
	Tymed  uint32
}

// stgMedium is STGMEDIUM.
type stgMedium struct {
	Tymed         uint32
	Handle        uintptr
	UnkForRelease uintptr
}

var hdropFormat = formatEtc{Format: cfHDrop, Aspect: dvaspectContent, Index: -1, Tymed: tymedHGlobal}

func initDropTarget() {
	procOleInitialize.Call(0)
	initDataSource()
	target := func(this uintptr) *dropTarget { return dropTargets[this] }
	dropTargetVtbl = [7]uintptr{
		syscall.NewCallback(func(this, riid, out uintptr) uintptr {
			if iid := *(*GUID)(native(riid)); iid != iidIUnknown && iid != iidIDropTarget {
				*(*uintptr)(native(out)) = 0
				return eNoInterface
			}
			*(*uintptr)(native(out)) = this
			target(this).refs++
			return sOK
		}),
		syscall.NewCallback(func(this uintptr) uintptr {
			t := target(this)
			t.refs++
			return uintptr(t.refs)
		}),
		syscall.NewCallback(func(this uintptr) uintptr {
			t := target(this)
			if t.refs--; t.refs == 0 {
				t.leave()
				release(t.helper)
				delete(dropTargets, this)
			}
			return uintptr(t.refs)
		}),
		// DragEnter(data, keys, point, effect)
		syscall.NewCallback(func(this, data, keys, pt, effect uintptr) uintptr {
			t := target(this)
			allowed := *(*uint32)(native(effect))
			t.enter(data, allowed, keys)
			if !t.overPage(keys, pt, effect) {
				t.over(keys, pt, effect)
			}
			t.helperEnter(data, pt, effect)
			return sOK
		}),
		// DragOver(keys, point, effect)
		syscall.NewCallback(func(this, keys, pt, effect uintptr) uintptr {
			t := target(this)
			if !t.overPage(keys, pt, effect) {
				t.over(keys, pt, effect)
			}
			if t.helper != 0 {
				p := point{int32(uint32(pt)), int32(uint32(pt >> 32))}
				comCall(t.helper, 5, uintptr(unsafe.Pointer(&p)), uintptr(*(*uint32)(native(effect))))
			}
			return sOK
		}),
		// DragLeave()
		syscall.NewCallback(func(this uintptr) uintptr {
			t := target(this)
			if t.helper != 0 {
				comCall(t.helper, 4)
			}
			if !t.leavePage() {
				t.s.send(platform.SurfaceEvent{Kind: platform.DataDragLeave, Drag: &platform.DataDragEvent{}})
			}
			t.leave()
			return sOK
		}),
		// Drop(data, keys, point, effect)
		syscall.NewCallback(func(this, data, keys, pt, effect uintptr) uintptr {
			t := target(this)
			if t.data != data {
				t.enter(data, *(*uint32)(native(effect)), keys)
			}
			if v := t.pageAt(pt); v != nil {
				if v != t.page {
					t.overPage(keys, pt, effect) // enters it
				}
				comCall(v.comp3, comp3Drop, data, keys, v.pagePoint(pt), effect)
				t.page = nil
			} else {
				t.leavePage()
				t.over(keys, pt, effect)
				t.drop(data, pt, effect)
			}
			if t.helper != 0 {
				p := point{int32(uint32(pt)), int32(uint32(pt >> 32))}
				comCall(t.helper, 6, data, uintptr(unsafe.Pointer(&p)), uintptr(*(*uint32)(native(effect))))
			}
			t.leave()
			return sOK
		}),
	}
}

// acceptFileDrops makes the surface's window a drop target.
func (s *surface) acceptFileDrops() {
	dropOnce.Do(initDropTarget)
	t := &dropTarget{vtbl: &dropTargetVtbl, refs: 1, s: s, helper: dragHelper(&iidIDropTargetHelper)}
	p := uintptr(unsafe.Pointer(t))
	dropTargets[p] = t
	procRegisterDragDrop.Call(s.hwnd, p) // takes its own reference
	s.dropTarget = p
}

func (s *surface) revokeFileDrops() {
	if s.dropTarget != 0 {
		procRevokeDragDrop.Call(s.hwnd)
		release(s.dropTarget)
		s.dropTarget = 0
	}
}

// overPage passes the drag at a point of the screen on to the page of the
// web view there, if one takes it, and reports whether one does: entering
// it, the content hears that the drag left, and leaving it, the page.
func (t *dropTarget) overPage(keys, pt, effect uintptr) bool {
	v := t.pageAt(pt)
	if v == t.page {
		if v == nil {
			return false
		}
		comCall(v.comp3, comp3DragOver, keys, v.pagePoint(pt), effect)
		return true
	}
	if !t.leavePage() {
		t.s.send(platform.SurfaceEvent{Kind: platform.DataDragLeave, Drag: &platform.DataDragEvent{}})
	}
	if v == nil {
		return false
	}
	t.page = v
	comCall(v.comp3, comp3DragEnter, t.data, keys, v.pagePoint(pt), effect)
	return true
}

// pageAt returns the web view whose page takes a drag at a point of the
// screen, if one does.
func (t *dropTarget) pageAt(pt uintptr) *window {
	if t.data == 0 {
		return nil
	}
	v := t.s.webViewAt(t.s.screenDIP(pt))
	if v == nil || v.comp3 == 0 { // a runtime older than drags into pages
		return nil
	}
	return v
}

// leavePage tells the page that had the drag that it left, and reports
// whether one had it.
func (t *dropTarget) leavePage() bool {
	v := t.page
	t.page = nil
	if v == nil {
		return false
	}
	if !v.closed && v.comp3 != 0 {
		comCall(v.comp3, comp3DragLeave)
	}
	return true
}

// over answers whether the content takes the files dragged to a point of
// the screen, by the effect it lets the drop have.
func (t *dropTarget) over(keys, pt, effect uintptr) {
	t.legacyLink = false
	allowed := *(*uint32)(native(effect))
	*(*uint32)(native(effect)) = dropEffectNone
	if t.drag == nil {
		return
	}
	d := t.drag
	d.Offer.Operations = transfer.Operation(allowed) & (transfer.Copy | transfer.Move)
	d.Offer.Suggested = transfer.None
	if keys&8 != 0 {
		d.Offer.Suggested = transfer.Copy
	} else if keys&4 != 0 {
		d.Offer.Suggested = transfer.Move
	}
	if x, y := t.s.screenDIP(pt); t.s.send(platform.SurfaceEvent{Kind: platform.DataDragOver, X: x, Y: y, Drag: d}) {
		*(*uint32)(native(effect)) = uint32(d.Operation)
	} else if d.HasFiles && allowed&dropEffectLink != 0 && allowed&(dropEffectCopy|dropEffectMove) == 0 && t.s.fileDragOver(x, y) {
		// Preserve the pre-existing file-only link fallback. Serialized
		// destinations and new sources continue to negotiate copy/move.
		t.legacyLink = true
		*(*uint32)(native(effect)) = dropEffectLink
	}
}

func (t *dropTarget) enter(data uintptr, allowed uint32, keys uintptr) {
	t.leave()
	addRef(data)
	t.data = data
	t.drag, t.formats = oleOffer(data, allowed, keys)
}
func (t *dropTarget) leave() {
	data := t.data
	t.data, t.drag, t.formats = 0, nil, nil
	release(data)
}
func (t *dropTarget) helperEnter(data, pt, effect uintptr) {
	if t.helper == 0 {
		return
	}
	p := point{int32(uint32(pt)), int32(uint32(pt >> 32))}
	comCall(t.helper, 3, t.s.hwnd, data, uintptr(unsafe.Pointer(&p)), uintptr(*(*uint32)(native(effect))))
}
func (t *dropTarget) drop(data, pt, effect uintptr) {
	if t.legacyLink {
		*(*uint32)(native(effect)) = 0
		x, y := t.s.screenDIP(pt)
		if paths := droppedPaths(data); len(paths) > 0 && t.s.dropFiles(x, y, paths) {
			*(*uint32)(native(effect)) = dropEffectLink
		}
		return
	}
	d := t.drag
	if d == nil || *(*uint32)(native(effect)) == 0 {
		return
	}
	*(*uint32)(native(effect)) = 0
	var nativeFiles []string
	if len(d.Data.Formats()) == 0 {
		var reps []transfer.Representation
		for _, f := range d.Formats {
			id := t.formats[f]
			if id == 0 {
				return
			}
			var b []byte
			if id == cfHDrop {
				if nativeFiles == nil {
					nativeFiles = droppedPaths(data)
				}
				if len(nativeFiles) == 0 {
					return
				}
				files, err := transfer.FileData(nativeFiles...)
				// Legacy file-drop APIs deliver CF_HDROP paths as received.
				// Their contract predates portable URL representations, so a
				// path that cannot be encoded must still reach those listeners.
				if err != nil {
					continue
				}
				b, _ = files.Read(f)
			} else {
				buf, err := oleDataBytes(data, id)
				if err != nil {
					return
				}
				b = buf
				switch id {
				case cfUnicodeText:
					b = []byte(oleText(buf))
				case uint16(cfHTML):
					text := string(buf)
					start, end := htmlOffset(text, "StartFragment:"), htmlOffset(text, "EndFragment:")
					if start < 0 || end < start || end > len(text) {
						return
					}
					b = buf[start:end]
				default:
					if id == uint16(registerClipboardFormat("UniformResourceLocatorW")) {
						b = []byte(oleText(buf))
					}
				}
			}
			reps = append(reps, transfer.Bytes(f, b))
		}
		d.Data = transfer.New(transfer.NewItem(reps...))
	}
	x, y := t.s.screenDIP(pt)
	if t.s.send(platform.SurfaceEvent{Kind: platform.DataDrop, X: x, Y: y, Drag: d, Files: nativeFiles}) {
		*(*uint32)(native(effect)) = uint32(d.Operation)
	}
}

// acceptEffect returns the effect of a drop the content takes, of those
// the source allows: never a move, which would delete the files.
func acceptEffect(allowed uint32) uint32 {
	switch {
	case allowed&dropEffectCopy != 0:
		return dropEffectCopy
	case allowed&dropEffectLink != 0:
		return dropEffectLink
	}
	return dropEffectNone
}

// screenDIP converts a POINTL of the screen, passed by value, to DIPs
// relative to the surface.
func (s *surface) screenDIP(pt uintptr) (float64, float64) {
	p := point{int32(uint32(pt)), int32(uint32(pt >> 32))}
	procScreenToClient.Call(s.hwnd, uintptr(unsafe.Pointer(&p)))
	return s.toDIP(p.X), s.toDIP(p.Y)
}

// fileDragOver reports whether the content takes files dragged to (x, y).
func (s *surface) fileDragOver(x, y float64) bool {
	return s.send(platform.SurfaceEvent{Kind: platform.FileDragOver, X: x, Y: y})
}

// dropFiles drops files at (x, y), and reports whether the content took
// them.
func (s *surface) dropFiles(x, y float64, paths []string) bool {
	return s.send(platform.SurfaceEvent{Kind: platform.FileDrop, X: x, Y: y, Files: paths})
}

// droppedPaths returns the paths of the files of a data object.
func droppedPaths(data uintptr) []string {
	format := hdropFormat
	var m stgMedium
	if comCall(data, dataGetData, uintptr(unsafe.Pointer(&format)), uintptr(unsafe.Pointer(&m))) != sOK {
		return nil
	}
	defer procReleaseStgMedium.Call(uintptr(unsafe.Pointer(&m)))
	return droppedHDropPaths(m.Handle)
}

func droppedHDropPaths(handle uintptr) []string {
	n, _, _ := procDragQueryFileW.Call(handle, 0xFFFFFFFF, 0, 0)
	var paths []string
	for i := range n {
		size, _, _ := procDragQueryFileW.Call(handle, i, 0, 0)
		buf := make([]uint16, size+1)
		procDragQueryFileW.Call(handle, i, uintptr(unsafe.Pointer(&buf[0])), size+1)
		if p := syscall.UTF16ToString(buf); p != "" {
			paths = append(paths, p)
		}
	}
	return paths
}
