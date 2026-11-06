//go:build windows && (amd64 || arm64)

package windows

import (
	"bytes"
	"encoding/binary"
	"errors"
	"image/png"
	"slices"
	"strings"
	"syscall"
	"unsafe"

	"github.com/egoist/mygo/internal/platform"
	"github.com/egoist/mygo/transfer"
)

const (
	tymedStream        = 4
	dataEnumFormatEtc  = 8
	dataSetData        = 7
	dvFormatEtc        = 0x80040064
	eNotImpl           = 0x80004001
	sFalse             = 1
	dragDropDrop       = 0x40100
	dragDropCancel     = 0x40101
	dragDefaultCursors = 0x40102
)

var (
	procDoDragDrop       = ole32.NewProc("DoDragDrop")
	procOleDuplicateData = ole32.NewProc("OleDuplicateData")
	iidIDataObject       = guid("0000010e-0000-0000-c000-000000000046")
	iidIDropSource       = guid("00000121-0000-0000-c000-000000000046")
	iidIEnumFormatEtc    = guid("00000103-0000-0000-c000-000000000046")
	iidIDragSourceHelper = guid("de5bf786-477a-11d2-839d-00c04fd918d0")
	iidIDropTargetHelper = guid("4657278b-411b-11d2-839a-00c04fd918d0")
	clsDragDropHelper    = guid("4657278a-411b-11d2-839a-00c04fd918d0")
	dragDataVtbl         [12]uintptr
	dragSourceVtbl       [5]uintptr
	formatEnumVtbl       [7]uintptr
	dragObjects          = map[uintptr]*dragDataObject{}
	dragSources          = map[uintptr]*oleDragSource{}
	formatEnums          = map[uintptr]*formatEnum{}
)

type dragDataObject struct {
	vtbl      *[12]uintptr
	refs      int32
	providers map[uint16]func() ([]byte, error)
	formats   []uint16
	extra     map[uint16]stgMedium
	err       error
	clipboard bool
	onRelease func()
}
type oleDragSource struct {
	vtbl     *[5]uintptr
	refs     int32
	canceled bool
}
type formatEnum struct {
	vtbl    *[7]uintptr
	refs    int32
	formats []formatEtc
	n       int
}

func initDataSource() {
	query := func(this, riid, out uintptr, want GUID) uintptr {
		*(*uintptr)(native(out)) = 0
		if iid := *(*GUID)(native(riid)); iid != iidIUnknown && iid != want {
			return eNoInterface
		}
		*(*uintptr)(native(out)) = this
		addRef(this)
		return sOK
	}
	notImpl := syscall.NewCallback(func(this, a, b uintptr) uintptr { return eNotImpl })
	dragDataVtbl = [12]uintptr{
		syscall.NewCallback(func(this, riid, out uintptr) uintptr { return query(this, riid, out, iidIDataObject) }),
		syscall.NewCallback(func(this uintptr) uintptr { d := dragObjects[this]; d.refs++; return uintptr(d.refs) }),
		syscall.NewCallback(func(this uintptr) uintptr {
			d := dragObjects[this]
			d.refs--
			if d.refs == 0 {
				delete(dragObjects, this)
				for _, m := range d.extra {
					procReleaseStgMedium.Call(uintptr(unsafe.Pointer(&m)))
				}
				d.providers, d.extra = nil, nil
				if fn := d.onRelease; fn != nil {
					d.onRelease = nil
					fn()
				}
			}
			return uintptr(d.refs)
		}),
		syscall.NewCallback(func(this, format, medium uintptr) uintptr {
			d := dragObjects[this]
			f := *(*formatEtc)(native(format))
			*(*stgMedium)(native(medium)) = stgMedium{}
			if !validDataFormat(f) {
				return dvFormatEtc
			}
			if m, ok := d.extra[f.Format]; ok {
				if f.Tymed&tymedHGlobal == 0 {
					return dvFormatEtc
				}
				h, _, _ := procOleDuplicateData.Call(m.Handle, uintptr(f.Format), 0)
				if h == 0 {
					return eFail
				}
				*(*stgMedium)(native(medium)) = stgMedium{Tymed: tymedHGlobal, Handle: h}
				return sOK
			}
			provider := d.providers[f.Format]
			if provider == nil {
				return dvFormatEtc
			}
			b, err := provider()
			if err != nil {
				d.err = err
				return eFail
			}
			if f.Tymed&tymedStream != 0 {
				stream := memStream(b)
				if stream == 0 {
					return eFail
				}
				// IDataObject::GetData transfers bytes from zero through the
				// stream's current position. OLE uses that position as the
				// length when rendering a stream onto the Win32 clipboard.
				if failed(comCall(stream, 5, 0, 2, 0)) { // IStream::Seek(0, STREAM_SEEK_END)
					release(stream)
					return eFail
				}
				*(*stgMedium)(native(medium)) = stgMedium{Tymed: tymedStream, Handle: stream}
				return sOK
			}
			h := dataGlobal(b)
			if h == 0 {
				return eFail
			}
			*(*stgMedium)(native(medium)) = stgMedium{Tymed: tymedHGlobal, Handle: h}
			return sOK
		}),
		notImpl,
		syscall.NewCallback(func(this, format uintptr) uintptr {
			d := dragObjects[this]
			f := *(*formatEtc)(native(format))
			if validDataFormat(f) && (d.providers[f.Format] != nil || (f.Tymed&tymedHGlobal != 0 && d.extra[f.Format].Handle != 0)) {
				return sOK
			}
			return dvFormatEtc
		}),
		syscall.NewCallback(func(this, in, out uintptr) uintptr {
			*(*formatEtc)(native(out)) = *(*formatEtc)(native(in))
			(*formatEtc)(native(out)).Device = 0
			return 0x40130 // DATA_S_SAMEFORMATETC
		}),
		syscall.NewCallback(func(this, format, medium, owned uintptr) uintptr {
			d := dragObjects[this]
			f, m := *(*formatEtc)(native(format)), *(*stgMedium)(native(medium))
			if !validDataFormat(f) || m.Tymed != tymedHGlobal {
				return dvFormatEtc
			}
			if owned == 0 {
				h, _, _ := procOleDuplicateData.Call(m.Handle, uintptr(f.Format), 0)
				if h == 0 {
					return eFail
				}
				m = stgMedium{Tymed: tymedHGlobal, Handle: h}
			}
			if old, ok := d.extra[f.Format]; ok {
				procReleaseStgMedium.Call(uintptr(unsafe.Pointer(&old)))
			}
			d.extra[f.Format] = m
			if !slices.Contains(d.formats, f.Format) {
				d.formats = append(d.formats, f.Format)
			}
			return sOK
		}),
		syscall.NewCallback(func(this, direction, out uintptr) uintptr {
			*(*uintptr)(native(out)) = 0
			if direction != 1 {
				return eNotImpl
			}
			var formats []formatEtc
			for _, f := range dragObjects[this].formats {
				format := nativeFormat(f)
				if dragObjects[this].providers[f] != nil {
					format.Tymed |= tymedStream
				}
				formats = append(formats, format)
			}
			*(*uintptr)(native(out)) = newFormatEnum(formats, 0)
			return sOK
		}),
		syscall.NewCallback(func(this, format, flags, sink, out uintptr) uintptr { return 0x80040003 }), // OLE_E_ADVISENOTSUPPORTED
		notImpl, notImpl,
	}
	dragSourceVtbl = [5]uintptr{
		syscall.NewCallback(func(this, riid, out uintptr) uintptr { return query(this, riid, out, iidIDropSource) }),
		syscall.NewCallback(func(this uintptr) uintptr { s := dragSources[this]; s.refs++; return uintptr(s.refs) }),
		syscall.NewCallback(func(this uintptr) uintptr {
			s := dragSources[this]
			s.refs--
			if s.refs == 0 {
				delete(dragSources, this)
			}
			return uintptr(s.refs)
		}),
		syscall.NewCallback(func(this, escape, keys uintptr) uintptr {
			if escape != 0 || dragSources[this].canceled {
				return dragDropCancel
			}
			if keys&1 == 0 {
				return dragDropDrop
			}
			return sOK
		}),
		syscall.NewCallback(func(this, effect uintptr) uintptr { return dragDefaultCursors }),
	}
	formatEnumVtbl = [7]uintptr{
		syscall.NewCallback(func(this, riid, out uintptr) uintptr { return query(this, riid, out, iidIEnumFormatEtc) }),
		syscall.NewCallback(func(this uintptr) uintptr { e := formatEnums[this]; e.refs++; return uintptr(e.refs) }),
		syscall.NewCallback(func(this uintptr) uintptr {
			e := formatEnums[this]
			e.refs--
			if e.refs == 0 {
				delete(formatEnums, this)
			}
			return uintptr(e.refs)
		}),
		syscall.NewCallback(func(this, count, out, fetched uintptr) uintptr {
			e := formatEnums[this]
			n := min(int(count), len(e.formats)-e.n)
			copy(unsafe.Slice((*formatEtc)(native(out)), n), e.formats[e.n:e.n+n])
			e.n += n
			if fetched != 0 {
				*(*uint32)(native(fetched)) = uint32(n)
			}
			if n < int(count) {
				return sFalse
			}
			return sOK
		}),
		syscall.NewCallback(func(this, count uintptr) uintptr {
			e := formatEnums[this]
			n := min(int(count), len(e.formats)-e.n)
			e.n += n
			if n < int(count) {
				return sFalse
			}
			return sOK
		}),
		syscall.NewCallback(func(this uintptr) uintptr { formatEnums[this].n = 0; return sOK }),
		syscall.NewCallback(func(this, out uintptr) uintptr {
			e := formatEnums[this]
			*(*uintptr)(native(out)) = newFormatEnum(e.formats, e.n)
			return sOK
		}),
	}
}

func newFormatEnum(formats []formatEtc, n int) uintptr {
	e := &formatEnum{vtbl: &formatEnumVtbl, refs: 1, formats: slices.Clone(formats), n: n}
	p := uintptr(unsafe.Pointer(e))
	formatEnums[p] = e
	return p
}
func nativeFormat(f uint16) formatEtc {
	return formatEtc{Format: f, Aspect: dvaspectContent, Index: -1, Tymed: tymedHGlobal}
}
func validDataFormat(f formatEtc) bool {
	return f.Aspect == dvaspectContent && f.Index == -1 && f.Tymed&(tymedHGlobal|tymedStream) != 0
}
func dataGlobal(b []byte) uintptr {
	h, _, _ := procGlobalAlloc.Call(gmemMoveable|0x40, uintptr(max(len(b), 1)))
	if h == 0 {
		return 0
	}
	p, _, _ := procGlobalLock.Call(h)
	if p == 0 {
		procGlobalFree.Call(h)
		return 0
	}
	copy(unsafe.Slice((*byte)(native(p)), len(b)), b)
	procGlobalUnlock.Call(h)
	return h
}

func newDragData(r platform.DragRequest) uintptr { return newTransferData(r.Data, r.Session, false) }

func newTransferData(data transfer.Data, session string, clipboard bool) uintptr {
	d := &dragDataObject{vtbl: &dragDataVtbl, refs: 1, providers: map[uint16]func() ([]byte, error){}, extra: map[uint16]stgMedium{}, clipboard: clipboard}
	setTransferProviders(d, data, session)
	p := uintptr(unsafe.Pointer(d))
	dragObjects[p] = d
	return p
}

func setTransferProviders(d *dragDataObject, data transfer.Data, session string) {
	d.providers = map[uint16]func() ([]byte, error){}
	d.formats = nil
	add := func(f uint16, provider func() ([]byte, error)) {
		if !slices.Contains(d.formats, f) {
			d.formats = append(d.formats, f)
		}
		d.providers[f] = provider
	}
	for _, f := range data.Formats() {
		f := f
		add(uint16(registerClipboardFormat(string(f))), func() ([]byte, error) { return data.Read(f) })
		if d.clipboard {
			// HGLOBAL allocations can include padding after OleFlushClipboard.
			// Keep the byte length as a separate, optional interoperability
			// hint; the actual format still contains unwrapped standard bytes.
			add(uint16(registerClipboardFormat(clipboardLengthPrefix+string(f))), func() ([]byte, error) {
				b, err := data.Read(f)
				if err != nil {
					return nil, err
				}
				length := make([]byte, 8)
				binary.LittleEndian.PutUint64(length, uint64(len(b)))
				return length, nil
			})
		}
		switch f {
		case transfer.Text:
			add(cfUnicodeText, func() ([]byte, error) { b, err := data.Read(transfer.Text); return utf16Bytes(string(b)), err })
		case transfer.HTML:
			add(uint16(cfHTML), func() ([]byte, error) { b, err := data.Read(transfer.HTML); return htmlData(string(b)), err })
		case transfer.PNG:
			add(uint16(cfPNG), func() ([]byte, error) { return data.Read(transfer.PNG) })
			if d.clipboard {
				add(cfDIB, func() ([]byte, error) {
					b, err := data.Read(transfer.PNG)
					if err != nil {
						return nil, err
					}
					img, err := png.Decode(bytes.NewReader(b))
					if err != nil {
						return nil, err
					}
					return imageToDIB(img), nil
				})
			}
		case transfer.URIList:
			add(uint16(registerClipboardFormat("UniformResourceLocatorW")), func() ([]byte, error) {
				urls, err := data.URLs()
				if err != nil || len(urls) == 0 {
					return nil, transfer.ErrFormat
				}
				return utf16Bytes(urls[0]), nil
			})
		case transfer.FileList:
			add(cfHDrop, func() ([]byte, error) {
				paths, err := data.Files()
				if err != nil {
					return nil, err
				}
				b := make([]byte, 20)
				binary.LittleEndian.PutUint32(b, 20)
				binary.LittleEndian.PutUint32(b[16:], 1)
				for _, p := range paths {
					b = append(b, utf16Bytes(p)...)
				}
				return append(b, 0, 0), nil
			})
		}
	}
	if session != "" {
		add(uint16(registerClipboardFormat(string(platform.DragSessionFormat))), func() ([]byte, error) { return []byte(session), nil })
	}
}

func (s *surface) SetDropFormats([]transfer.Format) {} // OLE enumerates the offered IDataObject.
func (s *surface) StartDataDrag(r platform.DragRequest) {
	dropOnce.Do(initDropTarget)
	data := newDragData(r)
	source := &oleDragSource{vtbl: &dragSourceVtbl, refs: 1}
	p := uintptr(unsafe.Pointer(source))
	dragSources[p] = source
	s.dragSource = source
	defer func() { s.dragSource = nil; release(p); release(data) }()
	if len(r.Preview) > 0 {
		s.dragImage(r, data)
	}
	// OLE owns capture during its modal tracking loop.
	procReleaseCapture.Call()
	var effect uint32
	hr, _, _ := procDoDragDrop.Call(data, p, uintptr(r.Operations), uintptr(unsafe.Pointer(&effect)))
	result := transfer.Result{Operation: transfer.Operation(effect) & (transfer.Copy | transfer.Move)}
	if hr != dragDropDrop {
		result.Operation = transfer.None
		result.Canceled = hr == dragDropCancel || hr == sOK
	}
	if failed(hr) {
		result.Err = hresultError("DoDragDrop", hr)
	}
	if err := dragObjects[data].err; err != nil {
		result.Operation, result.Canceled, result.Err = transfer.None, false, err
	}
	r.Done(result)
}
func (s *surface) CancelDataDrag() {
	if s.dragSource != nil {
		s.dragSource.canceled = true
		procPostMessageW.Call(s.hwnd, wmMouseMove, 0, 0)
	}
}

func dragHelper(iid *GUID) uintptr {
	var helper uintptr
	procCoCreateInstance.Call(uintptr(unsafe.Pointer(&clsDragDropHelper)), 0, 1, uintptr(unsafe.Pointer(iid)), uintptr(unsafe.Pointer(&helper)))
	return helper
}

func (s *surface) dragImage(r platform.DragRequest, data uintptr) {
	helper := dragHelper(&iidIDragSourceHelper)
	if helper == 0 {
		return
	}
	defer release(helper)
	scale := float64(s.dpi()) / 96
	w, h := max(int(float64(r.Width)*scale), 1), max(int(float64(r.Height)*scale), 1)
	bi := bitmapInfoHeader{Width: int32(w), Height: -int32(h), Planes: 1, BitCount: 32}
	bi.Size = uint32(unsafe.Sizeof(bi))
	var bits uintptr
	bmp, _, _ := procCreateDIBSection.Call(0, uintptr(unsafe.Pointer(&bi)), 0, uintptr(unsafe.Pointer(&bits)), 0, 0)
	if bmp == 0 {
		return
	}
	defer procDeleteObject.Call(bmp)
	pix := unsafe.Slice((*byte)(native(bits)), w*h*4)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := (min(int(float64(y)/scale), r.Height-1)*r.Width + min(int(float64(x)/scale), r.Width-1)) * 4
			copy(pix[(y*w+x)*4:][:4], r.Preview[i:i+4])
		}
	}
	image := struct {
		Size   point
		Offset point
		Bitmap uintptr
		Key    uint32
	}{point{int32(w), int32(h)}, point{int32(float64(r.HotX) * scale), int32(float64(r.HotY) * scale)}, bmp, 0xFFFFFFFF}
	comCall(helper, 3, uintptr(unsafe.Pointer(&image)), data)
}

func oleDataBytes(data uintptr, f uint16) ([]byte, error) {
	format := nativeFormat(f)
	var m stgMedium
	// A stream preserves exact binary lengths, including trailing NULs.
	// HGLOBAL may contain allocator padding. Prefer streams where offered.
	format.Tymed = tymedStream
	if !failed(comCall(data, dataGetData, uintptr(unsafe.Pointer(&format)), uintptr(unsafe.Pointer(&m)))) {
		defer procReleaseStgMedium.Call(uintptr(unsafe.Pointer(&m)))
		if m.Tymed != tymedStream || m.Handle == 0 {
			return nil, transfer.ErrFormat
		}
		return readDragStream(m.Handle)
	}
	format.Tymed = tymedHGlobal
	if failed(comCall(data, dataGetData, uintptr(unsafe.Pointer(&format)), uintptr(unsafe.Pointer(&m)))) {
		return nil, transfer.ErrFormat
	}
	defer procReleaseStgMedium.Call(uintptr(unsafe.Pointer(&m)))
	if m.Tymed != tymedHGlobal || m.Handle == 0 {
		return nil, transfer.ErrFormat
	}
	p, _, _ := procGlobalLock.Call(m.Handle)
	if p == 0 {
		return nil, errors.New("mygo: cannot lock drag data")
	}
	defer procGlobalUnlock.Call(m.Handle)
	n, _, _ := procGlobalSize.Call(m.Handle)
	if n > 64<<20 {
		return nil, errors.New("mygo: drag representation exceeds 64 MiB")
	}
	return bytes.Clone(unsafe.Slice((*byte)(native(p)), n)), nil
}

func readDragStream(stream uintptr) ([]byte, error) {
	// GetData leaves the cursor at the end of the transferred bytes.
	if failed(comCall(stream, 5, 0, 0, 0)) { // IStream::Seek(0, STREAM_SEEK_SET)
		return nil, errors.New("mygo: cannot seek drag stream")
	}
	var out []byte
	buf := make([]byte, 64<<10)
	for {
		var n uint32
		hr := comCall(stream, 3, uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)), uintptr(unsafe.Pointer(&n)))
		if failed(hr) || n > uint32(len(buf)) {
			return nil, errors.New("mygo: cannot read drag stream")
		}
		if len(out)+int(n) > 64<<20 {
			return nil, errors.New("mygo: drag representation exceeds 64 MiB")
		}
		out = append(out, buf[:n]...)
		if n == 0 || hr == sFalse {
			return out, nil
		}
	}
}
func oleText(b []byte) string {
	u := make([]uint16, len(b)/2)
	for n := range u {
		u[n] = binary.LittleEndian.Uint16(b[n*2:])
	}
	return syscall.UTF16ToString(u)
}
func oleFormat(f uint16) transfer.Format {
	switch f {
	case cfUnicodeText:
		return transfer.Text
	case cfHDrop:
		return transfer.FileList
	case uint16(cfHTML):
		return transfer.HTML
	case uint16(cfPNG):
		return transfer.PNG
	}
	name := oleRegisteredFormatName(f)
	switch name {
	case "UniformResourceLocatorW":
		return transfer.URIList
	case "Rich Text Format":
		// RichEdit's, which WordPad and Word copy.
		return "text/rtf"
	}
	return transfer.Format(name)
}

func oleRegisteredFormatName(f uint16) string {
	var buf [256]uint16
	n, _, _ := procGetClipboardFormatNameW.Call(uintptr(f), uintptr(unsafe.Pointer(&buf[0])), 256)
	if n == 0 {
		return ""
	}
	return syscall.UTF16ToString(buf[:n])
}

func oleOffer(data uintptr, allowed uint32, keys uintptr) (*platform.DataDragEvent, map[transfer.Format]uint16) {
	d := &platform.DataDragEvent{Offer: transfer.Offer{Operations: transfer.Operation(allowed) & (transfer.Copy | transfer.Move)}}
	if keys&8 != 0 {
		d.Offer.Suggested = transfer.Copy
	} else if keys&4 != 0 {
		d.Offer.Suggested = transfer.Move
	}
	formats := map[transfer.Format]uint16{}
	add := func(f uint16) {
		canonical := oleFormat(f)
		if canonical == "" || strings.HasPrefix(string(canonical), clipboardLengthPrefix) || strings.ContainsAny(string(canonical), "\x00\r\n") {
			return
		}
		if canonical == platform.DragSessionFormat {
			if b, err := oleDataBytes(data, f); err == nil {
				d.Session = strings.TrimRight(string(b), "\x00")
			}
			return
		}
		if !slices.Contains(d.Offer.Formats, canonical) {
			d.Offer.Formats = append(d.Offer.Formats, canonical)
		}
		formats[canonical] = f
		if canonical == transfer.FileList {
			d.HasFiles = true
			formats[transfer.URIList] = f
			if !slices.Contains(d.Offer.Formats, transfer.URIList) {
				d.Offer.Formats = append(d.Offer.Formats, transfer.URIList)
			}
		}
	}
	var enum uintptr
	if comCall(data, dataEnumFormatEtc, 1, uintptr(unsafe.Pointer(&enum))) == sOK && enum != 0 {
		defer release(enum)
		for n := 0; n < 1024; n++ {
			var f formatEtc
			if comCall(enum, 3, 1, uintptr(unsafe.Pointer(&f)), 0) != sOK {
				break
			}
			if f.Device != 0 {
				procCoTaskMemFree.Call(f.Device)
			}
			if validDataFormat(f) {
				add(f.Format)
			}
		}
	}
	// Some sources do not enumerate. Query common formats too.
	for _, f := range []uint16{cfUnicodeText, cfHDrop, uint16(cfHTML), uint16(cfPNG), uint16(registerClipboardFormat("UniformResourceLocatorW")), uint16(registerClipboardFormat(string(platform.DragSessionFormat)))} {
		format := nativeFormat(f)
		if comCall(data, dataQueryGetData, uintptr(unsafe.Pointer(&format))) == sOK {
			add(f)
		}
	}
	return d, formats
}
