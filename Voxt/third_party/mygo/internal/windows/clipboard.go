//go:build windows && (amd64 || arm64)

package windows

import (
	"bytes"
	"encoding/binary"
	"errors"
	"image/png"
	"slices"
	"strings"
	"unsafe"

	"github.com/egoist/mygo/internal/platform"
	"github.com/egoist/mygo/transfer"
)

const clipboardLengthPrefix = "MyGo.Transfer.Length:"

func privateClipboardFormat(f transfer.Format) bool {
	return f == platform.DragSessionFormat || f == "DataObject" || f == "Ole Private Data" || strings.HasPrefix(string(f), clipboardLengthPrefix)
}

var (
	procOleSetClipboard       = ole32.NewProc("OleSetClipboard")
	procOleFlushClipboard     = ole32.NewProc("OleFlushClipboard")
	procOleIsCurrentClipboard = ole32.NewProc("OleIsCurrentClipboard")
	winClipboard              *winClipboardSource
)

type winClipboardSource struct {
	object   uintptr
	data     transfer.Data
	released func()
}

func (s *winClipboardSource) release() {
	if winClipboard == s {
		winClipboard = nil
	}
	s.data = transfer.Data{}
	if fn := s.released; fn != nil {
		s.released = nil
		fn()
	}
}

func currentWinClipboard() *winClipboardSource {
	s := winClipboard
	if s != nil {
		hr, _, _ := procOleIsCurrentClipboard.Call(s.object)
		if hr != sOK {
			winClipboard = nil
			return nil
		}
	}
	return s
}

func (clipboard) WriteData(data transfer.Data, released func()) error {
	// One OLE initialization/vtable per process, shared with native drags.
	dropOnce.Do(initDropTarget)
	if len(data.Formats()) == 0 {
		hr := oleClipboard(func() uintptr { hr, _, _ := procOleSetClipboard.Call(0); return hr })
		if failed(hr) {
			return hresultError("OleSetClipboard", hr)
		}
		if released != nil {
			released()
		}
		return nil
	}
	for _, f := range data.Formats() {
		if privateClipboardFormat(f) {
			return errors.New("mygo: clipboard cannot contain native ownership metadata")
		}
		if registerClipboardFormat(string(f)) == 0 || registerClipboardFormat(clipboardLengthPrefix+string(f)) == 0 {
			return errors.New("mygo: cannot register clipboard format")
		}
	}
	object := newTransferData(data, "", true)
	defer release(object)
	hr := oleClipboard(func() uintptr { hr, _, _ := procOleSetClipboard.Call(object); return hr })
	if failed(hr) {
		return hresultError("OleSetClipboard", hr)
	}
	s := &winClipboardSource{object: object, data: data, released: released}
	winClipboard = s
	dragObjects[object].onRelease = s.release
	return nil
}

// clipboardFormats enumerates the open clipboard without rendering data.
// A stable OpenClipboard lock covers discovery and every selected read.
func clipboardFormats() ([]transfer.Format, map[transfer.Format]uint16) {
	var offered []transfer.Format
	nativeFormats := map[transfer.Format]uint16{}
	add := func(id uint16) {
		f := oleFormat(id)
		if id == cfDIB || id == cfDIBV5 {
			f = transfer.PNG
		}
		if f == "" || privateClipboardFormat(f) || strings.ContainsAny(string(f), "\x00\r\n") {
			return
		}
		if !slices.Contains(offered, f) {
			offered = append(offered, f)
		}
		// The registered "PNG" format, as browsers and Office copy it,
		// comes before bitmaps, whose alpha a DIB may lose, as ReadImage
		// reads it.
		if nativeFormats[f] == 0 || oleRegisteredFormatName(id) == string(f) ||
			id == uint16(cfPNG) && oleRegisteredFormatName(nativeFormats[f]) != string(f) {
			nativeFormats[f] = id
		}
		if f == transfer.FileList {
			if nativeFormats[transfer.URIList] == 0 {
				nativeFormats[transfer.URIList] = id
			}
			if !slices.Contains(offered, transfer.URIList) {
				offered = append(offered, transfer.URIList)
			}
		}
	}
	for f := uintptr(0); ; {
		f, _, _ = procEnumClipboardFormats.Call(f)
		if f == 0 {
			break
		}
		add(uint16(f))
	}
	// Windows synthesizes Unicode text and DIBs for older clipboard owners.
	for _, id := range []uint16{cfUnicodeText, cfDIB} {
		if available, _, _ := procIsClipboardFormatAvailable.Call(uintptr(id)); available != 0 {
			add(id)
		}
	}
	return offered, nativeFormats
}

func (c clipboard) Formats() []transfer.Format {
	if s := currentWinClipboard(); s != nil {
		return s.data.Formats()
	}
	if !c.open() {
		return nil
	}
	defer closeClipboard()
	formats, _ := clipboardFormats()
	return formats
}

func clipboardBytes(format uint16) ([]byte, error) {
	h, _, _ := procGetClipboardData.Call(uintptr(format))
	if h == 0 {
		return nil, transfer.ErrFormat
	}
	n, _, _ := procGlobalSize.Call(h)
	if n > 64<<20 {
		return nil, errors.New("mygo: clipboard representation exceeds 64 MiB")
	}
	p, _, _ := procGlobalLock.Call(h)
	if p == 0 {
		return nil, errors.New("mygo: cannot lock clipboard data")
	}
	defer procGlobalUnlock.Call(h)
	return bytes.Clone(unsafe.Slice((*byte)(native(p)), n)), nil
}

func (c clipboard) ReadData(formats []transfer.Format) (transfer.Data, error) {
	if s := currentWinClipboard(); s != nil {
		if len(formats) == 0 {
			formats = s.data.Formats()
		}
		d, err := s.data.Materialize(formats)
		if currentWinClipboard() != s {
			return transfer.Data{}, platform.ErrClipboardChanged
		}
		return d, err
	}
	// The lock also prevents mixing different owners while delayed data is
	// rendered. Sequence numbers cannot do this: Windows increments them
	// for ordinary delayed rendering, as well as for ownership changes.
	if !c.open() {
		return transfer.Data{}, errors.New("mygo: the clipboard is busy")
	}
	defer closeClipboard()
	offered, nativeFormats := clipboardFormats()
	if len(formats) == 0 {
		formats = offered
		if len(formats) == 0 {
			return transfer.Data{}, nil
		}
	}
	var reps []transfer.Representation
	for n, f := range formats {
		id := nativeFormats[f]
		if id == 0 || slices.Contains(formats[:n], f) {
			continue
		}
		var b []byte
		var err error
		if id == cfHDrop {
			h, _, _ := procGetClipboardData.Call(uintptr(id))
			if h == 0 {
				return transfer.Data{}, transfer.ErrFormat
			}
			files, e := transfer.FileData(droppedHDropPaths(h)...)
			if e != nil {
				return transfer.Data{}, e
			}
			b, err = files.Read(f)
		} else {
			b, err = clipboardBytes(id)
			if err != nil {
				return transfer.Data{}, err
			}
			switch id {
			case cfUnicodeText:
				b = []byte(oleText(b))
			case uint16(cfHTML):
				s := string(b)
				start, end := htmlOffset(s, "StartFragment:"), htmlOffset(s, "EndFragment:")
				if start < 0 || end < start || end > len(b) {
					return transfer.Data{}, errors.New("mygo: invalid CF_HTML clipboard data")
				}
				b = b[start:end]
			case cfDIB, cfDIBV5:
				img := dibToImage(b)
				if img == nil {
					return transfer.Data{}, errors.New("mygo: cannot convert clipboard bitmap")
				}
				var encoded bytes.Buffer
				if err = png.Encode(&encoded, img); err != nil {
					return transfer.Data{}, err
				}
				b = encoded.Bytes()
			default:
				if oleRegisteredFormatName(id) == "UniformResourceLocatorW" {
					b = []byte(oleText(b))
				}
			}
			// Plain registered payloads stay interoperable. The optional
			// length hint restores empty/trailing-NUL custom bytes exactly.
			if oleRegisteredFormatName(id) == string(f) {
				lengthID := uint16(registerClipboardFormat(clipboardLengthPrefix + string(f)))
				if available, _, _ := procIsClipboardFormatAvailable.Call(uintptr(lengthID)); available != 0 {
					length, e := clipboardBytes(lengthID)
					if e == nil && len(length) >= 8 {
						n := binary.LittleEndian.Uint64(length)
						if n <= uint64(len(b)) {
							b = b[:n]
						}
					}
				}
			}
		}
		if err != nil {
			return transfer.Data{}, err
		}
		reps = append(reps, transfer.Bytes(f, b))
	}
	if len(reps) == 0 {
		return transfer.Data{}, transfer.ErrFormat
	}
	return transfer.New(transfer.NewItem(reps...)), nil
}

func (clipboard) Flush() error {
	s := currentWinClipboard()
	if s == nil {
		return nil
	}
	// Keep the object alive while OLE relinquishes its reference.
	addRef(s.object)
	defer release(s.object)
	d, err := s.data.Materialize(s.data.Formats())
	if err != nil {
		return err
	}
	if currentWinClipboard() != s {
		return platform.ErrClipboardChanged
	}
	object := dragObjects[s.object]
	// Validate native conversions (notably PNG -> DIB) before native flush,
	// so an encoding error cannot remove the source's delayed representations.
	for _, provider := range object.providers {
		if _, err := provider(); err != nil {
			return err
		}
	}
	setTransferProviders(object, d, "")
	s.data = d
	hr := oleClipboard(func() uintptr { hr, _, _ := procOleFlushClipboard.Call(); return hr })
	if failed(hr) {
		return hresultError("OleFlushClipboard", hr)
	}
	object.onRelease = nil
	s.release()
	return nil
}

func (c clipboard) Close() {
	if currentWinClipboard() != nil {
		c.Clear()
	}
	// Remote COM references may outlive clipboard ownership. Native tracking
	// has ended; remove their Go provider closures before the app stops.
	for _, d := range dragObjects {
		if d.clipboard {
			d.providers = nil
			if fn := d.onRelease; fn != nil {
				d.onRelease = nil
				fn()
			}
		}
	}
}
