//go:build darwin

package darwin

import (
	"bytes"
	"encoding/hex"
	"errors"
	"image"
	"image/png"
	"slices"
	"strings"

	"github.com/ebitengine/purego"
	"github.com/ebitengine/purego/objc"
	"github.com/egoist/mygo/internal/platform"
	"github.com/egoist/mygo/transfer"
)

var (
	msgDraggingFrame func(id, objc.SEL, NSRect, id)
	msgDragKeyEvent  func(id, objc.SEL, uint, NSPoint, uint, float64, int, id, id, id, bool, uint16) id
	macDragSources   = map[id]*macDataSource{}
	macDragProviders = map[id]*macDataProvider{}
	macDragTypes     = map[transfer.Format]string{}
	macDragFormats   = map[string]transfer.Format{}
)

type macDataSource struct {
	r            platform.DragRequest
	s            *surface
	obj, session id
	providers    []id
	err          error
}
type macDataProvider struct {
	src  *macDataSource
	item transfer.Item
}

func registerDataDragClasses() {
	purego.RegisterFunc(&msgDraggingFrame, msgSendAddr)
	purego.RegisterFunc(&msgDragKeyEvent, msgSendAddr)
	classDef("MyGoDragSource", "NSObject", []string{"NSDraggingSource"}, []objc.MethodDef{
		method("draggingSession:sourceOperationMaskForDraggingContext:", func(self id, _ objc.SEL, session id, context int) uint {
			if src := macDragSources[self]; src != nil {
				return macOperations(src.r.Operations)
			}
			return 0
		}),
		method("draggingSession:endedAtPoint:operation:", func(self id, _ objc.SEL, session id, point NSPoint, operation uint) {
			if src := macDragSources[self]; src != nil {
				src.finish(operation)
			}
		}),
	})
	classDef("MyGoDragProvider", "NSObject", []string{"NSPasteboardItemDataProvider"}, []objc.MethodDef{
		method("pasteboard:item:provideDataForType:", func(self id, _ objc.SEL, board, item, typ id) {
			p := macDragProviders[self]
			if p == nil {
				return
			}
			name := goString(typ)
			var b []byte
			var err error
			if name == macType(platform.DragSessionFormat) {
				b = []byte(p.src.r.Session)
			} else {
				f := macFormat(name)
				b, err = p.item.Read(f)
				if f == transfer.FileList || name == "public.url" {
					b = []byte(strings.TrimSpace(strings.Split(string(b), "\n")[0]))
				}
			}
			if err != nil {
				p.src.err = err
				return
			}
			send(item, "setData:forType:", uintptr(nsData(b)), uintptr(typ))
		}),
	})
}

func macType(f transfer.Format) string {
	switch f {
	case transfer.Text:
		return utString
	case transfer.HTML:
		return utHTML
	case transfer.PNG:
		return utPNG
	case transfer.FileList:
		return "public.file-url"
	case transfer.URIList:
		// public.url carries one URL. Keep the full URI-list bytes in a
		// separate type and offer public.url as an interoperable fallback.
		return "dev.mygo.uri-list"
	}
	if t := macDragTypes[f]; t != "" {
		return t
	}
	var t string
	if cls := class("UTType"); cls != 0 {
		ut := send(cls, "typeWithMIMEType:", uintptr(nsString(string(f))))
		if ut != 0 {
			t = goString(send(ut, "identifier"))
		}
	}
	if t == "" {
		t = "dev.mygo.mime." + hex.EncodeToString([]byte(f))
	}
	macDragTypes[f], macDragFormats[t] = t, f
	return t
}
func macFormat(t string) transfer.Format {
	switch t {
	case utString, "public.plain-text", "NSStringPboardType":
		return transfer.Text
	case utHTML:
		return transfer.HTML
	case utPNG:
		return transfer.PNG
	case "public.file-url":
		return transfer.FileList
	case "public.url", "dev.mygo.uri-list":
		return transfer.URIList
	}
	if f := macDragFormats[t]; f != "" {
		return f
	}
	if strings.HasPrefix(t, "dev.mygo.mime.") {
		if b, err := hex.DecodeString(strings.TrimPrefix(t, "dev.mygo.mime.")); err == nil && len(b) > 0 {
			return transfer.Format(b)
		}
	}
	if cls := class("UTType"); cls != 0 {
		ut := send(cls, "typeWithIdentifier:", uintptr(nsString(t)))
		if ut != 0 {
			if mime := goString(send(ut, "preferredMIMEType")); mime != "" {
				return transfer.Format(mime)
			}
		}
	}
	return transfer.Format(t)
}
func macOperations(o transfer.Operation) uint {
	var n uint
	if o&transfer.Copy != 0 {
		n |= 1
	}
	if o&transfer.Move != 0 {
		n |= 16
	}
	return n
}
func goOperations(n uint) transfer.Operation {
	var o transfer.Operation
	if n&1 != 0 {
		o |= transfer.Copy
	}
	if n&16 != 0 {
		o |= transfer.Move
	}
	return o
}

func (s *surface) SetDropFormats(formats []transfer.Format) {
	var types []id
	for _, f := range formats {
		types = append(types, nsString(macType(f)))
	}
	if slices.Contains(formats, transfer.URIList) {
		types = append(types, nsString("public.url"))
	}
	send(s.view, "registerForDraggedTypes:", uintptr(nsArray(types...)))
}

func (s *surface) StartDataDrag(r platform.DragRequest) {
	if s.w.closed || s.w.lastMouseDown == 0 {
		r.Done(transfer.Result{Err: errors.New("mygo: native drag needs a pointer gesture")})
		return
	}
	withPool(func() {
		src := &macDataSource{r: r, s: s, obj: send(send(class("MyGoDragSource"), "alloc"), "init")}
		macDragSources[src.obj], s.dataDrag = src, src
		img := send(class("NSImage"), "imageNamed:", uintptr(nsString("NSMultipleDocuments")))
		width, height := r.Width, r.Height
		if len(r.Preview) > 0 {
			pix := bytes.Clone(r.Preview)
			for i := 0; i < len(pix); i += 4 {
				pix[i], pix[i+2] = pix[i+2], pix[i]
			}
			var b bytes.Buffer
			png.Encode(&b, &image.RGBA{Pix: pix, Stride: 4 * width, Rect: image.Rect(0, 0, width, height)})
			img = autorelease(send(send(class("NSImage"), "alloc"), "initWithData:", uintptr(nsData(b.Bytes()))))
		} else {
			width, height = 32, 32
		}
		x, y := r.X, r.Y
		if x < 0 || y < 0 {
			p := msgPointFromView(s.view, sel("convertPoint:fromView:"), msgPoint(s.w.lastMouseDown, sel("locationInWindow")), 0)
			x, y = p.X, p.Y
		}
		var dragging []id
		for n, pb := range src.pasteboardItems() {
			di := autorelease(send(send(class("NSDraggingItem"), "alloc"), "initWithPasteboardWriter:", uintptr(pb)))
			frame := NSRect{Origin: NSPoint{x - float64(r.HotX) + float64(n*3), y - float64(r.HotY) + float64(n*3)}, Size: NSSize{float64(width), float64(height)}}
			msgDraggingFrame(di, sel("setDraggingFrame:contents:"), frame, img)
			dragging = append(dragging, di)
		}
		src.session = retain(send(s.view, "beginDraggingSessionWithItems:event:source:", uintptr(nsArray(dragging...)), uintptr(s.w.lastMouseDown), uintptr(src.obj)))
		if src.session == 0 {
			src.err = errors.New("mygo: AppKit failed to start drag")
			src.finish(0)
			return
		}
		send(src.session, "setAnimatesToStartingPositionsOnCancelOrFail:", 1)
	})
}

func (src *macDataSource) pasteboardItems() []id {
	var items []id
	for _, item := range src.r.Data.Items() {
		pb := autorelease(send(send(class("NSPasteboardItem"), "alloc"), "init"))
		provider := send(send(class("MyGoDragProvider"), "alloc"), "init")
		macDragProviders[provider] = &macDataProvider{src, item}
		src.providers = append(src.providers, provider)
		var types []id
		for _, f := range item.Formats() {
			types = append(types, nsString(macType(f)))
			if f == transfer.URIList {
				types = append(types, nsString("public.url"))
			}
		}
		types = append(types, nsString(macType(platform.DragSessionFormat)))
		send(pb, "setDataProvider:forTypes:", uintptr(provider), uintptr(nsArray(types...)))
		items = append(items, pb)
	}
	return items
}

func (src *macDataSource) finish(operation uint) {
	delete(macDragSources, src.obj)
	if src.s.dataDrag == src {
		src.s.dataDrag = nil
	}
	for _, p := range src.providers {
		delete(macDragProviders, p)
		release(p)
	}
	release(src.session)
	// The callback still uses self. Autorelease after AppKit returns.
	autorelease(src.obj)
	op := goOperations(operation)
	if src.err != nil {
		op = transfer.None
	}
	done := src.r.Done
	src.r = platform.DragRequest{}
	done(transfer.Result{Operation: op, Canceled: op == transfer.None && src.err == nil, Err: src.err})
}

func (s *surface) CancelDataDrag() {
	if s.dataDrag == nil {
		return
	}
	// NSDraggingSession has no public cancel method. Posting Escape to
	// AppKit's tracking loop cancels through its normal keyboard path.
	withPool(func() {
		ev := msgDragKeyEvent(id(class("NSEvent")), sel("keyEventWithType:location:modifierFlags:timestamp:windowNumber:context:characters:charactersIgnoringModifiers:isARepeat:keyCode:"),
			10, NSPoint{}, 0, 0, sendInt(s.w.win, "windowNumber"), 0, nsString("\x1b"), nsString("\x1b"), false, 53)
		send(s.w.b.app, "postEvent:atStart:", uintptr(ev), 1)
	})
}

func (s *surface) nativeDataEvent(info id, kind platform.SurfaceEventKind) bool {
	s.dragOperation = transfer.None
	pb := send(info, "draggingPasteboard")
	d := &platform.DataDragEvent{Offer: transfer.Offer{Operations: goOperations(uint(send(info, "draggingSourceOperationMask")))}}
	d.Session = goString(send(pb, "stringForType:", uintptr(nsString(macType(platform.DragSessionFormat)))))
	items := arrayItems(send(pb, "pasteboardItems"))
	for _, item := range items {
		for _, t := range arrayItems(send(item, "types")) {
			f := macFormat(goString(t))
			if f == platform.DragSessionFormat {
				continue
			}
			if !slices.Contains(d.Offer.Formats, f) {
				d.Offer.Formats = append(d.Offer.Formats, f)
			}
			if f == transfer.FileList {
				d.HasFiles = true
				if !slices.Contains(d.Offer.Formats, transfer.URIList) {
					d.Offer.Formats = append(d.Offer.Formats, transfer.URIList)
				}
			}
		}
	}
	mods := eventMods(send(s.w.b.app, "currentEvent"))
	if mods&platform.ModAlt != 0 {
		d.Offer.Suggested = transfer.Copy
	} else if mods&platform.ModSuper != 0 {
		d.Offer.Suggested = transfer.Move
	}
	x, y := s.dragPoint(info)
	if kind == platform.DataDragLeave {
		return s.send(platform.SurfaceEvent{Kind: kind, Drag: d})
	}
	if !s.send(platform.SurfaceEvent{Kind: platform.DataDragOver, X: x, Y: y, Drag: d}) {
		return false
	}
	s.dragOperation = d.Operation
	if kind == platform.DataDragOver {
		return true
	}
	// The native pasteboard is only read after the content accepted formats.
	var dataItems []transfer.Item
	for _, item := range items {
		var reps []transfer.Representation
		for _, f := range d.Formats {
			t := macType(f)
			available := send(item, "dataForType:", uintptr(nsString(t)))
			if available == 0 && f == transfer.URIList {
				available = send(item, "dataForType:", uintptr(nsString("public.url")))
				if available == 0 {
					t = "public.file-url"
					available = send(item, "dataForType:", uintptr(nsString(t)))
				}
			}
			if available != 0 {
				b := goBytes(available)
				if t == "public.file-url" {
					b = filePathURLBytes(b)
				}
				reps = append(reps, transfer.Bytes(f, b))
			}
		}
		if len(reps) > 0 {
			dataItems = append(dataItems, transfer.NewItem(reps...))
		}
	}
	d.Data = transfer.New(dataItems...)
	return s.send(platform.SurfaceEvent{Kind: platform.DataDrop, X: x, Y: y, Drag: d})
}

func (s *surface) nativeDataOperation(info id) uint {
	if !s.nativeDataEvent(info, platform.DataDragOver) {
		return 0
	}
	return macOperations(s.dragOperation)
}
