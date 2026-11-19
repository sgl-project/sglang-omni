//go:build darwin

package macx

import (
	"fmt"
	"runtime"
	"strings"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/ebitengine/purego/objc"
)

type pdfSize struct {
	width, height float64
}

var pdfNative struct {
	once      sync.Once
	err       error
	thumbnail func(objc.ID, objc.SEL, pdfSize, int) objc.ID
}

var (
	pdfSelAlloc          = objc.RegisterName("alloc")
	pdfSelAutorelease    = objc.RegisterName("autorelease")
	pdfSelRelease        = objc.RegisterName("release")
	pdfSelFileURL        = objc.RegisterName("fileURLWithPath:")
	pdfSelInitURL        = objc.RegisterName("initWithURL:")
	pdfSelInitBytes      = objc.RegisterName("initWithBytes:length:encoding:")
	pdfSelString         = objc.RegisterName("string")
	pdfSelPageCount      = objc.RegisterName("pageCount")
	pdfSelPageAt         = objc.RegisterName("pageAtIndex:")
	pdfSelThumbnail      = objc.RegisterName("thumbnailOfSize:forBox:")
	pdfSelTIFF           = objc.RegisterName("TIFFRepresentation")
	pdfSelImageRep       = objc.RegisterName("imageRepWithData:")
	pdfSelRepresentation = objc.RegisterName("representationUsingType:properties:")
	pdfSelLength         = objc.RegisterName("length")
	pdfSelBytes          = objc.RegisterName("bytes")
)

func loadPDF() error {
	if err := load(); err != nil {
		return err
	}
	pdfNative.once.Do(func() {
		const rtld = purego.RTLD_GLOBAL | purego.RTLD_NOW
		if _, err := purego.Dlopen("/System/Library/Frameworks/PDFKit.framework/PDFKit", rtld); err != nil {
			pdfNative.err = err
			return
		}
		messageSend, err := purego.Dlsym(purego.RTLD_DEFAULT, "objc_msgSend")
		if err != nil {
			pdfNative.err = err
			return
		}
		purego.RegisterFunc(&pdfNative.thumbnail, messageSend)
	})
	return pdfNative.err
}

func pdfNSString(value string) objc.ID {
	data := unsafe.StringData(value)
	text := class("NSString").Send(pdfSelAlloc).Send(pdfSelInitBytes,
		uintptr(unsafe.Pointer(data)), uintptr(len(value)), uintptr(4))
	runtime.KeepAlive(value)
	return text.Send(pdfSelAutorelease)
}

func pdfBytes(data objc.ID) []byte {
	if data == 0 {
		return nil
	}
	length := objc.Send[int](data, pdfSelLength)
	if length == 0 {
		return []byte{}
	}
	pointer := objc.Send[*byte](data, pdfSelBytes)
	if pointer == nil {
		return nil
	}
	return append([]byte(nil), unsafe.Slice(pointer, length)...)
}

// ReadPDF extracts a PDF's text layer, or renders its first maxPages pages
// when the document has no text. The returned page count covers the whole PDF.
func ReadPDF(path string, maxPages int) (text string, pages [][]byte, pageCount int, err error) {
	if err = loadPDF(); err != nil {
		return "", nil, 0, err
	}
	pooled(func() {
		url := class("NSURL").Send(pdfSelFileURL, pdfNSString(path))
		document := class("PDFDocument").Send(pdfSelAlloc).Send(pdfSelInitURL, url)
		if document == 0 {
			err = fmt.Errorf("macx: cannot open PDF %q", path)
			return
		}
		defer document.Send(pdfSelRelease)
		text = goString(document.Send(pdfSelString))
		pageCount = objc.Send[int](document, pdfSelPageCount)
		if strings.TrimSpace(text) != "" || pageCount == 0 {
			return
		}
		for index := 0; index < min(pageCount, maxPages); index++ {
			page := document.Send(pdfSelPageAt, index)
			image := pdfNative.thumbnail(page, pdfSelThumbnail, pdfSize{width: 1600, height: 1600}, 1)
			representation := class("NSBitmapImageRep").Send(pdfSelImageRep, image.Send(pdfSelTIFF))
			encoded := pdfBytes(representation.Send(pdfSelRepresentation, uintptr(4), uintptr(0)))
			if len(encoded) == 0 {
				err = fmt.Errorf("macx: cannot render PDF page %d", index+1)
				pages = nil
				return
			}
			pages = append(pages, encoded)
		}
	})
	return text, pages, pageCount, err
}
