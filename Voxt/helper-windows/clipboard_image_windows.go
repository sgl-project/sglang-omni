//go:build windows

package main

// The clipboard's picture for Rewrite's language model (the protocol's
// clipboard.image): a DIB placed by an app, a PNG, or the first image
// file copied in Explorer; empty when there is none.

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"image/jpeg"
	"os"
	"path/filepath"
	"strings"
	"unsafe"

	"golang.org/x/image/bmp"
	"golang.org/x/image/draw"
	"golang.org/x/sys/windows"
)

const (
	clipboardDIB       = 8
	clipboardDIBV5     = 17
	clipboardHDROP     = 15
	clipboardFormatPNG = "PNG"
)

var procRegisterClipboardFormat = user32.NewProc("RegisterClipboardFormatW")

// clipboardImage is the clipboard's picture as JPEG bytes (nil when it
// holds none): DIB, PNG, or a copied image file (PNG or JPEG only).
func clipboardImage(maxSide int) ([]byte, error) {
	if err := openClipboard(); err != nil {
		return nil, err
	}
	defer procCloseClipboard.Call()

	for _, format := range []uintptr{clipboardDIBV5, clipboardDIB} {
		available, _, _ := procIsClipboardFormatAvailable.Call(format)
		if available == 0 {
			continue
		}
		data, err := clipboardBytes(format)
		if err != nil {
			continue
		}
		img, err := decodeDIB(data)
		if err != nil {
			continue
		}
		return encodeJPEG(img, maxSide)
	}

	if available, _, _ := procIsClipboardFormatAvailable.Call(clipboardHDROP); available != 0 {
		files := clipboardFiles()
		for _, file := range files {
			if img := decodeImageFile(file); img != nil {
				return encodeJPEG(img, maxSide)
			}
		}
	}

	if format, _, _ := procRegisterClipboardFormat.Call(uintptr(unsafe.Pointer(windows.StringToUTF16Ptr(clipboardFormatPNG)))); format != 0 {
		if available, _, _ := procIsClipboardFormatAvailable.Call(format); available != 0 {
			data, err := clipboardBytes(format)
			if err == nil {
				if img, _, decErr := image.Decode(bytes.NewReader(data)); decErr == nil {
					return encodeJPEG(img, maxSide)
				}
			}
		}
	}
	return nil, nil
}

// clipboardBytes reads one clipboard format's global memory.
func clipboardBytes(format uintptr) ([]byte, error) {
	memory, _, callErr := procGetClipboardData.Call(format)
	if memory == 0 {
		return nil, fmt.Errorf("GetClipboardData(%d): %w", format, callErr)
	}
	address, _, callErr := procGlobalLock.Call(memory)
	if address == 0 {
		return nil, fmt.Errorf("GlobalLock: %w", callErr)
	}
	defer procGlobalUnlock.Call(memory)
	byteCount, _, _ := procGlobalSize.Call(memory)
	data := make([]byte, byteCount)
	procRtlMoveMemory.Call(uintptr(unsafe.Pointer(&data[0])), address, byteCount)
	return data, nil
}

// dropFiles is the CF_HDROP payload's header.
type dropFiles struct {
	Files uint32
	X, Y  int32
	NC    int32
	Wide  int32
}

// clipboardFiles is the file paths of a copied-file clipboard: the
// double-null-terminated UTF-16 list of a CF_HDROP.
func clipboardFiles() []string {
	data, err := clipboardBytes(clipboardHDROP)
	if err != nil || len(data) < int(unsafe.Sizeof(dropFiles{})) {
		return nil
	}
	head := (*dropFiles)(unsafe.Pointer(&data[0]))
	if head.Wide == 0 || int(head.Files) >= len(data) {
		return nil
	}
	words := make([]uint16, 0, 256)
	for _, w := range (*[1 << 16]uint16)(unsafe.Pointer(&data[head.Files]))[:] {
		if w == 0 {
			break
		}
		words = append(words, w)
	}
	if len(words) == 0 {
		return nil
	}
	return []string{windows.UTF16ToString(words)}
}

// decodeImageFile reads an image file copied in Explorer (PNG or JPEG).
func decodeImageFile(path string) image.Image {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".png", ".jpg", ".jpeg":
	default:
		return nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil
	}
	return img
}

// decodeDIB wraps a clipboard DIB in a BMP file header and decodes it.
func decodeDIB(data []byte) (image.Image, error) {
	if len(data) < 40 {
		return nil, fmt.Errorf("short DIB (%d bytes)", len(data))
	}
	size := binary.LittleEndian.Uint32(data[0:4])
	if size < 40 {
		return nil, fmt.Errorf("unknown DIB header %d", size)
	}
	file := make([]byte, 14+len(data))
	copy(file, "BM")
	binary.LittleEndian.PutUint32(file[2:6], uint32(len(file)))
	binary.LittleEndian.PutUint32(file[10:14], 14+size)
	copy(file[14:], data)
	return bmp.Decode(bytes.NewReader(file))
}

// encodeJPEG is img as JPEG, downsized to maxSide on the long edge
// (0: no resize).
func encodeJPEG(img image.Image, maxSide int) ([]byte, error) {
	if maxSide > 0 {
		bounds := img.Bounds()
		w, h := bounds.Dx(), bounds.Dy()
		long := max(w, h)
		if long > maxSide {
			nw, nh := w*maxSide/long, h*maxSide/long
			resized := image.NewRGBA(image.Rect(0, 0, nw, nh))
			draw.CatmullRom.Scale(resized, resized.Bounds(), img, bounds, draw.Over, nil)
			img = resized
		}
	}
	var out bytes.Buffer
	if err := jpeg.Encode(&out, img, &jpeg.Options{Quality: 85}); err != nil {
		return nil, err
	}
	return out.Bytes(), nil
}
