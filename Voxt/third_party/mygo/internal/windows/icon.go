//go:build windows && (amd64 || arm64)

package windows

import (
	"bytes"
	"fmt"
	"image"
	"image/png"
	"unsafe"
)

var (
	procGetObjectW = gdi32.NewProc("GetObjectW")

	iidIShellItemImageFactory = guid("bcc18b79-ba16-442f-80c4-8a59c30c463b")
)

// dibSection is DIBSECTION, which GetObject fills for a DIB section.
type dibSection struct {
	bmType, bmWidth, bmHeight, bmWidthBytes int32
	bmPlanes, bmBitsPixel                   uint16
	bmBits                                  uintptr
	biSize                                  uint32
	biWidth, biHeight                       int32
	// The rest of BITMAPINFOHEADER, dsBitfields, dshSection and dsOffset.
	rest [7]uint32
	_    [3]uint32
	_    uintptr
	_    uint32
}

// FileIcon returns the icon Explorer shows for a file, a folder or an
// application, or for a shell item such as shell:AppsFolder\<AppUserModelID>
// (IShellItemImageFactory, without thumbnails), as Windows draws it for
// size by size pixels.
func (shell) FileIcon(path string, size int) ([]byte, error) {
	var factory uintptr
	if r, _, _ := procSHCreateItemFromParsingName.Call(uintptr(unsafe.Pointer(u16(path))), 0, uintptr(unsafe.Pointer(&iidIShellItemImageFactory)), uintptr(unsafe.Pointer(&factory))); failed(r) || factory == 0 {
		return nil, hresultError(fmt.Sprintf("finding %q", path), r)
	}
	defer release(factory)
	const iconOnly = 0x4 // SIIGBF_ICONONLY
	var hbm uintptr
	sz := uintptr(uint32(size)) | uintptr(uint32(size))<<32                                           // SIZE, by value
	if r := comCall(factory, 3, sz, iconOnly, uintptr(unsafe.Pointer(&hbm))); failed(r) || hbm == 0 { // GetImage
		return nil, hresultError(fmt.Sprintf("the icon of %q", path), r)
	}
	defer procDeleteObject.Call(hbm)
	img, err := imageOfDIB(hbm)
	if err != nil {
		return nil, err
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// imageOfDIB copies a 32-bit DIB section of the shell, whose pixels are
// BGRA with straight alpha (TestFileIconAlpha), as Go's NRGBA images are.
func imageOfDIB(hbm uintptr) (*image.NRGBA, error) {
	var ds dibSection
	if r, _, _ := procGetObjectW.Call(hbm, unsafe.Sizeof(ds), uintptr(unsafe.Pointer(&ds))); r == 0 || ds.bmBits == 0 || ds.bmBitsPixel != 32 {
		return nil, fmt.Errorf("mygo: the icon is not a 32-bit DIB section (%d bits)", ds.bmBitsPixel)
	}
	w, h := int(ds.bmWidth), int(ds.bmHeight)
	stride := int(ds.bmWidthBytes)
	src := unsafe.Slice((*byte)(native(ds.bmBits)), stride*h)
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		row := y
		if ds.biHeight > 0 { // bottom-up
			row = h - 1 - y
		}
		s := src[row*stride:]
		d := img.Pix[y*img.Stride:]
		for x := 0; x < w; x++ {
			d[4*x], d[4*x+1], d[4*x+2], d[4*x+3] = s[4*x+2], s[4*x+1], s[4*x], s[4*x+3]
		}
	}
	return img, nil
}
