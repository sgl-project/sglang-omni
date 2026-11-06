//go:build darwin

package darwin

import (
	"fmt"
	"path/filepath"
)

// FileIcon returns the icon Finder shows for a file, a folder or an
// application (NSWorkspace iconForFile:), drawn at size by size pixels
// from the representation that fits it best. A symbolic link gets the
// icon of what it points to, without the arrow Finder adds to links.
func (shell) FileIcon(path string, size int) ([]byte, error) {
	real, err := filepath.EvalSymlinks(path)
	if err != nil {
		return nil, err
	}
	path = real
	var png []byte
	withPool(func() {
		if img := send(workspace(), "iconForFile:", uintptr(nsString(path))); img != 0 {
			png = pngOfImage(img, size)
		}
	})
	if png == nil {
		return nil, fmt.Errorf("mygo: no icon for %q", path)
	}
	return png, nil
}

// pngOfImage draws an NSImage into an sRGB bitmap of size by size pixels,
// one pixel to the point, and encodes it as PNG. It must run in a pool.
func pngOfImage(img id, size int) []byte {
	rep := autorelease(msgInitBitmap(send(class("NSBitmapImageRep"), "alloc"),
		sel("initWithBitmapDataPlanes:pixelsWide:pixelsHigh:bitsPerSample:samplesPerPixel:hasAlpha:isPlanar:colorSpaceName:bytesPerRow:bitsPerPixel:"),
		0, size, size, 8, 4, true, false, nsString("NSCalibratedRGBColorSpace"), 0, 0))
	if rep == 0 {
		return nil
	}
	rep = send(rep, "bitmapImageRepByRetaggingWithColorSpace:", uintptr(send(class("NSColorSpace"), "sRGBColorSpace")))
	ctx := send(class("NSGraphicsContext"), "graphicsContextWithBitmapImageRep:", uintptr(rep))
	if ctx == 0 {
		return nil
	}
	gc := class("NSGraphicsContext")
	send(gc, "saveGraphicsState")
	send(gc, "setCurrentContext:", uintptr(ctx))
	send(ctx, "setImageInterpolation:", 3) // NSImageInterpolationHigh
	const sourceOver = 2                   // NSCompositingOperationSourceOver
	dst := NSRect{Size: NSSize{float64(size), float64(size)}}
	msgDrawImage(img, sel("drawInRect:fromRect:operation:fraction:"), dst, NSRect{}, sourceOver, 1)
	send(gc, "restoreGraphicsState")
	return goBytes(send(rep, "representationUsingType:properties:", 4, uintptr(send(class("NSDictionary"), "dictionary")))) // NSBitmapImageFileTypePNG
}
