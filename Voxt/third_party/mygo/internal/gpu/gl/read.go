//go:build (linux && (amd64 || arm64)) || js

package gl

import (
	"fmt"
	"unsafe"
)

// ReadFramebuffer returns the pixels of the bound framebuffer, width×height,
// as the CPU renderer draws them: premultiplied BGRA rows from the top. Tests
// compare them.
func ReadFramebuffer(width, height int) ([]byte, error) {
	if err := load(); err != nil {
		return nil, err
	}
	if width <= 0 || height <= 0 {
		return nil, nil
	}
	pix := make([]byte, 4*width*height)
	glPixelStorei(glPackAlignment, 1)
	glReadPixels(0, 0, int32(width), int32(height), glRGBA, glUnsignedByte, unsafe.Pointer(&pix[0]))
	glPixelStorei(glPackAlignment, 4)
	if e := glGetError(); e != glNoError {
		return nil, fmt.Errorf("gl: reading the framebuffer failed (error %#x)", e)
	}
	// GL's rows go up, and its bytes are RGBA.
	out := make([]byte, len(pix))
	for y := range height {
		src, dst := pix[4*width*(height-1-y):][:4*width], out[4*width*y:][:4*width]
		for x := 0; x < 4*width; x += 4 {
			dst[x], dst[x+1], dst[x+2], dst[x+3] = src[x+2], src[x+1], src[x], src[x+3]
		}
	}
	return out, nil
}
