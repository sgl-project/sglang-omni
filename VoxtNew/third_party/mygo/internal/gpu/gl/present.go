//go:build (linux && (amd64 || arm64)) || js

package gl

import (
	"fmt"
	"unsafe"
)

// Presenter shows frames drawn in memory in the bound framebuffer: what a
// GtkGLArea shows when its content draws on the CPU after all, as when the
// GL renderer fails. GTK shows only what OpenGL draws into a GtkGLArea.
type Presenter struct {
	context uintptr
	tex, fb uint32
	w, h    int
	rgba    []byte
}

// Present copies premultiplied BGRA rows of stride bytes, width×height
// pixels, into the bound framebuffer, which is that size. The context
// current must be the one of the last call, or a new one.
func (p *Presenter) Present(pix []byte, stride, width, height int) error {
	if err := load(); err != nil {
		return err
	}
	if width <= 0 || height <= 0 || len(pix) < stride*(height-1)+4*width {
		return nil
	}
	if c := currentContext(); c != p.context {
		// A new context: the old one took the texture with it.
		p.context, p.tex, p.fb, p.w, p.h = c, 0, 0, 0, 0
	}
	// GL reads RGBA; frames in memory are BGRA.
	n := 4 * width * height
	if cap(p.rgba) < n {
		p.rgba = make([]byte, n)
	}
	rgba := p.rgba[:n]
	for y := range height {
		src, dst := pix[y*stride:][:4*width], rgba[4*width*y:][:4*width]
		for x := 0; x < 4*width; x += 4 {
			dst[x], dst[x+1], dst[x+2], dst[x+3] = src[x+2], src[x+1], src[x], src[x+3]
		}
	}
	var target int32
	glGetIntegerv(glFramebufferBinding, &target)
	if p.tex == 0 || p.w != width || p.h != height {
		if p.tex != 0 {
			glDeleteTextures(1, &p.tex)
		}
		p.tex = newTexture(glRGBA8, glRGBA, width, height, nil, 0)
		if p.fb == 0 {
			glGenFramebuffers(1, &p.fb)
		}
		glBindFramebuffer(glReadFramebuffer, p.fb)
		glFramebufferTexture2D(glReadFramebuffer, glColorAttachment0, glTexture2D, p.tex, 0)
		p.w, p.h = width, height
	}
	glActiveTexture(glTexture0)
	glBindTexture(glTexture2D, p.tex)
	unpack(4*width, glRGBA8)
	glTexSubImage2D(glTexture2D, 0, 0, 0, int32(width), int32(height), glRGBA, glUnsignedByte, unsafe.Pointer(&rgba[0]))
	unpackReset()
	glBindTexture(glTexture2D, 0)
	glBindFramebuffer(glReadFramebuffer, p.fb)
	glBindFramebuffer(glDrawFramebuffer, uint32(target))
	glDisable(glScissorTest)
	// Rows go down in memory and up in GL: the copy flips them.
	w, h := int32(width), int32(height)
	glBlitFramebuffer(0, 0, w, h, 0, h, w, 0, glColorBufferBit, glNearest)
	glBindFramebuffer(glFramebuffer, uint32(target))
	if e := glGetError(); e != glNoError {
		return fmt.Errorf("gl: presenting a frame failed (error %#x)", e)
	}
	return nil
}
