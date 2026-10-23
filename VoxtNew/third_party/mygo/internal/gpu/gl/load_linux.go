//go:build linux && (amd64 || arm64)

package gl

import (
	"fmt"
	"strings"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// The functions of libepoxy and GL this package calls.
var (
	loadOnce sync.Once
	errLoad  error

	epoxyGLVersion    func() int32
	epoxyIsDesktopGL  func() bool
	epoxyHasExtension func(name string) bool
	currentGDKContext func() uintptr // gdk_gl_context_get_current, when GDK is loaded

	glGetString             func(name uint32) uintptr
	glGetError              func() uint32
	glGetIntegerv           func(name uint32, data *int32)
	glCreateShader          func(typ uint32) uint32
	glShaderSource          func(shader uint32, count int32, sources **byte, lengths *int32)
	glCompileShader         func(shader uint32)
	glGetShaderiv           func(shader, name uint32, out *int32)
	glGetShaderInfoLog      func(shader uint32, size int32, length *int32, log *byte)
	glDeleteShader          func(shader uint32)
	glCreateProgram         func() uint32
	glAttachShader          func(program, shader uint32)
	glLinkProgram           func(program uint32)
	glGetProgramiv          func(program, name uint32, out *int32)
	glGetProgramInfoLog     func(program uint32, size int32, length *int32, log *byte)
	glUseProgram            func(program uint32)
	glDeleteProgram         func(program uint32)
	glGetUniformLocation    func(program uint32, name string) int32
	glGetUniformBlockIndex  func(program uint32, name string) uint32
	glUniformBlockBinding   func(program, index, binding uint32)
	glBindBufferBase        func(target, index, buffer uint32)
	glUniform1i             func(location, v int32)
	glUniform2f             func(location int32, x, y float32)
	glUniform1f             func(location int32, v float32)
	glUniform2i             func(location, x, y int32)
	glUniform4i             func(location, x, y, z, w int32)
	glCopyTexSubImage2D     func(target uint32, level, xoff, yoff, x, y, w, h int32)
	glGenVertexArrays       func(n int32, out *uint32)
	glBindVertexArray       func(array uint32)
	glDeleteVertexArrays    func(n int32, arrays *uint32)
	glGenBuffers            func(n int32, out *uint32)
	glBindBuffer            func(target, buffer uint32)
	glBufferData            func(target uint32, size int, data unsafe.Pointer, usage uint32)
	glDeleteBuffers         func(n int32, buffers *uint32)
	glEnableVertexAttrib    func(index uint32)
	glVertexAttribPointer   func(index uint32, size int32, typ uint32, normalized bool, stride int32, offset uintptr)
	glVertexAttribDivisor   func(index, divisor uint32)
	glGenTextures           func(n int32, out *uint32)
	glBindTexture           func(target, texture uint32)
	glActiveTexture         func(unit uint32)
	glTexImage2D            func(target uint32, level, internalFormat, w, h, border int32, format, typ uint32, pix unsafe.Pointer)
	glTexSubImage2D         func(target uint32, level, x, y, w, h int32, format, typ uint32, pix unsafe.Pointer)
	glTexParameteri         func(target, name uint32, v int32)
	glDeleteTextures        func(n int32, textures *uint32)
	glPixelStorei           func(name uint32, v int32)
	glViewport              func(x, y, w, h int32)
	glScissor               func(x, y, w, h int32)
	glEnable                func(cap uint32)
	glDisable               func(cap uint32)
	glBlendFunc             func(src, dst uint32)
	glBlendFuncSeparate     func(srcRGB, dstRGB, srcAlpha, dstAlpha uint32)
	glClearColor            func(r, g, b, a float32)
	glClear                 func(mask uint32)
	glDrawArraysInstanced   func(mode uint32, first, count, instances int32)
	glReadPixels            func(x, y, w, h int32, format, typ uint32, pix unsafe.Pointer)
	glGenFramebuffers       func(n int32, out *uint32)
	glBindFramebuffer       func(target, fb uint32)
	glFramebufferTexture2D  func(target, attachment, texTarget, texture uint32, level int32)
	glCheckFramebufferState func(target uint32) uint32
	glDeleteFramebuffers    func(n int32, fbs *uint32)
	glBlitFramebuffer       func(sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1 int32, mask, filter uint32)
)

func load() error {
	loadOnce.Do(func() {
		lib, err := purego.Dlopen("libepoxy.so.0", purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err != nil {
			errLoad = fmt.Errorf("gl: cannot load libepoxy: %w", err)
			return
		}
		var missing []string
		fn := func(f any, name string) {
			sym, err := purego.Dlsym(lib, name)
			if err != nil || sym == 0 {
				missing = append(missing, name)
				return
			}
			purego.RegisterFunc(f, sym)
		}
		// libepoxy exports a pointer for each GL function, to a stub that
		// finds the function of the current context when first called.
		gl := func(f any, name string) {
			sym, err := purego.Dlsym(lib, "epoxy_"+name)
			if err != nil || sym == 0 {
				missing = append(missing, name)
				return
			}
			purego.RegisterFunc(f, **(**uintptr)(unsafe.Pointer(&sym)))
		}
		fn(&epoxyGLVersion, "epoxy_gl_version")
		fn(&epoxyIsDesktopGL, "epoxy_is_desktop_gl")
		fn(&epoxyHasExtension, "epoxy_has_gl_extension")
		gl(&glGetString, "glGetString")
		gl(&glGetError, "glGetError")
		gl(&glGetIntegerv, "glGetIntegerv")
		gl(&glCreateShader, "glCreateShader")
		gl(&glShaderSource, "glShaderSource")
		gl(&glCompileShader, "glCompileShader")
		gl(&glGetShaderiv, "glGetShaderiv")
		gl(&glGetShaderInfoLog, "glGetShaderInfoLog")
		gl(&glDeleteShader, "glDeleteShader")
		gl(&glCreateProgram, "glCreateProgram")
		gl(&glAttachShader, "glAttachShader")
		gl(&glLinkProgram, "glLinkProgram")
		gl(&glGetProgramiv, "glGetProgramiv")
		gl(&glGetProgramInfoLog, "glGetProgramInfoLog")
		gl(&glUseProgram, "glUseProgram")
		gl(&glDeleteProgram, "glDeleteProgram")
		gl(&glGetUniformLocation, "glGetUniformLocation")
		gl(&glGetUniformBlockIndex, "glGetUniformBlockIndex")
		gl(&glUniformBlockBinding, "glUniformBlockBinding")
		gl(&glBindBufferBase, "glBindBufferBase")
		gl(&glUniform1i, "glUniform1i")
		gl(&glUniform2f, "glUniform2f")
		gl(&glUniform1f, "glUniform1f")
		gl(&glUniform2i, "glUniform2i")
		gl(&glUniform4i, "glUniform4i")
		gl(&glCopyTexSubImage2D, "glCopyTexSubImage2D")
		gl(&glGenVertexArrays, "glGenVertexArrays")
		gl(&glBindVertexArray, "glBindVertexArray")
		gl(&glDeleteVertexArrays, "glDeleteVertexArrays")
		gl(&glGenBuffers, "glGenBuffers")
		gl(&glBindBuffer, "glBindBuffer")
		gl(&glBufferData, "glBufferData")
		gl(&glDeleteBuffers, "glDeleteBuffers")
		gl(&glEnableVertexAttrib, "glEnableVertexAttribArray")
		gl(&glVertexAttribPointer, "glVertexAttribPointer")
		gl(&glVertexAttribDivisor, "glVertexAttribDivisor")
		gl(&glGenTextures, "glGenTextures")
		gl(&glBindTexture, "glBindTexture")
		gl(&glActiveTexture, "glActiveTexture")
		gl(&glTexImage2D, "glTexImage2D")
		gl(&glTexSubImage2D, "glTexSubImage2D")
		gl(&glTexParameteri, "glTexParameteri")
		gl(&glDeleteTextures, "glDeleteTextures")
		gl(&glPixelStorei, "glPixelStorei")
		gl(&glViewport, "glViewport")
		gl(&glScissor, "glScissor")
		gl(&glEnable, "glEnable")
		gl(&glDisable, "glDisable")
		gl(&glBlendFunc, "glBlendFunc")
		gl(&glBlendFuncSeparate, "glBlendFuncSeparate")
		gl(&glClearColor, "glClearColor")
		gl(&glClear, "glClear")
		gl(&glDrawArraysInstanced, "glDrawArraysInstanced")
		gl(&glReadPixels, "glReadPixels")
		gl(&glGenFramebuffers, "glGenFramebuffers")
		gl(&glBindFramebuffer, "glBindFramebuffer")
		gl(&glFramebufferTexture2D, "glFramebufferTexture2D")
		gl(&glCheckFramebufferState, "glCheckFramebufferStatus")
		gl(&glDeleteFramebuffers, "glDeleteFramebuffers")
		gl(&glBlitFramebuffer, "glBlitFramebuffer")
		if len(missing) > 0 {
			errLoad = fmt.Errorf("gl: libepoxy lacks %s", strings.Join(missing, ", "))
			return
		}
		// GDK is loaded in apps, where its context tells when GTK
		// realized the area again; tests make contexts of their own.
		if sym, err := purego.Dlsym(purego.RTLD_DEFAULT, "gdk_gl_context_get_current"); err == nil && sym != 0 {
			purego.RegisterFunc(&currentGDKContext, sym)
		}
	})
	return errLoad
}

func currentContext() uintptr {
	if currentGDKContext == nil {
		return 0
	}
	return currentGDKContext()
}

func goString(p uintptr) string {
	if p == 0 {
		return ""
	}
	var b []byte
	for i := uintptr(0); ; i++ {
		c := *(*byte)(unsafe.Add(*(*unsafe.Pointer)(unsafe.Pointer(&p)), i))
		if c == 0 {
			return string(b)
		}
		b = append(b, c)
	}
}

// contextVersion returns the version of the current context, major × 10 +
// minor, and whether it is OpenGL ES.
func contextVersion() (version int32, es bool) { return epoxyGLVersion(), !epoxyIsDesktopGL() }

// hasExtension reports whether the current context has an extension.
func hasExtension(name string) bool { return epoxyHasExtension(name) }

// versionString describes the current context's version, for errors.
func versionString() string { return goString(glGetString(glVersion)) }

// rendererString names the current context's renderer.
func rendererString() string { return goString(glGetString(glRenderer)) }

// setShaderSource sets the source of a shader.
func setShaderSource(s uint32, src string) {
	b := append([]byte(src), 0)
	p := &b[0]
	glShaderSource(s, 1, &p, nil)
}

// shaderLog and programLog return what compiling a shader or linking a
// program logged.
func shaderLog(s uint32) string  { return infoLog(s, glGetShaderiv, glGetShaderInfoLog) }
func programLog(p uint32) string { return infoLog(p, glGetProgramiv, glGetProgramInfoLog) }

func infoLog(obj uint32, get func(uint32, uint32, *int32), read func(uint32, int32, *int32, *byte)) string {
	var n int32
	get(obj, glInfoLogLength, &n)
	if n <= 1 {
		return "no log"
	}
	buf := make([]byte, n)
	read(obj, n, nil, &buf[0])
	return strings.TrimSpace(string(buf[:n-1]))
}
