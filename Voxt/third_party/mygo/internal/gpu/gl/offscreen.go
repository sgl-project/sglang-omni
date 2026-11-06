//go:build linux && (amd64 || arm64)

package gl

import (
	"errors"
	"fmt"
	"unsafe"

	"github.com/ebitengine/purego"

	"github.com/egoist/mygo/internal/scene"
)

// EGL enumerations.
const (
	eglPlatformSurfaceless   = 0x31DD // EGL_PLATFORM_SURFACELESS_MESA
	eglNone                  = 0x3038
	eglRenderableType        = 0x3040
	eglOpenGLBit             = 0x0008
	eglSurfaceType           = 0x3033
	eglPbufferBit            = 0x0001
	eglOpenGLAPI             = 0x30A2
	eglOpenGLESAPI           = 0x30A0
	eglOpenGLES3Bit          = 0x0040 // EGL_OPENGL_ES3_BIT_KHR
	eglContextMajorVersion   = 0x3098
	eglContextMinorVersion   = 0x30FB
	eglContextProfileMask    = 0x30FD
	eglContextCoreProfileBit = 0x1
)

// Offscreen is a GL context without a window, current on the thread that
// made it, which must stay locked to it, with a framebuffer bound: for
// drawing scenes in tests, as packages drawing effects do. EGL makes it
// with Mesa's surfaceless platform, which draws with llvmpipe without a
// GPU.
type Offscreen struct {
	display, context uintptr
	fb, tex          uint32
	w, h             int
	release          func()
	r                *Renderer
}

// NewOffscreen makes an OpenGL 3.3 context, or with es an OpenGL ES 3.0
// one, current on the calling thread, and binds a framebuffer of w×h
// pixels.
func NewOffscreen(es bool, w, h int) (o *Offscreen, err error) {
	lib, err := purego.Dlopen("libEGL.so.1", purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return nil, fmt.Errorf("gl: no EGL: %w", err)
	}
	var (
		getPlatformDisplay func(platform uint32, native uintptr, attribs unsafe.Pointer) uintptr
		initialize         func(display uintptr, major, minor *int32) uint32
		bindAPI            func(api uint32) uint32
		chooseConfig       func(display uintptr, attribs *int32, configs *uintptr, size int32, n *int32) uint32
		createContext      func(display, config, share uintptr, attribs *int32) uintptr
		makeCurrent        func(display, draw, read, context uintptr) uint32
		destroyContext     func(display, context uintptr) uint32
		terminate          func(display uintptr) uint32
	)
	for name, fn := range map[string]any{
		"eglGetPlatformDisplay": &getPlatformDisplay, "eglInitialize": &initialize, "eglBindAPI": &bindAPI,
		"eglChooseConfig": &chooseConfig, "eglCreateContext": &createContext, "eglMakeCurrent": &makeCurrent,
		"eglDestroyContext": &destroyContext, "eglTerminate": &terminate,
	} {
		sym, err := purego.Dlsym(lib, name)
		if err != nil || sym == 0 {
			return nil, fmt.Errorf("gl: EGL has no %s", name)
		}
		purego.RegisterFunc(fn, sym)
	}
	o = &Offscreen{w: w, h: h}
	var undo []func()
	o.release = func() {
		for i := len(undo) - 1; i >= 0; i-- {
			undo[i]()
		}
	}
	defer func() {
		if err != nil {
			o.release()
		}
	}()
	display := getPlatformDisplay(eglPlatformSurfaceless, 0, nil)
	if display == 0 || initialize(display, nil, nil) == 0 {
		return nil, errors.New("gl: no surfaceless EGL display")
	}
	undo = append(undo, func() { terminate(display) })
	api, bit, contextAttribs := uint32(eglOpenGLAPI), int32(eglOpenGLBit), []int32{eglContextMajorVersion, 3, eglContextMinorVersion, 3, eglContextProfileMask, eglContextCoreProfileBit, eglNone}
	if es {
		api, bit, contextAttribs = eglOpenGLESAPI, eglOpenGLES3Bit, []int32{eglContextMajorVersion, 3, eglContextMinorVersion, 0, eglNone}
	}
	bindAPI(api)
	// Configurations are for windows unless asked otherwise.
	configAttribs := []int32{eglRenderableType, bit, eglSurfaceType, eglPbufferBit, eglNone}
	var config uintptr
	var n int32
	if chooseConfig(display, &configAttribs[0], &config, 1, &n) == 0 || n == 0 {
		return nil, errors.New("gl: no EGL configuration for the API")
	}
	context := createContext(display, config, 0, &contextAttribs[0])
	if context == 0 {
		return nil, errors.New("gl: no context of the version")
	}
	undo = append(undo, func() {
		makeCurrent(display, 0, 0, 0)
		destroyContext(display, context)
	})
	if makeCurrent(display, 0, 0, context) == 0 {
		return nil, errors.New("gl: cannot make the context current without a surface")
	}
	if err := load(); err != nil {
		return nil, err
	}
	o.tex = newTexture(glRGBA8, glRGBA, w, h, nil, 0)
	glGenFramebuffers(1, &o.fb)
	glBindFramebuffer(glFramebuffer, o.fb)
	glFramebufferTexture2D(glFramebuffer, glColorAttachment0, glTexture2D, o.tex, 0)
	undo = append(undo, func() {
		if o.r != nil {
			o.r.Release()
		}
		glBindFramebuffer(glFramebuffer, 0)
		glDeleteFramebuffers(1, &o.fb)
		glDeleteTextures(1, &o.tex)
	})
	if status := glCheckFramebufferState(glFramebuffer); status != glFramebufferDone {
		return nil, fmt.Errorf("gl: framebuffer incomplete: %#x", status)
	}
	return o, nil
}

// Info names the context's version and renderer.
func (o *Offscreen) Info() string {
	return goString(glGetString(glVersion)) + ", " + goString(glGetString(glRenderer))
}

// Render draws s, which must be as large as the framebuffer, with a
// renderer of its own, and returns its pixels as the CPU renderer draws
// them: premultiplied BGRA rows from the top.
func (o *Offscreen) Render(s *scene.Scene) ([]byte, error) {
	if o.r == nil {
		r, err := New()
		if err != nil {
			return nil, err
		}
		o.r = r
	}
	if err := o.r.Render(s); err != nil {
		return nil, err
	}
	return ReadFramebuffer(o.w, o.h)
}

// Release deletes the context and what it holds.
func (o *Offscreen) Release() { o.release() }
