//go:build linux && (amd64 || arm64)

package linux

import (
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// Surfaces draw with OpenGL only where it runs on a GPU. A software
// renderer, such as Mesa's llvmpipe in virtual machines and in WSL, redraws
// the whole window on the CPU every frame, many times the work of drawing
// in memory, which redraws what changed; and once a window has a GL
// context, GTK composites the window with OpenGL, so the choice is made
// before: with a context GDK makes for a window that never shows, as the
// surfaces' would be, as the first window of native UI is made.

var glProbe struct {
	device, probe  sync.Once
	hasDevice, gpu bool
}

// hasGPUDevice reports whether the system has a device OpenGL could draw
// on (gpuDevice), saying once when it has none.
func hasGPUDevice() bool {
	glProbe.device.Do(func() {
		if glProbe.hasDevice = gpuDevice(); glProbe.hasDevice {
			return
		}
		// Without a GPU device, GL draws on the CPU: the probe, whose
		// driver stays loaded, would cost memory to say so.
		if _, err := os.Stat("/dev/dxg"); err == nil {
			log.Print("mygo: native UI draws without the GPU: Mesa draws on the CPU in WSL, unless GALLIUM_DRIVER=d3d12")
		} else {
			log.Print("mygo: native UI draws without the GPU: there is none")
		}
	})
	return glProbe.hasDevice
}

// gdkWindowAttr is GdkWindowAttr, for gdk_window_new.
type gdkWindowAttr struct {
	title                     ptr
	eventMask                 int32
	x, y, width, height       int32
	wclass                    int32
	visual                    ptr
	windowType                int32
	cursor                    ptr
	wmclassName, wmclassClass ptr
	overrideRedirect          int32
	typeHint                  int32
}

// gpuGL reports whether GDK's OpenGL contexts draw on a GPU.
func gpuGL() bool {
	glProbe.probe.Do(func() {
		// MYGO_GPU=1 draws with OpenGL wherever GDK makes a context, on
		// the CPU too: tests of the GL surface run so without a GPU.
		if os.Getenv("MYGO_GPU") == "1" {
			glProbe.gpu = true
			return
		}
		if !hasGPUDevice() {
			return
		}
		renderer, err := probeGL()
		switch {
		case err != "":
			log.Printf("mygo: native UI draws without the GPU: %s", err)
		case softwareGL(renderer):
			log.Printf("mygo: native UI draws without the GPU: OpenGL draws on the CPU (%s)", renderer)
		default:
			glProbe.gpu = true
		}
	})
	return glProbe.gpu
}

func softwareGL(renderer string) bool {
	r := strings.ToLower(renderer)
	for _, s := range []string{"llvmpipe", "softpipe", "swrast", "software rasterizer"} {
		if strings.Contains(r, s) {
			return true
		}
	}
	return false
}

// gdkGL holds GDK's functions of OpenGL contexts.
var gdkGL struct {
	once           sync.Once
	ok             bool
	createContext  func(w ptr, err *ptr) ptr
	requireVersion func(c ptr, major, minor int32)
	setUseES       func(c ptr, use int32) // GTK 3.22
	realize        func(c ptr, err *ptr) bool
}

func loadGDKGL() bool {
	g := &gdkGL
	g.once.Do(func() {
		g.ok = bind(libGDK, &g.createContext, "gdk_window_create_gl_context") &&
			bind(libGDK, &g.requireVersion, "gdk_gl_context_set_required_version") &&
			bind(libGDK, &g.realize, "gdk_gl_context_realize")
		if !bind(libGDK, &g.setUseES, "gdk_gl_context_set_use_es") {
			g.setUseES = nil
		}
	})
	return g.ok
}

// glContext makes and realizes an OpenGL context for a GdkWindow, as the
// renderer draws with: OpenGL 3.3, else OpenGL ES 3.0, for GPUs with
// OpenGL ES alone and GDK_GL=gles. It returns the context, or the GError of
// the last try.
func glContext(win ptr) (ctx, gerr ptr) {
	if !loadGDKGL() {
		return 0, 0
	}
	g := &gdkGL
	for _, es := range []bool{false, true} {
		if es && g.setUseES == nil {
			break
		}
		if gerr != 0 {
			gErrorFree(gerr)
			gerr = 0
		}
		if ctx = g.createContext(win, &gerr); ctx == 0 {
			return 0, gerr
		}
		if es {
			g.setUseES(ctx, 1)
			g.requireVersion(ctx, 3, 0)
		} else {
			g.requireVersion(ctx, 3, 3)
		}
		if g.realize(ctx, &gerr) {
			return ctx, 0
		}
		gObjectUnref(ctx)
	}
	return 0, gerr
}

// probeGL makes an OpenGL context as a GtkGLArea would and returns the name
// of its renderer, or why there is none.
func probeGL() (renderer, failure string) {
	var (
		windowNew     func(parent ptr, attr *gdkWindowAttr, mask int32) ptr
		windowDestroy func(w ptr)
		makeCurrent   func(c ptr)
		clearCurrent  func()
		getString     func(name uint32) ptr
		haveSyms      = loadGDKGL()
	)
	for name, fn := range map[string]any{
		"gdk_window_new": &windowNew, "gdk_window_destroy": &windowDestroy,
		"gdk_gl_context_make_current": &makeCurrent, "gdk_gl_context_clear_current": &clearCurrent,
	} {
		haveSyms = bind(libGDK, fn, name) && haveSyms
	}
	if !haveSyms {
		return "", "GDK has no OpenGL"
	}
	epoxy, err := purego.Dlopen("libepoxy.so.0", purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return "", "no libepoxy"
	}
	// libepoxy exports a pointer to each GL function.
	sym, err := purego.Dlsym(epoxy, "epoxy_glGetString")
	if err != nil || sym == 0 {
		return "", "no glGetString"
	}
	purego.RegisterFunc(&getString, **(**uintptr)(unsafe.Pointer(&sym)))

	const inputOutput, toplevel = 0, 1
	win := windowNew(0, &gdkWindowAttr{width: 1, height: 1, wclass: inputOutput, windowType: toplevel}, 0)
	if win == 0 {
		return "", "cannot make a window"
	}
	defer func() {
		windowDestroy(win)
		gObjectUnref(win) // with the GL context GDK made for painting it
	}()
	ctx, gerr := glContext(win)
	if ctx == 0 {
		return "", failed(gerr)
	}
	defer gObjectUnref(ctx)
	makeCurrent(ctx)
	defer clearCurrent()
	const glRenderer = 0x1F01
	return goStr(getString(glRenderer)), ""
}

// failed describes a GError of GDK's OpenGL.
func failed(gerr ptr) string {
	if err := gErr(gerr); err != nil {
		return err.Error()
	}
	return "no OpenGL context"
}

// gpuDevice reports whether the system has a device OpenGL could draw on:
// NVIDIA's, or a render node of a DRM driver with 3D. Mesa draws on the
// CPU, loading as much as the probe would for nothing, on display-only
// drivers, as virtual machines and servers have, and on WSL's device
// unless GALLIUM_DRIVER=d3d12 gives it the GPU.
func gpuDevice() bool { return gpuDeviceIn("/", os.Getenv("GALLIUM_DRIVER")) }

// displayOnly are DRM drivers that only show what the CPU drew.
var displayOnly = map[string]bool{
	"simpledrm": true, "simple-framebuffer": true, "bochs": true, "bochs-drm": true, "cirrus": true,
	"cirrus-qemu": true, "vboxvideo": true, "hyperv_drm": true, "ast": true, "mgag200": true, "udl": true,
	"vkms": true,
}

func gpuDeviceIn(root, gallium string) bool {
	if m, _ := filepath.Glob(filepath.Join(root, "dev", "nvidia*")); len(m) > 0 {
		return true
	}
	nodes, _ := filepath.Glob(filepath.Join(root, "sys", "class", "drm", "renderD*"))
	for _, node := range nodes {
		driver, err := os.Readlink(filepath.Join(node, "device", "driver"))
		if err != nil || !displayOnly[filepath.Base(driver)] {
			return true // a GPU's, or one to probe
		}
	}
	if _, err := os.Stat(filepath.Join(root, "dev", "dxg")); err == nil {
		return gallium == "d3d12"
	}
	return false
}
