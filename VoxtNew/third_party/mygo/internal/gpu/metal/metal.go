//go:build darwin

// Package metal draws scenes with Metal into a CAMetalLayer, called through
// the Objective-C runtime with purego like the rest of the macOS backend
// (no cgo). Every op of a scene is an instanced quad drawn by one shader
// (shader.metal, compiled ahead of time into shaderlib.go), from the
// instances package gpu builds; clips are scissor rectangles, with the
// innermost rounded clip computed in the shader, as in package d3d11.
package metal

//go:generate go run gen.go

import (
	_ "embed"
	"errors"
	"fmt"
	"image"
	"os"
	"runtime"
	"strings"
	"sync"
	"time"
	"unsafe"
	"weak"

	"github.com/ebitengine/purego"
	"github.com/ebitengine/purego/objc"

	"github.com/egoist/mygo/internal/gpu"
	"github.com/egoist/mygo/internal/scene"
)

//go:embed shader.metal
var shaderSource string

// effectSource is the head and the tail of an effect's library, around
// the line "// effect" (see EffectSource).
//
//go:embed effect.metal
var effectSource string

// EffectSource returns the source of an effect's library: shader.metal
// with EFFECT defined, the head of effect.metal, the effect's shader (see
// scene.Effect), and the tail, which draws its instances with it
// (effect_ps). Effects compile it ahead of time with CompileLibrary.
func EffectSource(src string) string {
	head, tail, _ := strings.Cut(effectSource, "\n// effect\n")
	return "#define EFFECT\n" + shaderSource + "\n" + head + "\n" + src + "\n" + tail
}

// shaderEffectSource is the tail of a shader's library (see
// ShaderSource).
//
//go:embed shadereffect.metal
var shaderEffectSource string

// ShaderSource returns the source of the library of a shader of package
// ui/shader, whose code src ends with shaderMain: as an effect's, with a
// tail of its own (shader_ps), which reads the shader's uniforms and
// images. The shader's effect holds it whole (scene.Effect's Metal), so
// that apps drawing no shader do not carry the tail.
func ShaderSource(src string) string {
	head, _, _ := strings.Cut(effectSource, "\n// effect\n")
	return "#define EFFECT\n" + shaderSource + "\n" + head + "\n" + src + "\n" + shaderEffectSource
}

type id = uintptr

// Metal and Core Graphics structs passed by value.
type (
	mtlRegion      struct{ X, Y, Z, W, H, D uint }
	mtlScissorRect struct{ X, Y, W, H uint }
	mtlClearColor  struct{ R, G, B, A float64 }
	cgSize         struct{ W, H float64 }
	cgRect         struct{ X, Y, W, H float64 }
)

const (
	pixelFormatR8Unorm    = 10
	pixelFormatRGBA8Unorm = 70
	pixelFormatBGRA8Unorm = 80
	// pixelFormatRGBA16Float holds the components of wide colors, which
	// leave 0 to 1.
	pixelFormatRGBA16Float = 115
	pixelFormatRGBA32Float = 125

	usageShaderRead   = 1
	usageRenderTarget = 4
	storageManaged    = 1

	loadActionDontCare     = 0
	loadActionLoad         = 1
	loadActionClear        = 2
	storeActionStore       = 1
	primitiveTriStrip      = 4
	blendZero              = 0
	blendOne               = 1
	blendOneMinusSrc1Color = 16
	blendOneMinusSrc1A     = 18
	filterLinear           = 1
	statusError            = 5
	layerWidthSizable      = 1 << 1
	layerHeightSizable     = 1 << 4
)

var (
	loadOnce sync.Once
	errLoad  error

	msgSend           uintptr
	poolPush, poolPop uintptr
	createDevice      func() id
	srgb              uintptr
	extendedSRGB      uintptr
	msgReplaceRegion  func(obj id, sel objc.SEL, r mtlRegion, level uint, bytes unsafe.Pointer, bytesPerRow uint)
	msgGetBytes       func(obj id, sel objc.SEL, bytes unsafe.Pointer, bytesPerRow uint, r mtlRegion, level uint)
	msgSetScissor     func(obj id, sel objc.SEL, r mtlScissorRect)
	msgSetClearColor  func(obj id, sel objc.SEL, c mtlClearColor)
	msgSetSize        func(obj id, sel objc.SEL, s cgSize)
	msgSetRect        func(obj id, sel objc.SEL, r cgRect)
	msgRect           func(obj id, sel objc.SEL) cgRect
	msgSetFloat       func(obj id, sel objc.SEL, v float64)
	selMu             sync.Mutex
	selectors         = map[string]objc.SEL{}

	dispatchDataCreate, dispatchRelease uintptr

	// Timers that trim the drawables of renderers drawing nothing for a
	// while (see trim), by the handle in their info.
	cfAbsoluteTimeGetCurrent      func() float64
	cfRunLoopTimerCreate          func(alloc uintptr, fireDate, interval float64, flags uint64, order int, callout uintptr, ctx *cfTimerContext) uintptr
	cfRunLoopTimerSetNextFireDate func(timer uintptr, date float64)
	cfRunLoopAddTimer             uintptr
	cfRunLoopGetMain              uintptr
	cfRunLoopTimerInvalidate      uintptr
	cfRelease                     uintptr
	commonModes                   uintptr
	trimCallback                  uintptr
	trimTimers                    = map[uintptr]*Renderer{}
	lastTrimHandle                uintptr
)

// cfTimerContext is CFRunLoopTimerContext.
type cfTimerContext struct {
	version                          int
	info                             uintptr
	retain, release, copyDescription uintptr
}

func load() error {
	loadOnce.Do(func() {
		lib := func(path string) uintptr {
			h, err := purego.Dlopen(path, purego.RTLD_GLOBAL|purego.RTLD_NOW)
			if err != nil && errLoad == nil {
				errLoad = fmt.Errorf("metal: cannot load %s: %w", path, err)
			}
			return h
		}
		objcLib := lib("/usr/lib/libobjc.A.dylib")
		metalLib := lib("/System/Library/Frameworks/Metal.framework/Metal")
		lib("/System/Library/Frameworks/QuartzCore.framework/QuartzCore") // CAMetalLayer
		cg := lib("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
		if errLoad != nil {
			return
		}
		sym := func(h uintptr, name string) uintptr {
			p, err := purego.Dlsym(h, name)
			if err != nil && errLoad == nil {
				errLoad = fmt.Errorf("metal: %w", err)
			}
			return p
		}
		msgSend = sym(objcLib, "objc_msgSend")
		poolPush = sym(objcLib, "objc_autoreleasePoolPush")
		poolPop = sym(objcLib, "objc_autoreleasePoolPop")
		create := sym(metalLib, "MTLCreateSystemDefaultDevice")
		name := sym(cg, "kCGColorSpaceSRGB")
		extendedName := sym(cg, "kCGColorSpaceExtendedSRGB")
		colorSpace := sym(cg, "CGColorSpaceCreateWithName")
		system := lib("/usr/lib/libSystem.B.dylib")
		cf := lib("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
		if errLoad != nil {
			return
		}
		dispatchDataCreate = sym(system, "dispatch_data_create")
		dispatchRelease = sym(system, "dispatch_release")
		timeNow := sym(cf, "CFAbsoluteTimeGetCurrent")
		timerCreate := sym(cf, "CFRunLoopTimerCreate")
		timerSetNext := sym(cf, "CFRunLoopTimerSetNextFireDate")
		cfRunLoopAddTimer = sym(cf, "CFRunLoopAddTimer")
		cfRunLoopGetMain = sym(cf, "CFRunLoopGetMain")
		cfRunLoopTimerInvalidate = sym(cf, "CFRunLoopTimerInvalidate")
		cfRelease = sym(cf, "CFRelease")
		modes := sym(cf, "kCFRunLoopCommonModes")
		// Methods returning structs larger than 16 bytes use another entry
		// point on amd64.
		stret := msgSend
		if runtime.GOARCH == "amd64" {
			stret = sym(objcLib, "objc_msgSend_stret")
		}
		if errLoad != nil {
			return
		}
		purego.RegisterFunc(&createDevice, create)
		purego.RegisterFunc(&cfAbsoluteTimeGetCurrent, timeNow)
		purego.RegisterFunc(&cfRunLoopTimerCreate, timerCreate)
		purego.RegisterFunc(&cfRunLoopTimerSetNextFireDate, timerSetNext)
		commonModes = *(*uintptr)(ptr(modes))
		trimCallback = purego.NewCallback(func(timer, info uintptr) {
			if r := trimTimers[info]; r != nil {
				r.trimIfIdle()
			}
		})
		purego.RegisterFunc(&msgReplaceRegion, msgSend)
		purego.RegisterFunc(&msgGetBytes, msgSend)
		purego.RegisterFunc(&msgSetScissor, msgSend)
		purego.RegisterFunc(&msgSetClearColor, msgSend)
		purego.RegisterFunc(&msgSetSize, msgSend)
		purego.RegisterFunc(&msgSetRect, msgSend)
		purego.RegisterFunc(&msgSetFloat, msgSend)
		purego.RegisterFunc(&msgRect, stret)
		srgb, _, _ = purego.SyscallN(colorSpace, *(*uintptr)(ptr(name)))
		extendedSRGB, _, _ = purego.SyscallN(colorSpace, *(*uintptr)(ptr(extendedName)))
	})
	return errLoad
}

func sel(name string) objc.SEL {
	selMu.Lock()
	defer selMu.Unlock()
	s, ok := selectors[name]
	if !ok {
		s = objc.RegisterName(name)
		selectors[name] = s
	}
	return s
}

func class(name string) id { return id(objc.GetClass(name)) }

// ptr turns an address from native code into a pointer.
func ptr(a uintptr) unsafe.Pointer { return *(*unsafe.Pointer)(unsafe.Pointer(&a)) }

// send sends a message whose arguments are integers or pointers. Go
// pointers converted in the call stay where they are until it returns
// (go:uintptrescapes moves them to the heap).
//
//go:uintptrescapes
func send(obj id, selector string, args ...uintptr) id {
	var a [10]uintptr
	a[0], a[1] = obj, uintptr(sel(selector))
	n := copy(a[2:], args)
	r, _, _ := purego.SyscallN(msgSend, a[:n+2]...)
	return r
}

func release(obj *id) {
	if *obj != 0 {
		send(*obj, "release")
		*obj = 0
	}
}

// pool runs fn in an autorelease pool, which takes the objects Metal
// returns without giving them to the caller.
func pool(fn func()) {
	// The pool belongs to the thread, which the goroutine must not leave
	// before popping it.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	p, _, _ := purego.SyscallN(poolPush)
	defer purego.SyscallN(poolPop, p)
	fn()
}

// nsString returns an autoreleased NSString.
func nsString(s string) id {
	b := append([]byte(s), 0)
	return send(class("NSString"), "stringWithUTF8String:", uintptr(unsafe.Pointer(&b[0])))
}

// describe returns the description of an NSError.
func describe(err id) string {
	if err == 0 {
		return "unknown error"
	}
	p := send(send(err, "localizedDescription"), "UTF8String")
	if p == 0 {
		return "unknown error"
	}
	var b []byte
	for i := uintptr(0); ; i++ {
		c := *(*byte)(ptr(p + i))
		if c == 0 {
			return string(b)
		}
		b = append(b, c)
	}
}

// need reports which of the methods obj lacks, so that a renderer fails
// rather than crash on a system without them.
func need(obj id, what string, methods ...string) error {
	if obj == 0 {
		return fmt.Errorf("metal: no %s", what)
	}
	for _, m := range methods {
		if byte(send(obj, "respondsToSelector:", uintptr(sel(m)))) == 0 {
			return fmt.Errorf("metal: %s has no %s", what, m)
		}
	}
	return nil
}

type texture struct {
	tex       id
	w, h      int
	gen, ver  uint64
	lastFrame uint64
}

// drawState is what drawing scenes into a target keeps from frame to
// frame: a frame's, and each canvas's.
type drawState struct {
	b gpu.Builder
	// Each frame slot owns its instances, including those of canvases,
	// and the texture of the stops of its gradients (gpu.Builder.Stops).
	instBufs [ringSize]id
	instCaps [ringSize]int
	stops    [ringSize]texture
}

func (d *drawState) release() {
	for i := range d.instBufs {
		release(&d.instBufs[i])
		d.instCaps[i] = 0
		release(&d.stops[i].tex)
		d.stops[i] = texture{}
	}
	d.b.Release()
}

// canvasTexture is the texture a canvas (scene.NewCanvas) is drawn into,
// and what drawing it keeps, for as long as the canvas lives: what it holds
// is drawn once, not again when shown.
type canvasTexture struct {
	texture
	drawState
	canvas weak.Pointer[scene.Image]
}

// formatState is what a renderer draws into targets of a pixel format
// with.
type formatState struct {
	format   uint
	pipeline id
	// holePipe draws holes (scene.OpHole), taking their coverage away, and
	// opaquePipe the cores of opaque fills (gpu.Batch.Opaque), without blending.
	holePipe, opaquePipe id
	// downPipe and blurPipe draw the passes computing the backdrops of
	// effects, into the textures of backdrop, the second only along rows.
	downPipe, blurPipe id
	backdrop           [2]texture
	// effects are the pipelines of the effects drawn, made as first drawn.
	effects map[*scene.Effect]*effectPipe
}

// Renderer draws scenes into a CAMetalLayer it adds to a view's layer.
type Renderer struct {
	device, queue, sampler id
	// shaderBytes holds the uniforms of the shader being drawn, laid out.
	shaderBytes []byte
	// lib is the library of the shader, which the pipelines of the classes
	// of batches are made from as they are first used.
	lib id
	// source, unless empty, is the shader's source for tests that draw with
	// another (the upstream one), which has no function constants; pipes are the
	// pipelines of the classes and the worker that makes them.
	source string
	pipes  pipeState
	// formats has what draws into BGRA8 targets, as frames do, and into
	// RGBA16Float ones, as frames showing colors outside the sRGB gamut do
	// (SetWide), made for the first of those; cur is that of the target
	// drawn into.
	formats [2]formatState
	cur     *formatState
	// wide tells that frames show the colors of scenes outside the sRGB
	// gamut, and layerWide that the layer's drawables hold them; wideErr
	// is why they cannot.
	wide, layerWide bool
	wideErr         error
	// layer is the CAMetalLayer in superlayer, the view's.
	layer, superlayer id
	w, h              int
	bounds            cgRect
	scale             float64

	// synced tells that the layer presents with the Core Animation
	// transaction, and holes that the last frame had holes (web views
	// under it). waited tells that the last frame, presented as soon as
	// drawn, waited for its drawable; behind that the last Render drew
	// nothing to catch up with the display, and skippedAt when one last
	// did.
	synced, holes, waited, behind bool
	skippedAt                     time.Time

	mask   texture
	color  texture
	empty  id // a texture to bind where there is none
	images map[uint64]*texture
	// canvases are the textures canvases are drawn into.
	canvases map[uint64]*canvasTexture
	// blurred are the textures of the images the frame draws blurred
	// (gpu.Builder.BlurredImage), the first used of them, and blurTmp the
	// textures their passes go through; stopsTex is the texture of the
	// stops of the scene drawn.
	blurred  []texture
	used     int
	blurTmp  [2]texture
	stopsTex id
	frame    uint64
	// Command buffers retain every resource used by their frame slot.
	inflight    [ringSize]id
	slot, inUse int
	err         error
	checked     bool

	// lastRender is when the last frame drew; trimTimer trims the
	// drawables once none has for a while, armed until it fires, and
	// trimmed is whether it did.
	lastRender         time.Time
	trimTimer          uintptr
	trimHandle         uintptr
	trimArmed, trimmed bool

	// drawState is the frames'.
	drawState
}

// New creates a renderer drawing into a CAMetalLayer that it adds to
// layer, the layer of a view, and sizes as it.
func New(layer uintptr) (r *Renderer, err error) {
	if layer == 0 {
		return nil, errors.New("metal: no layer")
	}
	if r, err = newRenderer(); err != nil {
		return nil, err
	}
	pool(func() {
		if class("CAMetalLayer") == 0 {
			err = errors.New("metal: no CAMetalLayer")
			return
		}
		ml := send(send(class("CAMetalLayer"), "alloc"), "init")
		if err = need(ml, "CAMetalLayer", "setDevice:", "setPixelFormat:", "setFramebufferOnly:", "setPresentsWithTransaction:",
			"setColorspace:", "setDrawableSize:", "setContentsScale:", "setFrame:", "setAutoresizingMask:", "nextDrawable"); err != nil {
			release(&ml)
			return
		}
		send(ml, "setDevice:", r.device)
		send(ml, "setPixelFormat:", pixelFormatBGRA8Unorm)
		// Effects read their backdrops from the drawable, which their
		// passes sample then.
		send(ml, "setFramebufferOnly:", 0)
		// Two drawables are too few: on a ProMotion display, frames
		// waited for one most of each refresh, and skipping a refresh to
		// catch up (render) helped for a few frames only. Three, as Core
		// Animation makes while a window animates, also let both instance
		// slots have a frame in flight while the display shows a third.
		// They cost a frame of memory until the window idles (trim).
		if respondsTo(ml, "setMaximumDrawableCount:") {
			send(ml, "setMaximumDrawableCount:", 3)
		}
		// The first frame shows with the window's other changes; render
		// decides for each frame.
		send(ml, "setPresentsWithTransaction:", 1)
		r.synced = true
		send(ml, "setColorspace:", srgb)
		send(ml, "setOpaque:", 0)
		send(ml, "setAutoresizingMask:", layerWidthSizable|layerHeightSizable)
		send(layer, "addSublayer:", ml)
		r.layer, r.superlayer = ml, layer
	})
	if err != nil {
		r.Release()
		return nil, err
	}
	return r, nil
}

// newRenderer creates a renderer without a layer, for New and tests.
func newRenderer() (r *Renderer, err error) {
	if err := load(); err != nil {
		return nil, err
	}
	r = &Renderer{images: map[uint64]*texture{}}
	r.b.Classes = true
	r.b.Opaque = true // Metal's GPUs discard what opaque cores cover
	r.formats = [2]formatState{{format: pixelFormatBGRA8Unorm}, {format: pixelFormatRGBA16Float}}
	r.cur = &r.formats[0]
	pool(func() { err = r.init() })
	if err != nil {
		r.Release()
		return nil, err
	}
	return r, nil
}

func (r *Renderer) init() (err error) {
	r.device = createDevice() // owned: NS_RETURNS_RETAINED
	if err := need(r.device, "Metal device", "newCommandQueue", "newLibraryWithSource:options:error:",
		"newRenderPipelineStateWithDescriptor:error:", "newSamplerStateWithDescriptor:", "newTextureWithDescriptor:",
		"newBufferWithLength:options:"); err != nil {
		return err
	}
	return nil
}

// initGPU makes what only drawing on the GPU needs. A window presenting
// CPU frames needs the device and layer alone: neither a command queue
// nor shaders, samplers and textures.
func (r *Renderer) initGPU() error {
	if r.queue != 0 {
		return nil
	}
	queue := send(r.device, "newCommandQueue")
	if queue == 0 {
		return errors.New("metal: no command queue")
	}
	defer release(&queue)
	sd := send(send(class("MTLSamplerDescriptor"), "alloc"), "init")
	defer release(&sd)
	send(sd, "setMinFilter:", filterLinear)
	send(sd, "setMagFilter:", filterLinear)
	sampler := send(r.device, "newSamplerStateWithDescriptor:", sd)
	defer release(&sampler)
	empty := r.newTexture(1, 1, pixelFormatRGBA8Unorm, usageShaderRead, []byte{0, 0, 0, 0}, 4)
	defer release(&empty)
	if empty == 0 || sampler == 0 {
		return errors.New("metal: cannot create a texture")
	}
	r.queue, r.sampler, r.empty = queue, sampler, empty
	queue, sampler, empty = 0, 0, 0
	return nil
}

// makePipelines makes the pipelines of f's format, or none.
func (r *Renderer) makePipelines(f *formatState) (err error) {
	if r.lib == 0 {
		var errObj id
		var lib id
		src := shaderSource
		if r.source != "" {
			src = r.source // a test's, never the compiled library
		} else {
			lib = r.compiledLibrary()
		}
		if lib == 0 {
			lib = send(r.device, "newLibraryWithSource:options:error:", nsString(src), 0, uintptr(unsafe.Pointer(&errObj)))
		}
		if lib == 0 {
			return fmt.Errorf("metal: cannot compile the shader: %s", describe(errObj))
		}
		r.lib = lib
	}
	lib := r.lib
	defer func() {
		if err == nil {
			r.startPipelines(f)
		}
	}()
	defer func() {
		if err != nil {
			release(&f.pipeline)
			release(&f.holePipe)
			release(&f.opaquePipe)
			release(&f.downPipe)
			release(&f.blurPipe)
		}
	}()
	if f.pipeline, err = r.newPipeline(lib, "ps", f.format, blendOver); err != nil {
		return err
	}
	if f.holePipe, err = r.newPipeline(lib, "ps", f.format, blendHole); err != nil {
		return err
	}
	if f.format == pixelFormatRGBA16Float {
		// Opaque cores are exactly 1 in coverage only to within the rasterizer's
		// interpolation of the fragment's position, whose noise (a unit in the
		// last place of a half, in the columns where the position crosses a
		// power of two) blending hides in the destination's alpha but a plain
		// write keeps: float16 targets blend them, byte for byte as whole fills.
		f.opaquePipe = send(f.pipeline, "retain")
	} else if f.opaquePipe, err = r.newPipeline(lib, "ps", f.format, blendNone); err != nil {
		return err
	}
	if f.downPipe, err = r.passPipeline(lib, "down", f.format); err != nil {
		return err
	}
	f.blurPipe, err = r.passPipeline(lib, "blur", f.format)
	return err
}

// SetWide has the frames drawn from now on show the colors of scenes
// outside the sRGB gamut (scene.Scene.Wide), into drawables of RGBA16Float
// in the extended sRGB color space, which keeps components beyond 0 to 1
// and which the system shows in the display's gamut, or their nearest sRGB
// colors, into drawables of BGRA8, as without; and reports whether they
// do. The pipelines drawing the wide colors are made the first time.
func (r *Renderer) SetWide(on bool) bool {
	f := &r.formats[1]
	if on && f.pipeline == 0 && r.wideErr == nil {
		r.wideErr = errors.New("metal: no extended sRGB color space")
		if extendedSRGB != 0 {
			pool(func() { r.wideErr = r.makePipelines(f) })
		}
		if r.wideErr != nil {
			fmt.Fprintf(os.Stderr, "%v: drawing the nearest sRGB colors\n", r.wideErr)
		}
	}
	r.wide = on && r.wideErr == nil
	return r.wide
}

// setLayerWide has the layer's drawables hold the colors outside the sRGB
// gamut, or not, and reports whether that changed: the system makes
// others, which hold no frame drawn in memory.
func (r *Renderer) setLayerWide(on bool) (changed bool) {
	if r.layerWide == on || r.layer == 0 {
		return false
	}
	format, space := uintptr(pixelFormatBGRA8Unorm), srgb
	if on {
		format, space = pixelFormatRGBA16Float, extendedSRGB
	}
	tx := class("CATransaction")
	send(tx, "begin")
	send(tx, "setDisableActions:", 1)
	send(r.layer, "setPixelFormat:", format)
	send(r.layer, "setColorspace:", space)
	send(tx, "commit")
	r.layerWide = on
	return true
}

// blendMode is how a pipeline blends the shader's colors with the target.
type blendMode int

const (
	blendOver blendMode = iota // premultiplied colors over what is drawn
	blendHole                  // takes the coverage of holes away
	blendNone                  // none: writes the colors, for opaque cores
)

// newPipeline returns a pipeline drawing instances with the vertex
// function vs and the fragment function fs of lib, blending premultiplied
// colors over what is drawn, or as mode says.
func (r *Renderer) newPipeline(lib id, fs string, format uint, mode blendMode) (id, error) {
	return r.newClassPipeline(lib, fs, format, mode, gpu.ClassGeneral, true)
}

// classConstants are the function constants (the second, continuous corners,
// being the batch's) (shader.metal) of each class
// of batches: whether the fragment function has the code of instances under a
// transform, continuous corners, plain fills, other fills, shadows, glyphs,
// subpixel glyphs and images.
var classConstants = [gpu.NumClasses][8]bool{
	gpu.ClassPlain:    {2: true},
	gpu.ClassFill:     {3: true},
	gpu.ClassShadow:   {4: true},
	gpu.ClassGlyph:    {5: true},
	gpu.ClassSubpixel: {6: true},
	gpu.ClassImage:    {7: true},
}

// specialized returns the function fs of lib with the constants of class cl
// (owned). A function that has none (a library of an effect compiled before
// they existed) is returned as it is; for the general class, the constants are
// all true.
func (r *Renderer) specialized(lib id, fs string, cl gpu.Class, continuous bool) (fn id, errObj id) {
	if cl == gpu.ClassGeneral {
		fn = send(lib, "newFunctionWithName:", nsString(fs))
		if fn == 0 || send(send(fn, "functionConstantsDictionary"), "count") == 0 {
			return fn, 0
		}
		release(&fn)
	}
	values := send(send(class("MTLFunctionConstantValues"), "alloc"), "init")
	defer release(&values)
	const dataTypeBool = 53
	for i, on := range classConstants[cl] {
		v := byte(0)
		if on || cl == gpu.ClassGeneral || i == 1 && continuous {
			v = 1
		}
		send(values, "setConstantValue:type:atIndex:", uintptr(unsafe.Pointer(&v)), dataTypeBool, uintptr(i))
	}
	fn = send(lib, "newFunctionWithName:constantValues:error:", nsString(fs), values, uintptr(unsafe.Pointer(&errObj)))
	return fn, errObj
}

// newClassPipeline is newPipeline with the fragment function specialized for
// the class cl.
func (r *Renderer) newClassPipeline(lib id, fs string, format uint, mode blendMode, cl gpu.Class, continuous bool) (id, error) {
	vsFn := send(lib, "newFunctionWithName:", nsString("vs"))
	defer release(&vsFn)
	fsFn, fnErr := r.specialized(lib, fs, cl, continuous)
	defer release(&fsFn)
	if vsFn == 0 || fsFn == 0 {
		return 0, fmt.Errorf("metal: the shader has no vs or %s: %s", fs, describe(fnErr))
	}
	desc := send(send(class("MTLRenderPipelineDescriptor"), "alloc"), "init")
	defer release(&desc)
	send(desc, "setVertexFunction:", vsFn)
	send(desc, "setFragmentFunction:", fsFn)
	ca := send(send(desc, "colorAttachments"), "objectAtIndexedSubscript:", 0)
	send(ca, "setPixelFormat:", uintptr(format))
	// Premultiplied colors over what is drawn.
	src := uintptr(blendOne)
	if mode == blendHole {
		src = blendZero
	}
	if mode == blendNone {
		send(ca, "setBlendingEnabled:", 0)
	} else {
		send(ca, "setBlendingEnabled:", 1)
	}
	send(ca, "setSourceRGBBlendFactor:", src)
	// Dual-source blending: the shader's second color is the source's
	// alpha of each channel.
	send(ca, "setDestinationRGBBlendFactor:", blendOneMinusSrc1Color)
	send(ca, "setSourceAlphaBlendFactor:", src)
	send(ca, "setDestinationAlphaBlendFactor:", blendOneMinusSrc1A)
	var errObj id
	pipe := send(r.device, "newRenderPipelineStateWithDescriptor:error:", desc, uintptr(unsafe.Pointer(&errObj)))
	if pipe == 0 {
		return 0, fmt.Errorf("metal: cannot create the %s pipeline: %s", fs, describe(errObj))
	}
	return pipe, nil
}

// effectPipe is the pipeline of an effect, or why it has none.
type effectPipe struct {
	pipe id
	err  error
}

// effectPipeline returns the pipeline of e: from the library e compiled
// ahead of time, unless it was compiled from another source, as when the
// renderer's head or tail changed since, which it then compiles. It is 0
// for an effect that does not compile, which draws nothing.
func (r *Renderer) effectPipeline(e *scene.Effect) id {
	f := r.cur
	if p := f.effects[e]; p != nil {
		return p.pipe
	}
	p := &effectPipe{}
	if f.effects == nil {
		f.effects = map[*scene.Effect]*effectPipe{}
	}
	f.effects[e] = p
	src, fn := EffectSource(e.Metal.Source), "effect_ps"
	if e.Shader != nil {
		src, fn = e.Metal.Source, "shader_ps" // ShaderSource's
	}
	var lib id
	if e.Shader != nil {
		// Made as the shader compiled (PrecompileShader).
		if shared, pipe, ok := sharedShader(r.device, src, f.format); ok {
			if pipe != 0 {
				release(&shared)
				p.pipe = pipe
				return p.pipe
			}
			lib = shared
		}
	}
	if lib == 0 && len(e.Metal.Compiled) > 0 && e.Metal.Sum == gpu.SourceSum(src) {
		lib = r.libraryFromData(e.Metal.Compiled)
	}
	if lib == 0 {
		var errObj id
		if lib = send(r.device, "newLibraryWithSource:options:error:", nsString(src), 0, uintptr(unsafe.Pointer(&errObj))); lib == 0 {
			p.err = fmt.Errorf("metal: cannot compile the effect %s: %s", e.Name, describe(errObj))
		}
	}
	if lib != 0 {
		p.pipe, p.err = r.newPipeline(lib, fn, f.format, blendOver)
		release(&lib)
	}
	if p.err != nil && r.err == nil {
		// Shown once, as frames go on without the effect.
		fmt.Fprintln(os.Stderr, p.err)
	}
	return p.pipe
}

// passPipeline returns the pipeline of a pass computing backdrops, which
// draws the fragment function name of lib over its target, unblended.
func (r *Renderer) passPipeline(lib id, name string, format uint) (id, error) {
	vs := send(lib, "newFunctionWithName:", nsString("passVS"))
	defer release(&vs)
	fs := send(lib, "newFunctionWithName:", nsString(name))
	defer release(&fs)
	if vs == 0 || fs == 0 {
		return 0, fmt.Errorf("metal: the shader has no passVS or %s", name)
	}
	desc := send(send(class("MTLRenderPipelineDescriptor"), "alloc"), "init")
	defer release(&desc)
	send(desc, "setVertexFunction:", vs)
	send(desc, "setFragmentFunction:", fs)
	ca := send(send(desc, "colorAttachments"), "objectAtIndexedSubscript:", 0)
	send(ca, "setPixelFormat:", uintptr(format))
	var errObj id
	pipe := send(r.device, "newRenderPipelineStateWithDescriptor:error:", desc, uintptr(unsafe.Pointer(&errObj)))
	if pipe == 0 {
		return 0, fmt.Errorf("metal: cannot create the %s pipeline: %s", name, describe(errObj))
	}
	return pipe, nil
}

// compiledLibrary returns the library compiled from the shader ahead of
// time, or 0 when it was compiled from another shader.metal (go generate
// was not run since it changed) or Metal cannot load it.
func (r *Renderer) compiledLibrary() id {
	if gpu.SourceSum(shaderSource) != shaderLibrarySum {
		return 0
	}
	return r.libraryFromData(shaderLibrary)
}

// libraryFromData returns a library loaded from the code of one, or 0
// when Metal cannot load it.
func (r *Renderer) libraryFromData(code []byte) id {
	if dispatchDataCreate == 0 || !respondsTo(r.device, "newLibraryWithData:error:") {
		return 0
	}
	// Without a destructor, dispatch_data_create copies the bytes.
	data, _, _ := purego.SyscallN(dispatchDataCreate, uintptr(unsafe.Pointer(&code[0])), uintptr(len(code)), 0, 0)
	if data == 0 {
		return 0
	}
	defer purego.SyscallN(dispatchRelease, data)
	var errObj id
	return send(r.device, "newLibraryWithData:error:", data, uintptr(unsafe.Pointer(&errObj)))
}

func respondsTo(obj id, method string) bool {
	return byte(send(obj, "respondsToSelector:", uintptr(sel(method)))) != 0
}

// newTexture returns an owned w×h texture holding pix, rows of stride
// bytes, when given.
func (r *Renderer) newTexture(w, h int, format, usage uint, pix []byte, stride int) id {
	desc := send(class("MTLTextureDescriptor"), "texture2DDescriptorWithPixelFormat:width:height:mipmapped:", uintptr(format), uintptr(w), uintptr(h), 0)
	if desc == 0 {
		return 0
	}
	send(desc, "setUsage:", uintptr(usage))
	if usage&usageRenderTarget != 0 {
		send(desc, "setStorageMode:", storageManaged)
	}
	tex := send(r.device, "newTextureWithDescriptor:", desc)
	if tex != 0 && len(pix) > 0 {
		msgReplaceRegion(tex, sel("replaceRegion:mipmapLevel:withBytes:bytesPerRow:"), mtlRegion{W: uint(w), H: uint(h), D: 1}, 0, unsafe.Pointer(&pix[0]), uint(stride))
	}
	return tex
}

// syncAtlas uploads what changed in an atlas since the texture had it.
func (r *Renderer) syncAtlas(t *texture, a *scene.Atlas, format uint) error {
	if a == nil {
		return nil
	}
	if t.tex == 0 || t.gen != a.Generation() || t.w != a.W || t.h != a.H {
		release(&t.tex)
		if t.tex = r.newTexture(a.W, a.H, format, usageShaderRead, a.Pix, a.W*a.BPP); t.tex == 0 {
			return errors.New("metal: cannot create an atlas texture")
		}
		t.w, t.h, t.gen, t.ver = a.W, a.H, a.Generation(), a.Version()
		return nil
	}
	rects, full := a.Changes(t.gen, t.ver)
	if full {
		rects = []image.Rectangle{image.Rect(0, 0, a.W, a.H)}
	}
	if len(rects) > 0 {
		r.waitLast() // frames in flight read this texture
	}
	for _, rc := range rects {
		src := a.Pix[(rc.Min.Y*a.W+rc.Min.X)*a.BPP:]
		region := mtlRegion{X: uint(rc.Min.X), Y: uint(rc.Min.Y), W: uint(rc.Dx()), H: uint(rc.Dy()), D: 1}
		msgReplaceRegion(t.tex, sel("replaceRegion:mipmapLevel:withBytes:bytesPerRow:"), region, 0, unsafe.Pointer(&src[0]), uint(a.W*a.BPP))
	}
	t.ver = a.Version()
	return nil
}

// imageTexture returns the texture of an image, uploading it when new or
// changed.
func (r *Renderer) imageTexture(img *scene.Image) uintptr {
	if img.Scene != nil {
		// A canvas, drawn before the scene showing it.
		if c := r.canvases[img.ID()]; c != nil {
			return c.tex
		}
		return 0
	}
	t := r.images[img.ID()]
	if t == nil {
		tex := r.newTexture(img.W, img.H, pixelFormatRGBA8Unorm, usageShaderRead, img.Pix, img.W*4)
		if tex == 0 {
			return 0
		}
		t = &texture{tex: tex, w: img.W, h: img.H, ver: img.Version()}
		r.images[img.ID()] = t
	} else if t.w != img.W || t.h != img.H {
		tex := r.newTexture(img.W, img.H, pixelFormatRGBA8Unorm, usageShaderRead, img.Pix, img.W*4)
		if tex == 0 {
			return 0
		}
		release(&t.tex)
		t.tex, t.w, t.h, t.ver = tex, img.W, img.H, img.Version()
	} else if t.ver != img.Version() {
		r.waitLast() // frames in flight read this texture
		msgReplaceRegion(t.tex, sel("replaceRegion:mipmapLevel:withBytes:bytesPerRow:"), mtlRegion{W: uint(img.W), H: uint(img.H), D: 1}, 0, unsafe.Pointer(&img.Pix[0]), uint(img.W*4))
		t.ver = img.Version()
	}
	t.lastFrame = r.frame
	return t.tex
}

// ringSize bounds frames in flight. Each slot owns the main target's and
// every canvas's instance buffer, so CPU encoding can overlap GPU drawing.
// A slot is reused only after its previous frame completes; in-place atlas
// and image updates wait for all slots. Two slots bound the extra queueing
// when the GPU is slower than the display. Native presentation can still
// wait for a drawable, independently of these buffers.
const ringSize = 2

// waitSlot waits for the GPU to finish the frame that last used the slot's
// instance buffer.
func (r *Renderer) waitSlot(slot int) {
	cb := r.inflight[slot]
	if cb == 0 {
		return
	}
	send(cb, "waitUntilCompleted")
	if int(send(cb, "status")) == statusError && r.err == nil {
		r.err = fmt.Errorf("metal: a frame failed: %s", describe(send(cb, "error")))
	}
	release(&r.inflight[slot])
}

// waitLast waits for the GPU to finish the frames in flight, so that this
// one can change the textures they read, or the renderer free what they use.
func (r *Renderer) waitLast() {
	for i := range r.inflight {
		// The newest first: the queue finishes frames in order, so the
		// others are done when it is.
		r.waitSlot((r.slot + ringSize - 1 - i) % ringSize)
	}
}

// track records cb, committed, as the frame drawn with the instance buffer
// encode used, and moves on to the next.
func (r *Renderer) track(cb id) {
	r.inflight[r.inUse] = send(cb, "retain")
	r.slot = (r.inUse + 1) % ringSize
}

// encode encodes drawing s into target, returning the autoreleased
// command buffer, not yet committed.
func (r *Renderer) encode(s *scene.Scene, target id) (id, error) {
	if err := r.initGPU(); err != nil {
		return 0, err
	}
	if r.cur.pipeline == 0 {
		if err := r.makePipelines(r.cur); err != nil {
			return 0, err
		}
	}
	r.inUse = r.slot
	r.waitSlot(r.inUse)
	if r.err != nil {
		return 0, r.err
	}
	r.frame++
	if err := r.syncAtlas(&r.mask, s.MaskAtlas, pixelFormatR8Unorm); err != nil {
		return 0, err
	}
	if err := r.syncAtlas(&r.color, s.ColorAtlas, pixelFormatRGBA8Unorm); err != nil {
		return 0, err
	}
	r.pipes.fallbacks = 0
	r.used = 0
	cb := send(r.queue, "commandBuffer")
	if !r.checked {
		if err := need(cb, "command buffer", "renderCommandEncoderWithDescriptor:", "commit", "presentDrawable:", "waitUntilScheduled", "waitUntilCompleted", "status", "blitCommandEncoder"); err != nil {
			return 0, err
		}
	}
	// Canvases first, into textures of their own, which the frame samples.
	var err error
	s.EachCanvas(func(c *scene.Image) {
		if err == nil {
			err = r.drawCanvas(cb, c)
		}
	})
	if err != nil {
		return 0, err
	}
	r.b.Wide = r.cur.format == pixelFormatRGBA16Float
	if err := r.drawInto(cb, s, target, &r.drawState); err != nil {
		return 0, err
	}
	return cb, nil
}

// drawCanvas draws a canvas into its texture, unless it holds its version
// already.
func (r *Renderer) drawCanvas(cb id, c *scene.Image) error {
	if r.canvases == nil {
		r.canvases = map[uint64]*canvasTexture{}
	}
	t := r.canvases[c.ID()]
	if t == nil {
		t = &canvasTexture{canvas: weak.Make(c)}
		r.canvases[c.ID()] = t
	}
	t.lastFrame = r.frame
	if t.tex != 0 && t.w == c.W && t.h == c.H && t.ver == c.Version() {
		return nil
	}
	if t.tex == 0 || t.w != c.W || t.h != c.H {
		release(&t.tex)
		if t.tex = r.newTexture(c.W, c.H, pixelFormatBGRA8Unorm, usageRenderTarget|usageShaderRead, nil, 0); t.tex == 0 {
			return errors.New("metal: cannot create a canvas's texture")
		}
		t.w, t.h = c.W, c.H
	}
	// In BGRA8, in sRGB, whatever the frame's format.
	saved := r.cur
	defer func() { r.cur = saved }()
	r.cur = &r.formats[0]
	if r.cur.pipeline == 0 {
		if err := r.makePipelines(r.cur); err != nil {
			return err
		}
	}
	cs := c.Scene
	cs.Width, cs.Height = c.W, c.H
	t.b.Wide = false
	t.b.Opaque, t.b.Classes = r.b.Opaque, r.b.Classes
	t.b.NoCull, t.b.NoShadowSplit, t.b.NoRingSplit = r.b.NoCull, r.b.NoShadowSplit, r.b.NoRingSplit
	t.b.NoClasses, t.b.GeneralClasses = r.b.NoClasses, r.b.GeneralClasses
	t.b.GeneralFill, t.b.NoClipSkip = r.b.GeneralFill, r.b.NoClipSkip
	if err := r.drawInto(cb, cs, t.tex, &t.drawState); err != nil {
		return err
	}
	t.ver = c.Version()
	return nil
}

// drawInto encodes drawing s into target with d, in the format r.cur is.
func (r *Renderer) drawInto(cb id, s *scene.Scene, target id, d *drawState) error {
	d.b.ClipSkip = true
	d.b.BlurredImage = func(img *scene.Image, bk scene.Backdrop) uintptr { return r.blurImage(cb, img, bk) }
	s = d.b.Build(s, r.imageTexture)
	d.b.BlurredImage = nil
	if err := r.syncStops(d); err != nil {
		return err
	}
	instBuf := &d.instBufs[r.inUse]
	instCap := &d.instCaps[r.inUse]
	if n := len(d.b.Instances); n > *instCap {
		release(instBuf)
		capacity := max(n*3/2, 1024)
		*instBuf = send(r.device, "newBufferWithLength:options:", uintptr(capacity*gpu.InstanceSize), 0) // shared storage
		if *instBuf == 0 {
			*instCap = 0
			return errors.New("metal: cannot create the instance buffer")
		}
		*instCap = capacity
	}
	if n := len(d.b.Instances); n > 0 {
		dst := unsafe.Slice((*gpu.Instance)(ptr(send(*instBuf, "contents"))), n)
		copy(dst, d.b.Instances)
	}

	if err := r.fitBackdrop(&d.b); err != nil {
		return err
	}
	enc, err := r.begin(cb, target, loadActionClear, s, &d.b)
	if err != nil {
		return err
	}
	bound, pipe := r.empty, r.cur.pipeline
	for _, b := range d.b.Batches {
		// Metal requires scissor rectangles within the target.
		sc := b.Scissor
		sc.Left, sc.Top = max(sc.Left, 0), max(sc.Top, 0)
		sc.Right, sc.Bottom = min(sc.Right, int32(s.Width)), min(sc.Bottom, int32(s.Height))
		if b.Count == 0 || sc.Empty() {
			continue
		}
		var want id
		switch {
		case b.Effect != nil:
			if want = r.effectPipeline(b.Effect); want == 0 {
				continue
			}
		case b.Hole:
			want = r.pipeFor(r.cur, b.Class, b.Continuous, blendHole)
		case b.Opaque:
			want = r.pipeFor(r.cur, b.Class, b.Continuous, blendNone)
		default:
			want = r.pipeFor(r.cur, b.Class, b.Continuous, blendOver)
		}
		if b.Backdrop != 0 {
			// The effect shows what is drawn so far.
			send(enc, "endEncoding")
			r.readBackdrop(cb, target, d.b.Backdrops[b.Backdrop-1])
			if enc, err = r.begin(cb, target, loadActionLoad, s, &d.b); err != nil {
				return err
			}
			send(enc, "setFragmentTexture:atIndex:", r.cur.backdrop[0].tex, 3)
			bound, pipe = r.empty, r.cur.pipeline
		}
		if want != pipe {
			pipe = want
			send(enc, "setRenderPipelineState:", pipe)
		}
		if b.Image != 0 && b.Image != bound {
			bound = b.Image
			send(enc, "setFragmentTexture:atIndex:", bound, 2)
		}
		send(enc, "setFragmentTexture:atIndex:", or(id(b.ClipMask), r.empty), 4)
		if b.Shader != nil {
			r.bindShader(enc, b)
		}
		offset := uintptr(b.Start * gpu.InstanceSize)
		send(enc, "setVertexBuffer:offset:atIndex:", *instBuf, offset, 0)
		send(enc, "setFragmentBuffer:offset:atIndex:", *instBuf, offset, 0)
		msgSetScissor(enc, sel("setScissorRect:"), mtlScissorRect{uint(sc.Left), uint(sc.Top), uint(sc.Right - sc.Left), uint(sc.Bottom - sc.Top)})
		send(enc, "drawPrimitives:vertexStart:vertexCount:instanceCount:", primitiveTriStrip, 0, 4, uintptr(b.Count))
	}
	send(enc, "endEncoding")
	return nil
}

// maxBytes is the most setFragmentBytes takes; larger uniforms go in a
// buffer of their own.
const maxBytes = 4096

// bindShader binds what the shader of batch b reads: its uniforms, laid
// out as its effect says, in buffer 2, and its images in textures 5 to 8.
func (r *Renderer) bindShader(enc id, b gpu.Batch) {
	r.shaderBytes = b.Effect.Shader.Pack(r.shaderBytes, b.Uniforms)
	if n := len(r.shaderBytes); n > 0 && n <= maxBytes {
		send(enc, "setFragmentBytes:length:atIndex:", uintptr(unsafe.Pointer(&r.shaderBytes[0])), uintptr(n), 2)
	} else if n > 0 {
		buf := send(r.device, "newBufferWithBytes:length:options:", uintptr(unsafe.Pointer(&r.shaderBytes[0])), uintptr(n), 0)
		send(enc, "setFragmentBuffer:offset:atIndex:", buf, 0, 2)
		release(&buf) // the encoder keeps it
	}
	for i, t := range b.Textures {
		if i == gpu.BackdropImage && b.Backdrop != 0 {
			// A shader reading its backdrop reads it as this image.
			t = uintptr(r.cur.backdrop[0].tex)
		}
		send(enc, "setFragmentTexture:atIndex:", or(id(t), r.empty), uintptr(5+i))
	}
}

// begin starts encoding the drawing of s into target, which it clears or
// keeps (load), with the state the instances draw with.
func (r *Renderer) begin(cb, target id, load uintptr, s *scene.Scene, b *gpu.Builder) (id, error) {
	rpd := send(class("MTLRenderPassDescriptor"), "renderPassDescriptor")
	ca := send(send(rpd, "colorAttachments"), "objectAtIndexedSubscript:", 0)
	send(ca, "setTexture:", target)
	send(ca, "setLoadAction:", load)
	send(ca, "setStoreAction:", storeActionStore)
	c := s.Clear.Premul(1)
	msgSetClearColor(ca, sel("setClearColor:"), mtlClearColor{float64(c[0]), float64(c[1]), float64(c[2]), float64(c[3])})
	enc := send(cb, "renderCommandEncoderWithDescriptor:", rpd)
	if !r.checked {
		if err := need(enc, "render encoder", "setRenderPipelineState:", "setVertexBytes:length:atIndex:", "setVertexBuffer:offset:atIndex:",
			"setFragmentBuffer:offset:atIndex:", "setFragmentBytes:length:atIndex:", "setFragmentTexture:atIndex:", "setFragmentSamplerState:atIndex:",
			"setScissorRect:", "drawPrimitives:vertexStart:vertexCount:instanceCount:", "endEncoding"); err != nil {
			if enc != 0 {
				send(enc, "endEncoding")
			}
			return 0, err
		}
		r.checked = true
	}
	send(enc, "setRenderPipelineState:", r.cur.pipeline)
	// The frame's size, and whether its target keeps colors outside the
	// sRGB gamut.
	globals := [4]float32{float32(s.Width), float32(s.Height), 0, 0}
	if b.Wide {
		globals[2] = 1
	}
	send(enc, "setVertexBytes:length:atIndex:", uintptr(unsafe.Pointer(&globals[0])), unsafe.Sizeof(globals), 1)
	send(enc, "setFragmentBytes:length:atIndex:", uintptr(unsafe.Pointer(&globals[0])), unsafe.Sizeof(globals), 1)
	send(enc, "setFragmentSamplerState:atIndex:", r.sampler, 0)
	send(enc, "setFragmentTexture:atIndex:", or(r.mask.tex, r.empty), 0)
	send(enc, "setFragmentTexture:atIndex:", or(r.color.tex, r.empty), 1)
	send(enc, "setFragmentTexture:atIndex:", r.empty, 2)
	send(enc, "setFragmentTexture:atIndex:", r.empty, 3)
	send(enc, "setFragmentTexture:atIndex:", or(r.stopsTex, r.empty), 9)
	return enc, nil
}

// syncStops uploads the stops of the gradients d's builder built into the
// texture of the frame's slot, which r.stopsTex then is, or 0 for none.
func (r *Renderer) syncStops(d *drawState) error {
	r.stopsTex = 0
	w, h := d.b.StopsSize()
	if w == 0 {
		return nil
	}
	t := &d.stops[r.inUse]
	if t.tex == 0 || t.w != w || t.h < h {
		release(&t.tex)
		if t.tex = r.newTexture(w, h, pixelFormatRGBA32Float, usageShaderRead, nil, 0); t.tex == 0 {
			*t = texture{}
			return errors.New("metal: cannot create the texture of the stops of gradients")
		}
		t.w, t.h = w, h
	}
	// The slot's last frame, which read it, is done (waitSlot).
	data := d.b.Stops
	for len(data) < 4*w*h {
		data = append(data, 0)
	}
	d.b.Stops = data[:len(d.b.Stops)]
	msgReplaceRegion(t.tex, sel("replaceRegion:mipmapLevel:withBytes:bytesPerRow:"), mtlRegion{W: uint(w), H: uint(h), D: 1}, 0, unsafe.Pointer(&data[0]), uint(w*16))
	r.stopsTex = t.tex
	return nil
}

// blurImage encodes into cb the passes blurring the area of img that bk
// says (gpu.Builder.BlurredImage) into a texture of the frame's own, in
// BGRA8 whatever the frame's format, as internal/raster computes them,
// and returns it, or 0.
func (r *Renderer) blurImage(cb id, img *scene.Image, bk scene.Backdrop) uintptr {
	src := id(r.imageTexture(img))
	w, h := bk.Size()
	f := &r.formats[0]
	if src == 0 || w <= 0 || h <= 0 {
		return 0
	}
	if f.downPipe == 0 {
		if err := r.makePipelines(f); err != nil {
			return 0
		}
	}
	for i := range r.blurTmp {
		t := &r.blurTmp[i]
		if t.w < w || t.h < h {
			release(&t.tex)
			nw, nh := max(w, t.w), max(h, t.h)
			if t.tex = r.newTexture(nw, nh, pixelFormatBGRA8Unorm, usageRenderTarget|usageShaderRead, nil, 0); t.tex == 0 {
				*t = texture{}
				return 0
			}
			t.w, t.h = nw, nh
		}
	}
	if r.used == len(r.blurred) {
		r.blurred = append(r.blurred, texture{})
	}
	out := &r.blurred[r.used]
	r.used++
	if out.tex == 0 || out.w != w || out.h != h {
		release(&out.tex)
		if out.tex = r.newTexture(w, h, pixelFormatBGRA8Unorm, usageRenderTarget|usageShaderRead, nil, 0); out.tex == 0 {
			*out = texture{}
			return 0
		}
		out.w, out.h = w, h
	}
	p := gpu.ImageDownPass(bk)
	if bk.Radius == 0 {
		r.pass(cb, f.downPipe, src, out.tex, w, h, &p)
		return uintptr(out.tex)
	}
	r.pass(cb, f.downPipe, src, r.blurTmp[0].tex, w, h, &p)
	p = gpu.ImageBlurPass(bk, [2]int32{1, 0})
	r.pass(cb, f.blurPipe, r.blurTmp[0].tex, r.blurTmp[1].tex, w, h, &p)
	p = gpu.ImageBlurPass(bk, [2]int32{0, 1})
	r.pass(cb, f.blurPipe, r.blurTmp[1].tex, out.tex, w, h, &p)
	return uintptr(out.tex)
}

// readBackdrop encodes the passes computing the backdrop bk of what
// target holds, into r.cur.backdrop[0].
func (r *Renderer) readBackdrop(cb, target id, bk scene.Backdrop) {
	f := r.cur
	w, h := bk.Size()
	p := gpu.DownPass(bk, [2]int32{})
	r.pass(cb, f.downPipe, target, f.backdrop[0].tex, w, h, &p)
	if bk.Radius == 0 {
		return
	}
	p = gpu.BlurPass(bk, [2]int32{1, 0})
	r.pass(cb, f.blurPipe, f.backdrop[0].tex, f.backdrop[1].tex, w, h, &p)
	p = gpu.BlurPass(bk, [2]int32{0, 1})
	r.pass(cb, f.blurPipe, f.backdrop[1].tex, f.backdrop[0].tex, w, h, &p)
}

// pass encodes drawing the w×h texels at the start of dst with pipe,
// which reads src.
func (r *Renderer) pass(cb, pipe, src, dst id, w, h int, p *gpu.Pass) {
	rpd := send(class("MTLRenderPassDescriptor"), "renderPassDescriptor")
	ca := send(send(rpd, "colorAttachments"), "objectAtIndexedSubscript:", 0)
	send(ca, "setTexture:", dst)
	send(ca, "setLoadAction:", loadActionDontCare)
	send(ca, "setStoreAction:", storeActionStore)
	enc := send(cb, "renderCommandEncoderWithDescriptor:", rpd)
	send(enc, "setRenderPipelineState:", pipe)
	send(enc, "setFragmentTexture:atIndex:", src, 0)
	send(enc, "setFragmentBytes:length:atIndex:", uintptr(unsafe.Pointer(p)), unsafe.Sizeof(*p), 0)
	msgSetScissor(enc, sel("setScissorRect:"), mtlScissorRect{0, 0, uint(w), uint(h)})
	send(enc, "drawPrimitives:vertexStart:vertexCount:instanceCount:", primitiveTriStrip, 0, 4, 1)
	send(enc, "endEncoding")
}

// fitBackdrop makes the textures of backdrops as large as the scene built
// needs.
func (r *Renderer) fitBackdrop(b *gpu.Builder) error {
	f := r.cur
	w, h := b.BackdropSize()
	if w == 0 || f.backdrop[0].w >= w && f.backdrop[0].h >= h {
		return nil
	}
	// Somewhat larger, so that a pane growing does not make them again
	// every frame.
	w, h = max(w+w/4, f.backdrop[0].w), max(h+h/4, f.backdrop[0].h)
	for i := range f.backdrop {
		t := &f.backdrop[i]
		release(&t.tex)
		if t.tex = r.newTexture(w, h, f.format, usageRenderTarget|usageShaderRead, nil, 0); t.tex == 0 {
			t.w, t.h = 0, 0
			return errors.New("metal: cannot create a backdrop texture")
		}
		t.w, t.h = w, h
	}
	return nil
}

// hasHoles reports whether s has holes (scene.OpHole), where it shows
// web views under it.
func hasHoles(s *scene.Scene) bool {
	for i := range s.Ops {
		if s.Ops[i].Kind == scene.OpHole {
			return true
		}
	}
	return false
}

// Behind reports whether the last Render drew nothing, to catch up with
// the display: the window draws the frame again at the next refresh.
func (r *Renderer) Behind() bool { return r.behind }

func or(a, b id) id {
	if a != 0 {
		return a
	}
	return b
}

// Render draws s into the layer and presents it.
func (r *Renderer) Render(s *scene.Scene) (err error) {
	if s.Width <= 0 || s.Height <= 0 || r.layer == 0 {
		return nil
	}
	pool(func() { err = r.render(s) })
	return err
}

// behindAfter is how long a frame presented as soon as drawn waits for its
// drawable before the next skips a refresh (Behind), at most once a
// second: a drawable frees within a millisecond or so while frames show at
// the refresh after they present.
const behindAfter = 4 * time.Millisecond

func (r *Renderer) render(s *scene.Scene) error {
	if r.err != nil {
		return r.err
	}
	scale := float64(s.Scale)
	if scale <= 0 {
		scale = 1
	}
	r.cur = &r.formats[0]
	if r.wide {
		r.cur = &r.formats[1]
	}
	changed := r.setLayerWide(r.wide)
	changed = r.fit(s.Width, s.Height, scale) || changed
	// A frame changing the layer, or showing web views, which move with
	// the window's other changes, presents with the Core Animation
	// transaction, so that they show together, as during a live resize.
	// Others present as soon as the GPU has them, sparing a commit of the
	// transaction for each.
	holes := hasHoles(s)
	sync := changed || holes || r.holes
	r.holes = holes
	now := time.Now()
	r.behind = !sync && r.waited && now.Sub(r.lastRender) < 50*time.Millisecond && now.Sub(r.skippedAt) > time.Second
	r.waited = false
	if r.behind {
		// The last frame waited for a drawable, as the display held the
		// others until its next refresh: the frames following it would wait as
		// long, most of each refresh, blocking the main thread. Drawing
		// none for a refresh catches up.
		r.skippedAt = now
		return nil
	}
	if sync != r.synced {
		on := uintptr(0)
		if sync {
			on = 1
		}
		send(r.layer, "setPresentsWithTransaction:", on)
		r.synced = sync
	}
	drawable := send(r.layer, "nextDrawable")
	if drawable == 0 {
		return nil // none came within a second: skip the frame
	}
	r.waited = !sync && time.Since(now) > behindAfter
	texture := send(drawable, "texture")
	cb, err := r.encode(s, texture)
	if err != nil {
		return err
	}
	if sync {
		send(cb, "commit")
		send(cb, "waitUntilScheduled")
		send(drawable, "present")
	} else {
		send(cb, "presentDrawable:", drawable)
		send(cb, "commit")
	}
	r.track(cb)
	r.lastRender = time.Now()
	r.armTrim()
	// Forget the textures of images no frame drew for a while, and those
	// of canvases gone.
	if r.frame%120 == 0 {
		for key, t := range r.images {
			if r.frame-t.lastFrame > 240 {
				release(&t.tex)
				delete(r.images, key)
			}
		}
		for key, t := range r.canvases {
			if t.canvas.Value() == nil {
				release(&t.tex)
				t.drawState.release()
				delete(r.canvases, key)
			}
		}
	}
	return nil
}

// fit sizes the layer and its drawables for frames of w×h pixels at
// scale, and as its view, and reports whether they changed.
func (r *Renderer) fit(w, h int, scale float64) (changed bool) {
	bounds := msgRect(r.superlayer, sel("bounds"))
	if w == r.w && h == r.h && scale == r.scale && bounds == r.bounds && !r.trimmed {
		return false
	}
	// Without animating the change, as layers do by default.
	tx := class("CATransaction")
	send(tx, "begin")
	send(tx, "setDisableActions:", 1)
	msgSetRect(r.layer, sel("setFrame:"), bounds)
	msgSetFloat(r.layer, sel("setContentsScale:"), scale)
	msgSetSize(r.layer, sel("setDrawableSize:"), cgSize{float64(w), float64(h)})
	send(tx, "commit")
	r.w, r.h, r.scale, r.bounds, r.trimmed = w, h, scale, bounds, false
	return true
}

// releaseTextures frees what the GPU's frames use, once the window is
// idle. The next frame makes them again, uploading the atlases whole.
func (r *Renderer) releaseTextures() {
	release(&r.mask.tex)
	release(&r.color.tex)
	for key, t := range r.images {
		release(&t.tex)
		delete(r.images, key)
	}
	for key, t := range r.canvases {
		release(&t.tex)
		t.drawState.release()
		delete(r.canvases, key)
	}
	r.drawState.release()
	for i := range r.blurred {
		release(&r.blurred[i].tex)
	}
	r.blurred, r.used = nil, 0
	for i := range r.blurTmp {
		release(&r.blurTmp[i].tex)
		r.blurTmp[i] = texture{}
	}
	for i := range r.formats {
		for j := range r.formats[i].backdrop {
			release(&r.formats[i].backdrop[j].tex)
			r.formats[i].backdrop[j] = texture{}
		}
	}
	r.b.Release()
	r.b.Classes = true
	r.b.Opaque = true // Release empties the builder
}

// trimAfter is how long a window draws nothing before its spare drawables
// go: about when the Metal driver frees its own memory of frames.
const trimAfter = 2 * time.Second

// armTrim has the renderer's timer trim the drawables once it draws
// nothing for trimAfter.
func (r *Renderer) armTrim() {
	if r.trimArmed {
		return
	}
	if r.trimTimer == 0 {
		lastTrimHandle++
		r.trimHandle = lastTrimHandle
		ctx := cfTimerContext{info: r.trimHandle}
		// Its first fire date comes below; it repeats only in theory.
		r.trimTimer = cfRunLoopTimerCreate(0, cfAbsoluteTimeGetCurrent()+1e9, 1e9, 0, 0, trimCallback, &ctx)
		if r.trimTimer == 0 {
			return
		}
		trimTimers[r.trimHandle] = r
		mainLoop, _, _ := purego.SyscallN(cfRunLoopGetMain)
		purego.SyscallN(cfRunLoopAddTimer, mainLoop, r.trimTimer, commonModes)
	}
	cfRunLoopTimerSetNextFireDate(r.trimTimer, cfAbsoluteTimeGetCurrent()+trimAfter.Seconds())
	r.trimArmed = true
}

// trimIfIdle trims the drawables if the window drew nothing for trimAfter,
// and waits for that time otherwise.
func (r *Renderer) trimIfIdle() {
	if wait := trimAfter - time.Since(r.lastRender); wait > 0 {
		cfRunLoopTimerSetNextFireDate(r.trimTimer, cfAbsoluteTimeGetCurrent()+wait.Seconds())
		return
	}
	r.trimArmed = false
	cfRunLoopTimerSetNextFireDate(r.trimTimer, cfAbsoluteTimeGetCurrent()+1e9)
	r.trim()
}

// trim gives back the drawables the layer does not show. Shrinking them
// frees those, and Core Animation keeps showing the last frame; the next
// makes them again at full size.
func (r *Renderer) trim() {
	if r.layer == 0 || r.trimmed {
		return
	}
	pool(func() {
		r.waitLast()
		r.releaseTextures()
		tx := class("CATransaction")
		send(tx, "begin")
		send(tx, "setDisableActions:", 1)
		msgSetSize(r.layer, sel("setDrawableSize:"), cgSize{1, 1})
		send(tx, "commit")
	})
	r.trimmed = true
}

// renderOffscreen draws s into a texture of its own and returns its
// premultiplied BGRA rows, for tests.
func (r *Renderer) renderOffscreen(s *scene.Scene) (pix []byte, err error) {
	pool(func() {
		r.waitLast()
		r.cur = &r.formats[0]
		tex := r.newTexture(s.Width, s.Height, pixelFormatBGRA8Unorm, usageRenderTarget|usageShaderRead, nil, 0)
		if tex == 0 {
			err = errors.New("metal: cannot create a target")
			return
		}
		defer release(&tex)
		var cb id
		if cb, err = r.encode(s, tex); err != nil {
			return
		}
		blit := send(cb, "blitCommandEncoder")
		send(blit, "synchronizeResource:", tex)
		send(blit, "endEncoding")
		send(cb, "commit")
		send(cb, "waitUntilCompleted")
		if int(send(cb, "status")) == statusError {
			err = fmt.Errorf("metal: the frame failed: %s", describe(send(cb, "error")))
			return
		}
		pix = make([]byte, s.Width*s.Height*4)
		msgGetBytes(tex, sel("getBytes:bytesPerRow:fromRegion:mipmapLevel:"), unsafe.Pointer(&pix[0]), uint(s.Width*4),
			mtlRegion{W: uint(s.Width), H: uint(s.Height), D: 1}, 0)
	})
	return pix, err
}

// Release frees the renderer's GPU objects and takes its layer out.
func (r *Renderer) Release() {
	r.b.Release()
	if r.trimTimer != 0 {
		purego.SyscallN(cfRunLoopTimerInvalidate, r.trimTimer)
		purego.SyscallN(cfRelease, r.trimTimer)
		delete(trimTimers, r.trimHandle)
		r.trimTimer = 0
	}
	r.stopPipelines()
	pool(func() {
		r.waitLast()
		if r.layer != 0 {
			tx := class("CATransaction")
			send(tx, "begin")
			send(tx, "setDisableActions:", 1)
			send(r.layer, "removeFromSuperlayer")
			send(tx, "commit")
			release(&r.layer)
		}
		r.releaseTextures()
		release(&r.empty)
		release(&r.sampler)
		for i := range r.formats {
			f := &r.formats[i]
			release(&f.pipeline)
			release(&f.holePipe)
			release(&f.opaquePipe)
			release(&f.downPipe)
			release(&f.blurPipe)
			for _, p := range f.effects {
				release(&p.pipe)
			}
		}
		release(&r.lib)
		release(&r.queue)
		release(&r.device)
	})
}

// FrameInfo describes the last frame for MYGO_FRAME_STATS.
func (r *Renderer) FrameInfo() string {
	info := r.b.Summary()
	if n := r.pipes.fallbacks; n > 0 {
		info += fmt.Sprintf(", %d batches awaiting specialized pipelines", n)
	}
	return info
}
