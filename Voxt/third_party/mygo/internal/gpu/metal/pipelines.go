//go:build darwin

package metal

import (
	"fmt"
	"os"
	"sync"

	"github.com/egoist/mygo/internal/gpu"
)

// The pipelines of the classes of batches (gpu.Class) are specialized from the
// shader at creation, which costs a compile of up to 120 ms. A frame never
// waits for one: pipeFor returns the general pipeline, which draws the same
// pixels, until a worker goroutine has made the specialized one, and a cache
// shared by every renderer of a device keeps them, so that a second window
// finds them made.

// pipeKey names a pipeline: the pixel format of the target, the class and
// whether its instances have continuous corners, and how it blends.
type pipeKey struct {
	format uint
	class  gpu.Class
	cont   bool
	mode   blendMode
}

// deviceKey is a pipeline of a device in the cache of all renderers.
type deviceKey struct {
	device id
	pipeKey
}

// pipeCache holds the specialized pipelines of each device, never released:
// they are small, and a device lives as long as the process does.
var pipeCache = struct {
	sync.Mutex
	m map[deviceKey]id
}{m: map[deviceKey]id{}}

// syncPipelines makes pipeFor build a missing pipeline at once, for tests that
// compare the specialized pipelines with the general ones and need them made.
var syncPipelines bool

// pipeState is a renderer's pipelines of the classes and the worker making them.
type pipeState struct {
	mu      sync.Mutex
	ready   map[pipeKey]id
	queued  map[pipeKey]bool
	work    chan pipeKey
	wg      sync.WaitGroup
	closed  bool
	sync    bool    // build in pipeFor, see syncPipelines
	warmed  [2]bool // by format
	owned   []id    // pipelines of renderers that do not use the cache
	nbuilt  int     // pipelines the worker made, for tests
	nfailed int

	beforeBuild func() // test hook, set before the first frame starts the worker
	// fallbacks counts the batches of the frame being encoded that drew with
	// the general pipeline because the specialized one was not made yet
	// (touched by the encoding goroutine only).
	fallbacks int
}

// warmSet returns the pipelines that frames most likely draw with in a target of
// format: every class with and without continuous corners drawn over what is
// there, and the plain and other fills drawn as opaque cores (which float16
// targets do not use, and which shadows do not need).
func warmSet(format uint) (keys []pipeKey) {
	for _, cl := range []gpu.Class{gpu.ClassPlain, gpu.ClassGlyph, gpu.ClassImage, gpu.ClassFill, gpu.ClassShadow, gpu.ClassSubpixel} {
		for _, cont := range []bool{false, true} {
			if specializable(format, cl, blendOver) {
				keys = append(keys, pipeKey{format, cl, cont, blendOver})
			}
			if (cl == gpu.ClassPlain || cl == gpu.ClassFill) && specializable(format, cl, blendNone) {
				keys = append(keys, pipeKey{format, cl, cont, blendNone})
			}
		}
	}
	return keys
}

// specializable reports whether batches of class cl blended as mode draw with a
// specialized pipeline in targets of format.
func specializable(format uint, cl gpu.Class, mode blendMode) bool {
	if cl == gpu.ClassGeneral || cl >= gpu.NumClasses {
		return false
	}
	if format == pixelFormatRGBA16Float {
		// A shadow's blur (erf, exp) compiles to results one unit in the last
		// place of a half apart in float16 targets, at a pixel in a scene, so
		// shadows keep the general function there; opaque cores are blended
		// there (see makePipelines).
		return cl != gpu.ClassShadow && mode != blendNone
	}
	return true
}

// startPipelines starts the worker that makes pipelines, if it is not running,
// and has it warm the likely ones of f's format.
func (r *Renderer) startPipelines(f *formatState) {
	p := &r.pipes
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return
	}
	if p.work == nil {
		p.ready, p.queued = map[pipeKey]id{}, map[pipeKey]bool{}
		p.work = make(chan pipeKey, 128)
		p.sync = syncPipelines
		p.wg.Add(1)
		go r.pipelineWorker()
	}
	i := 0
	if f.format == pixelFormatRGBA16Float {
		i = 1
	}
	if p.warmed[i] || p.sync {
		return
	}
	p.warmed[i] = true
	for _, k := range warmSet(f.format) {
		r.enqueueLocked(k)
	}
}

// enqueueLocked asks the worker for the pipeline k, unless it is made or asked
// for. It never blocks: a full queue drops the request, which the next frame
// makes again.
func (r *Renderer) enqueueLocked(k pipeKey) {
	p := &r.pipes
	if p.closed || p.queued[k] || p.ready[k] != 0 {
		return
	}
	select {
	case p.work <- k:
		p.queued[k] = true
	default:
	}
}

// pipelineWorker makes the pipelines asked for, until the renderer is released.
func (r *Renderer) pipelineWorker() {
	defer r.pipes.wg.Done()
	for k := range r.pipes.work {
		r.pipes.mu.Lock()
		closed := r.pipes.closed
		r.pipes.mu.Unlock()
		if closed {
			return // Release waits for the active compile, not the warmup backlog.
		}
		pool(func() { r.buildPipeline(k) })
	}
}

// buildPipeline makes the pipeline k, or takes it from the cache of the device,
// and publishes it.
func (r *Renderer) buildPipeline(k pipeKey) {
	p := &r.pipes
	if p.beforeBuild != nil {
		p.beforeBuild()
	}
	dk := deviceKey{r.device, k}
	cacheable := r.source == ""
	var pipe id
	if cacheable {
		pipeCache.Lock()
		pipe = pipeCache.m[dk]
		pipeCache.Unlock()
	}
	var err error
	if pipe == 0 {
		pipe, err = r.newClassPipeline(r.lib, "ps", k.format, k.mode, k.class, k.cont)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%v: drawing with the general shader\n", err)
		} else if cacheable {
			pipeCache.Lock()
			if cached := pipeCache.m[dk]; cached != 0 {
				// Another window finished the same pipeline while this one compiled.
				release(&pipe)
				pipe = cached
			} else {
				pipeCache.m[dk] = pipe
			}
			pipeCache.Unlock()
		}
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	delete(p.queued, k)
	if err != nil {
		p.nfailed++
		p.ready[k] = ^id(0) // tried: draws with the general pipeline
		return
	}
	if !cacheable {
		// Owned by this renderer: released with it.
		p.owned = append(p.owned, pipe)
	}
	p.nbuilt++
	p.ready[k] = pipe
}

// pipeFor returns the pipeline of f that draws batches of class cl blended as
// mode says. It never waits for a compile: until the specialized pipeline is
// made, by the worker or by another renderer of the device, it is the general one,
// which draws the same pixels.
func (r *Renderer) pipeFor(f *formatState, cl gpu.Class, continuous bool, mode blendMode) id {
	if mode == blendNone && f.format == pixelFormatRGBA16Float {
		mode = blendOver // see makePipelines
	}
	general := f.pipeline
	switch mode {
	case blendHole:
		general = f.holePipe
	case blendNone:
		general = f.opaquePipe
	}
	if !specializable(f.format, cl, mode) {
		return general
	}
	k := pipeKey{f.format, cl, continuous, mode}
	p := &r.pipes
	p.mu.Lock()
	pipe, known := p.ready[k]
	closed := p.closed
	p.mu.Unlock()
	if known {
		if pipe == ^id(0) {
			return general
		}
		return pipe
	}
	if closed || r.lib == 0 {
		return general
	}
	p.fallbacks++ // until a pipeline is found made, below, or made in sync mode
	if p.sync {
		r.buildPipeline(k)
		p.mu.Lock()
		pipe = p.ready[k]
		p.mu.Unlock()
		p.fallbacks--
		if pipe == 0 || pipe == ^id(0) {
			return general
		}
		return pipe
	}
	// Another renderer of the device may have made it.
	pipeCache.Lock()
	pipe = pipeCache.m[deviceKey{r.device, k}]
	pipeCache.Unlock()
	p.mu.Lock()
	if pipe != 0 && r.source == "" {
		p.ready[k] = pipe
	} else {
		r.enqueueLocked(k)
		pipe = 0
	}
	p.mu.Unlock()
	if pipe != 0 {
		p.fallbacks--
	}
	return or(pipe, general)
}

// stopPipelines ends the worker and waits for it, before the library it uses
// is released.
func (r *Renderer) stopPipelines() {
	p := &r.pipes
	p.mu.Lock()
	if p.work != nil && !p.closed {
		close(p.work)
	}
	p.closed = true
	p.mu.Unlock()
	p.wg.Wait()
	for i := range p.owned {
		release(&p.owned[i])
	}
	p.owned = nil
}

// pipelineCounts returns how many specialized pipelines the worker made and how
// many failed to build, for tests.
func (r *Renderer) pipelineCounts() (built, failed int) {
	r.pipes.mu.Lock()
	defer r.pipes.mu.Unlock()
	return r.pipes.nbuilt, r.pipes.nfailed
}

// usedSpecialized reports whether the pipeline k is made, and is a specialized
// one, not the general pipeline a failed build falls back to, for tests.
func (r *Renderer) usedSpecialized(k pipeKey) bool {
	r.pipes.mu.Lock()
	defer r.pipes.mu.Unlock()
	p := r.pipes.ready[k]
	return p != 0 && p != ^id(0)
}
