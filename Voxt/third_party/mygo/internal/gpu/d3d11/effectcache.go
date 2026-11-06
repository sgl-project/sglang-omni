//go:build windows

package d3d11

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/egoist/mygo/internal/gpu"
)

// PrecompileEffect starts compiling the pixel shader of an effect whose
// shader is src, which nothing compiled ahead of time, on a goroutine of
// its own: the renderers then draw it with that code, waiting for the
// compilation if it has not finished, rather than each compiling it on
// its first draw. The code is kept on disk too, for the next launches.
// Package ui/shader calls it as it compiles a shader, which links the
// cache into apps that draw shaders only.
func PrecompileEffect(src string) {
	effects.once.Do(func() {
		effects.byKey = map[string]*effectEntry{}
		effects.sem = make(chan struct{}, max(1, runtime.NumCPU()-1))
		effects.dir = effectCacheDir()
		code := effects.code
		sharedEffectCode.Store(&code)
	})
	e := effects.entry(src)
	go func() {
		effects.sem <- struct{}{}
		defer func() { <-effects.sem }()
		e.run()
	}()
}

// sharedEffectCode returns the code of an effect's pixel shader from the
// cache, once PrecompileEffect made it, from any goroutine.
var sharedEffectCode atomic.Pointer[func(src string) ([]byte, error)]

// compileEffectCode compiles the pixel shader of an effect whose code
// compiled ahead of time does not match its source: CompileEffect, until
// PrecompileEffect shares the code between the renderers.
func compileEffectCode(src string) ([]byte, error) {
	if f := sharedEffectCode.Load(); f != nil {
		return (*f)(src)
	}
	return CompileEffect(src)
}

// effectCache is the code of effects' pixel shaders, by their source.
type effectCache struct {
	once  sync.Once
	mu    sync.Mutex
	byKey map[string]*effectEntry
	sem   chan struct{} // the compilations on goroutines of their own
	dir   string        // where the code is kept, "" for nowhere
}

// effects is the cache PrecompileEffect and the renderers share.
var effects effectCache

// effectEntry is the code of an effect's pixel shader, compiled once.
type effectEntry struct {
	c    *effectCache
	src  string
	once sync.Once
	code []byte
	err  error
}

// entry returns the entry of src, created the first time.
func (c *effectCache) entry(src string) *effectEntry {
	c.mu.Lock()
	defer c.mu.Unlock()
	if e := c.byKey[src]; e != nil {
		return e
	}
	e := &effectEntry{c: c, src: src}
	c.byKey[src] = e
	return e
}

// code returns the code of src, compiling it unless a goroutine is, then
// waiting for it.
func (c *effectCache) code(src string) ([]byte, error) {
	e := c.entry(src)
	e.run()
	return e.code, e.err
}

// effectFlags names how effects' code compiles in the names of the files
// keeping it: the target and the flags of compile.
const effectFlags = "ps_4_0 effectps O3 IEEE\n"

// run compiles e once, or reads its code from the disk.
func (e *effectEntry) run() {
	e.once.Do(func() {
		full := EffectSource(e.src)
		var path string
		if e.c.dir != "" {
			path = filepath.Join(e.c.dir, gpu.SourceSum(effectFlags+full)+".dxbc")
			if b, err := os.ReadFile(path); err == nil && validDXBC(b) {
				now := time.Now()
				os.Chtimes(path, now, now) // the least recently used go first
				e.code = b
				return
			}
		}
		e.code, e.err = compile(full, "effectps", "ps_4_0")
		if e.err == nil && path != "" {
			saveEffectCode(path, e.code)
		}
	})
}

// validDXBC reports whether b looks like whole DXBC bytecode: its magic,
// and the size its header gives. Direct3D checks its checksum.
func validDXBC(b []byte) bool {
	return len(b) >= 32 && string(b[:4]) == "DXBC" && int(binary.LittleEndian.Uint32(b[24:])) == len(b)
}

// effectCacheDir returns the directory keeping effects' code: one for the
// user, as Windows' own shader cache (D3DSCache) is, since the files'
// names are the SHA-256 of what they compiled from.
func effectCacheDir() string {
	dir, err := os.UserCacheDir()
	if err != nil {
		return ""
	}
	return filepath.Join(dir, "MyGo", "ShaderCache")
}

// effectCacheSize is how much code the cache keeps, in bytes, before it
// removes the files least recently used, as sources change while an app
// is developed.
const effectCacheSize = 32 << 20

// saveEffectCode writes code to path, through a temporary file renamed
// as another process may be reading it, and trims the cache. It does
// without the cache on any error.
func saveEffectCode(path string, code []byte) {
	dir := filepath.Dir(path)
	if os.MkdirAll(dir, 0o755) != nil {
		return
	}
	f, err := os.CreateTemp(dir, "*.tmp")
	if err != nil {
		return
	}
	_, err = f.Write(code)
	if f.Close() != nil || err != nil || os.Rename(f.Name(), path) != nil {
		os.Remove(f.Name())
		return
	}
	trimEffectCache(dir)
}

// trimEffectCache removes the files of dir least recently used while
// they weigh more than effectCacheSize.
func trimEffectCache(dir string) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	type file struct {
		name string
		size int64
		used time.Time
	}
	var files []file
	var total int64
	for _, de := range entries {
		info, err := de.Info()
		if err != nil || !info.Mode().IsRegular() {
			continue
		}
		files = append(files, file{de.Name(), info.Size(), info.ModTime()})
		total += info.Size()
	}
	sort.Slice(files, func(i, j int) bool { return files[i].used.Before(files[j].used) })
	for _, f := range files {
		if total <= effectCacheSize {
			break
		}
		if os.Remove(filepath.Join(dir, f.name)) == nil {
			total -= f.size
		}
	}
}
