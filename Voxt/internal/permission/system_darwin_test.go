//go:build darwin

package permission

import (
	"runtime"
	"sync"
	"testing"

	"github.com/ebitengine/purego"
)

// threadID is the calling thread's id (pthread_threadid_np, 0 = self).
var threadID = func() func() uint64 {
	var once sync.Once
	var f func(uintptr, *uint64) int32
	return func() uint64 {
		once.Do(func() {
			h, err := purego.Dlopen("/usr/lib/libSystem.B.dylib", purego.RTLD_GLOBAL|purego.RTLD_NOW)
			if err != nil {
				panic(err)
			}
			purego.RegisterLibFunc(&f, h, "pthread_threadid_np")
		})
		var id uint64
		f(0, &id)
		return id
	}
}()

// TestPooledKeepsItsThread: an autorelease pool belongs to the thread
// that pushed it, and popping it on another one crashes: the sporadic
// SIGSEGV in the root package's tests, on the app's own path
// helperChanged → recheckPermissions → Microphone. While the pool is
// open, another goroutine locks the pushing thread to itself and keeps
// it; the pool must still be popped on the thread that pushed it.
func TestPooledKeepsItsThread(t *testing.T) {
	if err := load(); err != nil {
		t.Fatal(err)
	}
	// The pool's own calls record their thread instead (a real pop on
	// another thread would crash the test binary rather than fail it).
	push, pop := mac.poolPush, mac.poolPop
	t.Cleanup(func() { mac.poolPush, mac.poolPop = push, pop })
	var pushed, popped uint64
	mac.poolPush = func() uintptr { pushed = threadID(); return 1 }
	mac.poolPop = func(uintptr) { popped = threadID() }

	pooled(func() {
		taken, release := make(chan struct{}), make(chan struct{})
		// The new goroutine runs next on this thread once we block; it
		// keeps the thread to itself until we have resumed.
		go func() {
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()
			close(taken)
			<-release
		}()
		<-taken
		close(release)
	})
	if popped != pushed {
		t.Errorf("the pool was pushed on thread %d and popped on thread %d", pushed, popped)
	}
}
