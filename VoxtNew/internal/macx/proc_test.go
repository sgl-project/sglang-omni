package macx

import (
	"errors"
	"os"
	"runtime"
	"testing"
)

// TestMemory: on a Mac the test's own process has a footprint and
// resident pages (it allocated 64 MB, touched); off macOS it is
// ErrUnsupported.
func TestMemory(t *testing.T) {
	if runtime.GOOS != "darwin" {
		if _, err := Memory(os.Getpid()); !errors.Is(err, ErrUnsupported) {
			t.Fatalf("err %v", err)
		}
		return
	}
	ballast := make([]byte, 64<<20)
	for i := range ballast {
		ballast[i] = byte(i)
	}
	m, err := Memory(os.Getpid())
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("footprint %d MB, resident %d MB", m.Footprint>>20, m.Resident>>20)
	if m.Footprint < 64<<20 || m.Resident < 64<<20 || m.InUse() != max(m.Footprint, m.Resident) {
		t.Errorf("%+v", m)
	}
	runtime.KeepAlive(ballast)
	if _, err := Memory(0); err == nil {
		t.Error("pid 0: no error")
	}
}

func TestInUse(t *testing.T) {
	if got := (ProcMemory{Footprint: 110, Resident: 1150}).InUse(); got != 1150 {
		t.Errorf("GGUF: %d", got)
	}
	if got := (ProcMemory{Footprint: 3400, Resident: 900}).InUse(); got != 3400 {
		t.Errorf("MLX: %d", got)
	}
}
