package macx

import "errors"

// ProcMemory is a process's memory use, in bytes, as Activity Monitor
// counts it.
type ProcMemory struct {
	// Footprint is the physical footprint: the process's own memory,
	// Metal's buffers included (an MLX model's weights).
	Footprint int64
	// Resident is the resident memory, file-backed pages included: a
	// model mapped from its file (GGUF) counts here, not in Footprint.
	Resident int64
}

// InUse is the larger of the two: what the process holds in memory,
// whether its weights were copied (MLX) or mapped (GGUF).
func (m ProcMemory) InUse() int64 { return max(m.Footprint, m.Resident) }

// ErrUnsupported: not on macOS.
var ErrUnsupported = errors.New("macx: not supported on this system")

// Memory reads the memory use of the process with pid (one of the
// user's: proc_pid_rusage). It does not block on the process; call it
// off the main thread anyway, as any system call.
func Memory(pid int) (ProcMemory, error) { return procMemory(pid) }
