//go:build !darwin

package macx

func procMemory(int) (ProcMemory, error) { return ProcMemory{}, ErrUnsupported }

func procCPUTime(int) (ProcCPU, error) { return ProcCPU{}, ErrUnsupported }

func readSystemMemory() (SystemMemory, error) { return SystemMemory{}, ErrUnsupported }
