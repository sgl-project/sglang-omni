package omni

import "github.com/sgl-project/sglang-omni/VoxtNew/internal/macx"

// readSystemMemory reads the Mac's memory for the log; tests replace it.
var readSystemMemory = macx.ReadSystemMemory

// memoryNote is the Mac's memory as the log says it: a slow or stalled
// speech model is most often one whose weights were swapped out.
func memoryNote() string {
	m, err := readSystemMemory()
	if err != nil {
		return "memory unknown: " + err.Error()
	}
	return m.String()
}
