package omni

import (
	"encoding/json"
	"sync/atomic"
	"time"
)

// requestStage is one content-free timing span in an offline Final request.
type requestStage struct {
	Event       string  `json:"event"`
	RequestID   uint64  `json:"request_id"`
	Kind        string  `json:"kind"`
	Stage       string  `json:"stage"`
	DurationMS  float64 `json:"duration_ms"`
	InputBytes  int     `json:"input_bytes,omitempty"`
	OutputBytes int     `json:"output_bytes,omitempty"`
}

var offlineRequestSequence atomic.Uint64

func nextOfflineRequestID() uint64 { return offlineRequestSequence.Add(1) }

func logRequestStage(rt *Runtime, id uint64, kind, stage string, started time.Time, input, output int) {
	if id == 0 {
		return
	}
	payload, err := json.Marshal(requestStage{
		Event: "omni_request_stage", RequestID: id, Kind: kind, Stage: stage,
		DurationMS: float64(time.Since(started).Microseconds()) / 1000,
		InputBytes: input, OutputBytes: output,
	})
	if err == nil {
		rt.logf("info", "%s", payload)
	}
}
