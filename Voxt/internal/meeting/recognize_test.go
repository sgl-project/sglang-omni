package meeting

import (
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

func TestPickLivePath(t *testing.T) {
	if PickLivePath(asr.Caps{Stream: true}) != LivePathStream {
		t.Fatal("stream-only should use stream")
	}
	if PickLivePath(asr.Caps{File: true}) != LivePathFile {
		t.Fatal("file-only should use cutter")
	}
	if PickLivePath(asr.Caps{Stream: true, File: true}) != LivePathStream {
		t.Fatal("both caps should prefer stream")
	}
}
