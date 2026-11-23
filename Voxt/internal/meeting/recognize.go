package meeting

import "github.com/sgl-project/sglang-omni/Voxt/internal/asr"

// LivePath is how one meeting track transcribes while recording (design C1 §4.7).
type LivePath int

const (
	// LivePathStream keeps one ASR stream open and commits via Committer.
	LivePathStream LivePath = iota
	// LivePathFile cuts utterances and transcribes each with File mode.
	LivePathFile
)

// PickLivePath chooses stream vs file+cutter from provider caps (local omni prefers stream).
func PickLivePath(c asr.Caps) LivePath {
	if c.Stream {
		return LivePathStream
	}
	return LivePathFile
}
