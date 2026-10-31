package llmrt

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llm"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llmrt/wire"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/local"
)

// warmTimeout bounds the load Open starts.
const warmTimeout = 2 * time.Minute

// Open is curated model id, installed in dir, as a session uses it: an
// llm.Client (two attempts: a crashed process is started again; JSON
// repair; thinking stripped) over a Backend on rt. dir is the model's
// directory: an MLX snapshot, or a directory holding the GGUF file (or
// the file itself). It fails with an error wrapping local.ErrNotInstalled
// when id is unknown or its files are not there.
//
// Open starts loading the model in the background (a session opens its
// model while it records), so that the first chat finds it warm; ctx
// bounds nothing but the check.
func Open(ctx context.Context, rt *Runtime, id, dir string) (local.Chat, error) {
	b, err := NewModel(rt, id, dir)
	if err != nil {
		return nil, err
	}
	if ctx.Err() == nil {
		go func() {
			wctx, cancel := context.WithTimeout(context.Background(), warmTimeout)
			defer cancel()
			if err := b.Warm(wctx); err != nil {
				rt.log(core.SeverityWarning, fmt.Sprintf("warming %s up: %v", id, err))
			}
		}()
	}
	return &llm.Client{Backend: b, Retry: llm.RetryPolicy{MaxAttempts: 2}}, nil
}

// NewModel is the Backend of curated model id installed in dir (see
// Open), without a warm-up.
func NewModel(rt *Runtime, id, dir string) (*Backend, error) {
	p, ok := Lookup(id)
	if !ok {
		return nil, fmt.Errorf("%w: unknown model", local.NotInstalledError(local.LLM, id))
	}
	path, err := modelPath(p, dir)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", local.NotInstalledError(local.LLM, id), err)
	}
	return NewBackend(rt, p, path), nil
}

// modelPath is where p's files are under dir: the directory with
// config.json (MLX), or the .gguf file.
func modelPath(p Profile, dir string) (string, error) {
	if p.Format == wire.FormatGGUF {
		path := dir
		if !strings.HasSuffix(strings.ToLower(dir), ".gguf") {
			path = filepath.Join(dir, p.File)
		}
		if st, err := os.Stat(path); err != nil || st.IsDir() {
			return "", fmt.Errorf("no %s in %s", p.File, dir)
		}
		return path, nil
	}
	if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
		return "", fmt.Errorf("no config.json in %s", dir)
	}
	return dir, nil
}
