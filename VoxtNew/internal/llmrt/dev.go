package llmrt

// The developers' hook, until the catalogue and its downloads (M1b) fill
// local.Services: VOXTNEW_DEV_LLM names curated models and where they are
// on disk, and DevServices lists them as installed language models ahead
// of the others.
//
//	VOXTNEW_DEV_LLM=llm.balanced=~/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/<rev>
//	VOXTNEW_DEV_LLM=llm.balanced=/models/qwen,llm.translate=/models/hy-mt2
//
// Any MLX text model's snapshot proves the pipeline under llm.balanced or
// llm.fast; llm.translate wants Hy-MT2's GGUF (its directory or the file).

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/local"
)

// DevEnv is the hook's environment variable.
const DevEnv = "VOXTNEW_DEV_LLM"

// DevModel is a model the hook installs.
type DevModel struct {
	ID, Dir string
}

// ParseDev reads the hook's value: id=dir pairs, comma-separated; "~/" is
// the home directory. Unknown ids fail.
func ParseDev(spec string) ([]DevModel, error) {
	var out []DevModel
	for _, part := range strings.Split(spec, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		id, dir, ok := strings.Cut(part, "=")
		if !ok || dir == "" {
			return nil, fmt.Errorf("%s: %q is not id=dir", DevEnv, part)
		}
		if _, known := Lookup(id); !known {
			return nil, fmt.Errorf("%s: unknown model %q (llm.balanced, llm.fast, llm.translate)", DevEnv, id)
		}
		if rest, ok := strings.CutPrefix(dir, "~/"); ok {
			home, _ := os.UserHomeDir()
			dir = filepath.Join(home, rest)
		}
		out = append(out, DevModel{ID: id, Dir: dir})
	}
	return out, nil
}

// DevServices is base with the hook's models installed as language
// models, ahead of base's own; speech models are base's.
func DevServices(base local.Services, rt *Runtime, models []DevModel) local.Services {
	return &devServices{base: base, rt: rt, models: models}
}

type devServices struct {
	base   local.Services
	rt     *Runtime
	models []DevModel
}

func (d *devServices) list() []local.Model {
	out := make([]local.Model, 0, len(d.models))
	for _, m := range d.models {
		lm := local.Model{Kind: local.LLM, ID: m.ID, Name: "Dev: " + m.ID}
		if p, _ := Lookup(m.ID); p.TranslateOnly() {
			lm.Features = []string{"translate"}
		}
		out = append(out, lm)
	}
	return out
}

func (d *devServices) Installed(kind local.Kind) []local.Model {
	if kind != local.LLM {
		return d.base.Installed(kind)
	}
	return append(d.list(), d.base.Installed(kind)...)
}

func (d *devServices) ASR(ctx context.Context, id string) (asr.Provider, local.Model, error) {
	return d.base.ASR(ctx, id)
}

func (d *devServices) LLM(ctx context.Context, id string) (local.Chat, local.Model, error) {
	m, ok := local.Choose(d.Installed(local.LLM), id)
	for _, dm := range d.models {
		if ok && dm.ID == m.ID {
			chat, err := Open(ctx, d.rt, dm.ID, dm.Dir)
			return chat, m, err
		}
	}
	return d.base.LLM(ctx, id)
}
