package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/hehehai/actnowkit/conformance"

	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
)

// TestReuseLegacyWhisper: a Whisper model comes over with the tokenizer
// Voxt 1.x kept beside its weights, each into its own repo's snapshot; a
// model whose folder lacks the tokenizer stays where it is until the
// tokenizer is installed.
func TestReuseLegacyWhisper(t *testing.T) {
	voxtDir := filepath.Join(t.TempDir(), "Application Support", "Voxt")
	file := func(dir, path, content string, lfs bool) models.File {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, path), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256([]byte(content))
		return models.File{Path: path, Size: int64(len(content)), SHA256: hex.EncodeToString(sum[:]), LFS: lfs}
	}
	rev := "0123456789abcdef0123456789abcdef01234567"
	tok := models.Model{ID: "aux.tok", Kind: models.Aux, Runtime: models.Omni, Repo: "openai/whisper-test", Revision: rev, Family: "whisper"}
	whole := models.Model{ID: "asr.whole", Kind: models.ASR, Runtime: models.Omni, Repo: "owner/whole", Revision: rev, Family: "whisper", Needs: []string{tok.ID}}
	lacking := models.Model{ID: "asr.lacking", Kind: models.ASR, Runtime: models.Omni, Repo: "owner/lacking", Revision: rev, Family: "whisper", Needs: []string{tok.ID}}
	root := filepath.Join(voxtDir, "model-storage")
	wholeDir, lackingDir := legacyModelDir(root, whole), legacyModelDir(root, lacking)
	whole.Files = []models.File{file(wholeDir, "config.json", `{"n":1}`, false), file(wholeDir, "weights.npz", "weights", true)}
	tok.Files = []models.File{file(wholeDir, "vocab.json", `{"a":0}`, false)}
	lacking.Files = []models.File{file(lackingDir, "config.json", `{"n":2}`, false), file(lackingDir, "weights.npz", "other weights", true)}

	fx := conformance.New(t)
	m := models.Open(models.Options{Store: fx.Peer("dev.hehehai.voxtnew"), Catalog: []models.Model{lacking, whole, tok}})
	t.Cleanup(m.Close)
	ids, _ := reuseLegacyModels(context.Background(), m, voxtDir)
	if !slices.Equal(ids, []string{tok.ID, whole.ID}) {
		t.Fatalf("brought %v", ids)
	}
	if !m.Installed(whole.ID) || !m.Installed(tok.ID) || m.Installed(lacking.ID) {
		t.Errorf("installed: whole %v, tokenizer %v, lacking %v", m.Installed(whole.ID), m.Installed(tok.ID), m.Installed(lacking.ID))
	}
	if _, err := os.Stat(filepath.Join(m.Dir(tok.ID), "vocab.json")); err != nil {
		t.Errorf("the tokenizer is not in its snapshot: %v", err)
	}
	// Once the tokenizer is installed, a model that has its weights only comes over.
	if ids, _ := reuseLegacyModels(context.Background(), m, voxtDir, lacking.ID); !slices.Equal(ids, []string{lacking.ID}) {
		t.Errorf("with the tokenizer installed: %v", ids)
	}
}

// The previous Voxt's choice of a local speech model, as it mapped the
// repos it had retired.
func TestLegacyLocalSpeech(t *testing.T) {
	for in, want := range map[string]string{
		"mlx:mlx-community/whisper-large-v3-turbo":        models.ASRWhisperTurbo,
		"mlx:mlx-community/Qwen3-ASR-1.7B-8bit":           models.ASRAccurate,
		"mlx:OpenMOSS-Team/MOSS-Transcribe-Diarize":       models.ASRMoss,
		"mlx:mlx-community/parakeet-tdt-0.6b-v2":          models.ASRParakeet,
		"mlx:mlx-community/whisper-base-mlx":              models.ASRWhisperSmall,
		"mlx:mlx-community/Qwen3-ASR-1.7B-bf16":           models.ASRQwen6bit,
		"mlx:mlx-community/Voxtral-Mini-4B-Realtime-2602": models.ASRFast,
		"whisper:openai_whisper-small":                    models.ASRWhisperSmall,
		"whisper:medium":                                  models.ASRWhisperTurbo,
		"whisper:large-v3-v20240930":                      models.ASRWhisperLarge,
		"sherpa:paraformer":                               models.ASRFast,
		"remote-asr:doubaoASR":                            "",
		"dictation":                                       "",
		"":                                                "",
	} {
		if got := legacyLocalSpeech(in); got != want {
			t.Errorf("%q: %q, want %q", in, got, want)
		}
	}
}
