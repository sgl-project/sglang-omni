package hfhub

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/hehehai/actnowkit/models"
)

func TestCacheDir(t *testing.T) {
	fam := t.TempDir()
	t.Setenv("ACTNOW_FAMILY_DIR", fam)
	t.Setenv("HF_HOME", "/data/hf")
	t.Setenv("HF_HUB_CACHE", "/cache")
	// Since v0.8.0 the environment does not choose the library.
	if got := CacheDir(); got != filepath.Join(fam, "models", "hub") {
		t.Errorf("CacheDir %q", got)
	}
	if got := RepoDir("/cache", "mlx-community/Kokoro-82M-bf16"); got != filepath.Join("/cache", "models--mlx-community--Kokoro-82M-bf16") {
		t.Errorf("RepoDir %q", got)
	}
}

func TestOptionsDefaults(t *testing.T) {
	t.Setenv("HF_ENDPOINT", "")
	t.Setenv("ACTNOW_FAMILY_DIR", t.TempDir())
	c := New(Options{})
	o := c.Options()
	if o.UserAgent != "actnowkit-hfhub" || o.Official != "https://huggingface.co" || len(o.Endpoints) != 2 ||
		o.Parallel != 4 || o.ChunkSize != 16<<20 || o.ReserveBytes != 2e9 || o.CacheDir != filepath.Join(os.Getenv("ACTNOW_FAMILY_DIR"), "models", "hub") {
		t.Errorf("defaults %+v", o)
	}
	s := &models.Store{Hub: "/hub/of/store", BundleID: "dev.hehehai.dicta"}
	if o := New(Options{CacheDir: "/other", Store: s}).Options(); o.CacheDir != "/hub/of/store" || o.BundleID != "dev.hehehai.dicta" {
		t.Errorf("with a store %+v", o)
	}
}

func TestHumanSize(t *testing.T) {
	for in, want := range map[int64]string{0: "", 712781279: "≈ 713 MB", 2_340_000_000: "≈ 2.3 GB", 3e9: "≈ 3 GB"} {
		if got := HumanSize(in); got != want {
			t.Errorf("%d: %q, want %q", in, got, want)
		}
	}
}

func TestCode(t *testing.T) {
	for err, want := range map[error]string{
		nil:                               "",
		fmt.Errorf("x: %w", ErrGated):     CodeGated,
		fmt.Errorf("x: %w", ErrDiskSpace): CodeDiskSpace,
		fmt.Errorf("x: %w: %w", ErrOffline, errors.New("y")): CodeOffline,
		fmt.Errorf("x: %w", ErrChecksum):                     CodeChecksum,
		fmt.Errorf("x: %w", ErrNotFound):                     CodeNotFound,
		&models.BusyError{Lock: "x.lock"}:                    CodeBusy,
		fmt.Errorf("x: %w", context.Canceled):                CodeCancelled,
		errors.New("HTTP 500"):                               CodeDownloadFailed,
	} {
		if got := Code(err); got != want {
			t.Errorf("%v: %q, want %q", err, got, want)
		}
	}
}

func TestResolveToken(t *testing.T) {
	t.Setenv("HF_TOKEN", "")
	t.Setenv("HF_TOKEN_PATH", filepath.Join(t.TempDir(), "token"))
	shared := func() (string, error) { return "hf_shared\n", nil }
	own := func() (string, error) { return "hf_own", nil }
	none := func() (string, error) { return "", nil }
	broken := func() (string, error) { return "hf_x", errors.New("keychain locked") }
	check := func(name string, tok string, src TokenSource, wantTok string, wantSrc TokenSource) {
		t.Helper()
		if tok != wantTok || src != wantSrc {
			t.Errorf("%s: %q %q, want %q %q", name, tok, src, wantTok, wantSrc)
		}
	}
	tok, src := ResolveToken(nil, nil)
	check("nothing", tok, src, "", TokenNone)
	os.WriteFile(os.Getenv("HF_TOKEN_PATH"), []byte("hf_file\n"), 0o600)
	tok, src = ResolveToken(none, broken)
	check("file", tok, src, "hf_file", TokenCLI)
	tok, src = ResolveToken(none, own)
	check("own", tok, src, "hf_own", TokenApp)
	tok, src = ResolveToken(shared, own)
	check("shared", tok, src, "hf_shared", TokenShared)
	t.Setenv("HF_TOKEN", "hf_env")
	tok, src = ResolveToken(shared, own)
	check("env", tok, src, "hf_env", TokenEnv)

	t.Setenv("HF_TOKEN_PATH", "")
	t.Setenv("HF_HOME", "/data/hf")
	if TokenFile() != filepath.Join("/data/hf", "token") {
		t.Errorf("TokenFile %q", TokenFile())
	}
	if MaskToken("hf_abcdefghijkl") != "hf_…ijkl" || MaskToken("short") != "•••••" {
		t.Error(MaskToken("hf_abcdefghijkl"))
	}
}
