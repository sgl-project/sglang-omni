package main

// The previous Voxt's speech models (legacy_import.go's import, and once
// each at the start): the downloads in its model-storage whose files are
// the catalogue's pinned ones are hard-linked into the model library, so
// nothing is downloaded again. Voxt 1.x kept a Whisper model's tokenizer
// (OpenAI's files, an Aux of the catalogue) in the model's folder.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"log"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
)

// legacyReusable says whether a catalogue model may come over from the
// previous Voxt: not a borrowed one (another app's), not a retired one
// (Voxt no longer runs it, and Manager.Import leaves those where they
// are), and one with files to match.
func legacyReusable(mod models.Model) bool {
	return !mod.Borrowed && !mod.Retired && len(mod.Files) > 0
}

// legacyModelRoots are the folders the previous Voxt kept its models in:
// model-storage in its Application Support folder (voxtDir), and the
// folder the user chose instead in its Settings → Model, which its
// preferences keep as modelStorageRootPath (legacyPreference, readable
// when the import reads the whole Data/Library); without it only
// model-storage is looked in.
func legacyModelRoots(voxtDir string) []string {
	roots := []string{filepath.Join(voxtDir, "model-storage")}
	if p := legacyPreference(voxtDir, "modelStorageRootPath"); filepath.IsAbs(p) && p != roots[0] {
		roots = append(roots, p)
	}
	return roots
}

// legacyModelDir is where the previous Voxt kept a speech model under one
// of its roots: mlx-audio/<owner>_<name> (MLXModelStorageSupport.cacheDirectory).
func legacyModelDir(root string, mod models.Model) string {
	return filepath.Join(root, "mlx-audio", strings.ReplaceAll(mod.Repo, "/", "_"))
}

// reuseLegacyModels brings the models the previous Voxt downloaded into the
// library when its files are the catalogue's: every file of the pinned
// commit at its size, the large ones also at their SHA-256. The previous
// Voxt kept each model in <root>/mlx-audio/<owner>_<name> (legacyModelRoots,
// legacyModelDir), with a Whisper model's tokenizer beside its weights;
// the library wants $HUB/models--<owner>--<name>/snapshots/<commit>, the
// tokenizer in its own repo's. A model whose tokenizer is not there stays.
// Files are hard linked (copied across volumes), so nothing is downloaded
// and no space is used twice. only limits it to those catalogue ids (none:
// every one). It returns the models brought, tokenizers too, and their
// bytes.
func reuseLegacyModels(ctx context.Context, m *models.Manager, voxtDir string, only ...string) (ids []string, bytes int64) {
	roots := legacyModelRoots(voxtDir)
	for _, mod := range m.Catalog() {
		if !legacyReusable(mod) || mod.Kind == models.Aux || m.Installed(mod.ID) || ctx.Err() != nil ||
			len(only) > 0 && !slices.Contains(only, mod.ID) {
			continue
		}
		for _, root := range roots {
			src := legacyModelDir(root, mod)
			if !legacyModelMatches(src, mod) {
				continue
			}
			bring := []models.Model{mod}
			for _, need := range mod.Needs {
				if aux, _ := m.Lookup(need); aux.Kind == models.Aux && !m.Installed(need) && !slices.Contains(ids, need) {
					bring = append([]models.Model{aux}, bring...)
				}
			}
			if !slices.ContainsFunc(bring, func(x models.Model) bool { return !legacyModelMatches(src, x) }) {
				for _, x := range bring {
					n, err := linkModelFiles(src, models.SnapshotDir(m.Hub(), x, x.Revision), x)
					if err != nil {
						log.Printf("legacy import: model %s: %v", x.ID, err)
						break
					}
					ids, bytes = append(ids, x.ID), bytes+n
				}
			}
			break
		}
	}
	if len(ids) > 0 {
		m.Refresh()
		if err := m.Use(ctx, ids...); err != nil {
			log.Printf("legacy import: claiming %v: %v", ids, err)
		}
	}
	return ids, bytes
}

// legacyModelsChecked is the document of the catalogue models the start
// looked for in the previous Voxt's folder (legacyModelsAtStart).
const legacyModelsChecked = "legacy-models-checked"

// legacyModelsAtStart brings over, at the start, the previous Voxt's
// downloads of the speech models this version added to the catalogue (who
// imported with an older Voxt has them in the previous Voxt only), when
// its folder reads without asking (legacyReadable). Each model is looked
// for once, as matching its files hashes them; the import looks again.
// Off the main thread.
func (a *app) legacyModelsAtStart() {
	lib := legacyLibrary()
	if lib == "" || a.models == nil {
		return
	}
	docs, mods := a.lib.store.Documents(), a.models
	a.async(func() {
		ctx := context.Background()
		raw, _, err := docs.Get(ctx, legacyModelsChecked)
		if err != nil {
			log.Printf("legacy models: %v", err)
			return
		}
		var checked, look []string
		json.Unmarshal(raw, &checked) // none yet: every model
		for _, m := range mods.Catalog() {
			if legacyReusable(m) && m.Kind != models.Aux && !slices.Contains(checked, m.ID) {
				look = append(look, m.ID)
			}
		}
		path := legacyDatabase(lib)
		if len(look) == 0 || !legacyReadable(lib) || path == "" {
			return
		}
		ids, bytes := reuseLegacyModels(ctx, mods, filepath.Dir(path), look...)
		if b, _ := json.Marshal(append(checked, look...)); docs.Put(ctx, legacyModelsChecked, b) != nil {
			log.Printf("legacy models: could not record the models looked for")
		}
		if len(ids) > 0 {
			log.Printf("legacy models: brought %v (%d bytes) from the previous Voxt", ids, bytes)
			a.post(a.modelsChanged)
		}
	})
}

// legacyModelMatches reports whether dir holds every file of mod's pinned
// commit, at its size and, for the large (LFS) ones, its SHA-256.
func legacyModelMatches(dir string, mod models.Model) bool {
	for _, f := range mod.Files {
		p := filepath.Join(dir, filepath.FromSlash(f.Path))
		st, err := os.Stat(p)
		if err != nil || st.IsDir() || st.Size() != f.Size {
			return false
		}
		if f.LFS && f.SHA256 != "" {
			sum, err := fileSHA256(p)
			if err != nil || sum != f.SHA256 {
				return false
			}
		}
	}
	return true
}

func fileSHA256(p string) (string, error) {
	f, err := os.Open(p)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// linkModelFiles puts mod's files from src into the snapshot folder dst:
// a hard link, or a copy when the two are on different volumes.
func linkModelFiles(src, dst string, mod models.Model) (bytes int64, err error) {
	for _, f := range mod.Files {
		from, to := filepath.Join(src, filepath.FromSlash(f.Path)), filepath.Join(dst, filepath.FromSlash(f.Path))
		if err := os.MkdirAll(filepath.Dir(to), 0o755); err != nil {
			return 0, err
		}
		os.Remove(to)
		if err := os.Link(from, to); err != nil {
			if err := copyFile(from, to); err != nil {
				return 0, err
			}
		}
		bytes += f.Size
	}
	return bytes, nil
}

func copyFile(from, to string) error {
	in, err := os.Open(from)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(to)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}
