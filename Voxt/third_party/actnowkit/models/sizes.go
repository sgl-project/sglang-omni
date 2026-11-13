// Blob sizes: the "installed" check compares each linked file's size with
// the size its blob had when it was verified (spec §7.5, v1.0.9), so that
// a truncated weights file is not taken for a model.

package models

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// KindBlobs is the schema kind of a repo's blob sizes file.
const KindBlobs = "models.blobs"

// SizesDirName is the family's folder inside a repo folder
// (models--owner--name/.actnow). huggingface_hub reads only blobs/,
// snapshots/ and refs/ of a repo folder, so tools ignore it.
const SizesDirName = ".actnow"

// SizesFile is $HUB/models--owner--name/.actnow/blobs.json: the size of
// each blob of the repo (blob name → bytes) as verified by whoever wrote
// it. A blob's name is its hash, so its size never changes: entries are
// only added.
func SizesFile(hub, repo string) string {
	return filepath.Join(RepoDir(hub, repo), SizesDirName, "blobs.json")
}

type blobSizes struct {
	Schema string           `json:"schema"`
	Sizes  map[string]int64 `json:"sizes"`
}

// ReadBlobSizes reads a repo's blob sizes; nil when there are none (a
// download by an older app or a Python tool) or the file is unreadable.
func ReadBlobSizes(hub, repo string) map[string]int64 {
	data, err := os.ReadFile(SizesFile(hub, repo))
	if err != nil {
		return nil
	}
	var b blobSizes
	if json.Unmarshal(data, &b) != nil {
		return nil
	}
	return b.Sizes
}

// RecordBlobSizes adds blob sizes to a repo's file, for a downloader that
// verified them (hfhub does; an app's own downloader should). The caller
// holds the repo lock exclusively (finishing a snapshot), so writers do
// not race. Blob names that are not hashes are skipped.
func RecordBlobSizes(hub, repo string, sizes map[string]int64) error {
	if !ValidRepo(repo) {
		return errBadRepo
	}
	if len(sizes) == 0 {
		return nil
	}
	cur := ReadBlobSizes(hub, repo)
	if cur == nil {
		cur = map[string]int64{}
	}
	changed := false
	for b, n := range sizes {
		if !validBlob(b) || n < 0 {
			continue
		}
		if old, ok := cur[b]; !ok || old != n {
			cur[b], changed = n, true
		}
	}
	if !changed {
		return nil
	}
	data, err := json.MarshalIndent(blobSizes{Schema: "actnow." + KindBlobs + "/1", Sizes: cur}, "", "  ")
	if err != nil {
		return err
	}
	path := SizesFile(hub, repo)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, append(data, '\n'), 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}
