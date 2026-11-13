// Files in the cache: verifying blobs, writing atomically, linking snapshots.

package hfhub

import (
	"context"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"hash"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

// checkBlob hashes a file: sha256 for LFS files, the git blob id for others.
func checkBlob(ctx context.Context, path string, f remoteFile) (bool, error) {
	fh, err := os.Open(path)
	if err != nil {
		return false, err
	}
	defer fh.Close()
	st, err := fh.Stat()
	if err != nil {
		return false, err
	}
	if st.Size() != f.Size {
		return false, nil
	}
	var h hash.Hash
	want := f.blob
	if f.lfs || f.SHA256 != "" {
		h = sha256.New()
		want = f.SHA256
	} else {
		h = sha1.New()
		fmt.Fprintf(h, "blob %d\x00", f.Size)
	}
	if want == "" {
		return true, nil
	}
	if _, err := io.Copy(h, ctxReader{ctx, fh}); err != nil {
		return false, err
	}
	return hex.EncodeToString(h.Sum(nil)) == strings.ToLower(want), nil
}

type ctxReader struct {
	ctx context.Context
	r   io.Reader
}

func (r ctxReader) Read(p []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	return r.r.Read(p)
}

func writeParts(path string, p *partsFile) error {
	data, _ := json.Marshal(p)
	return writeAtomic(path, data)
}

// writeAtomic writes data to path through a temporary file and a rename
// (not synced: parts files are rewritten often, and a lost one only
// costs a re-download).
func writeAtomic(path string, data []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), "."+filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		os.Remove(tmp.Name())
		return err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmp.Name())
		return err
	}
	for attempt := 0; ; attempt++ {
		err := os.Rename(tmp.Name(), path)
		if err == nil {
			return nil
		}
		if runtime.GOOS != "windows" || attempt == 99 {
			os.Remove(tmp.Name())
			return err
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// linkSnapshot makes snapshots/<commit>/<path> → ../../blobs/<blob> for
// every file. A new snapshot is built beside its place and renamed in, so
// it is never half there; one that exists (made by another tool) gets the
// links it lacks.
func linkSnapshot(repoDir, commit string, files []remoteFile) error {
	snaps := filepath.Join(repoDir, "snapshots")
	final := filepath.Join(snaps, commit)
	_, err := os.Stat(final)
	exists := err == nil
	dir := final
	if !exists {
		dir = final + ".incomplete"
		os.RemoveAll(dir)
	}
	for _, f := range files {
		link := filepath.Join(dir, filepath.FromSlash(f.Path))
		if err := os.MkdirAll(filepath.Dir(link), 0o755); err != nil {
			return err
		}
		depth := strings.Count(f.Path, "/") + 2 // up from the file's directory to repoDir
		target := strings.Repeat("../", depth) + "blobs/" + f.blob
		if cur, err := os.Readlink(link); err == nil && cur == target {
			continue
		}
		os.Remove(link)
		if err := linkOrCopy(target, link, filepath.Join(repoDir, "blobs", f.blob)); err != nil {
			return err
		}
	}
	if !exists {
		return os.Rename(dir, final)
	}
	return nil
}

func linkOrCopy(target, link, blob string) error {
	if err := os.Symlink(target, link); err == nil {
		return nil
	}
	if err := os.Link(blob, link); err == nil {
		return nil
	}
	in, err := os.Open(blob)
	if err != nil {
		return err
	}
	defer in.Close()
	info, err := in.Stat()
	if err != nil {
		return err
	}
	out, err := os.OpenFile(link, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, info.Mode().Perm())
	if err != nil {
		return err
	}
	_, err = io.Copy(out, in)
	if closeErr := out.Close(); err == nil {
		err = closeErr
	}
	return err
}
