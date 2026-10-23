package update

import (
	"bytes"
	"compress/flate"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json/v2"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
)

// Delta updates, as Sparkle's: a delta turns the app of an earlier version
// into the new one, and holds only what changed. Its index lists the tree
// of the new app: directories, links, and files with their size and
// SHA-256, each a copy of a file of the old app, a patch of one
// (bsdiff.go), or new. A delta is
//
//	deltaMagic, the length of the index (uvarint), the index (JSON, DEFLATE),
//	the data of the files, in the order of the index: a patch, or the
//	file compressed with DEFLATE,
//
// and is signed like an archive. The data is read from the delta file in
// place, and applying it checks every file it makes, so that a delta
// applied to another app than the one it was made from fails: the updater
// then downloads the archive.

const deltaMagic = "mygo delta 1\n"

type deltaIndex struct {
	// From and Version are the versions of the app the delta updates and
	// makes.
	From    string       `json:"from"`
	Version string       `json:"version"`
	Entries []deltaEntry `json:"entries"`
}

// deltaEntry is a directory, a link or a file of the new app. Paths are
// relative to the app: its bundle on macOS, its directory elsewhere.
type deltaEntry struct {
	Path string `json:"path"`
	Dir  bool   `json:"dir,omitzero"`
	// Link is the target of a symbolic link.
	Link string `json:"link,omitzero"`
	// Mode holds the permissions of a file.
	Mode   uint32 `json:"mode,omitzero"`
	Size   int64  `json:"size,omitzero"`
	SHA256 string `json:"sha256,omitzero"`
	// From is the file of the old app that the file copies, or patches
	// with its Data. Without From, Data is the file, compressed.
	From string `json:"from,omitzero"`
	Data int64  `json:"data,omitzero"`
}

func (e *deltaEntry) isFile() bool { return !e.Dir && e.Link == "" }

// maxDiff is the size of the largest files that deltas patch: making the
// patch takes about ten times as much memory. Larger files that changed
// are included whole.
const maxDiff = 256 << 20

// stapledTicket is where stapling writes the notarization ticket of an app
// bundle. The archive's app has it, but the app installed from the disk
// image does not (dmg.go staples the app after copying it into the image),
// so deltas include it whole, never as a copy or patch of the old app's.
const stapledTicket = "Contents/CodeResources"

// WriteDelta writes the delta that turns the app in oldDir, of version from,
// into the app in newDir, of version to: the entries of newDir, or all of
// it when entries is nil.
func WriteDelta(w io.Writer, from, to, oldDir, newDir string, entries []string) error {
	// The files of the old app, and the first of each content.
	type oldFile struct {
		sum  string // SHA-256
		size int64
	}
	old := map[string]oldFile{}
	bySum := map[string]string{}
	err := filepath.WalkDir(oldDir, func(p string, d fs.DirEntry, err error) error {
		if err != nil || !d.Type().IsRegular() {
			return err
		}
		if relPath(oldDir, p) == stapledTicket {
			return nil
		}
		f, err := os.Open(p)
		if err != nil {
			return err
		}
		defer f.Close()
		h := sha256.New()
		size, err := io.Copy(h, f)
		if err != nil {
			return err
		}
		rel, sum := relPath(oldDir, p), hex.EncodeToString(h.Sum(nil))
		old[rel] = oldFile{sum, size}
		if _, ok := bySum[sum]; !ok {
			bySum[sum] = rel
		}
		return nil
	})
	if err != nil {
		return err
	}
	if entries == nil {
		all, err := os.ReadDir(newDir)
		if err != nil {
			return err
		}
		for _, e := range all {
			entries = append(entries, e.Name())
		}
	}

	index := deltaIndex{From: from, Version: to}
	var data [][]byte
	for _, top := range entries {
		err := filepath.WalkDir(filepath.Join(newDir, top), func(p string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			e := deltaEntry{Path: relPath(newDir, p)}
			switch {
			case d.IsDir():
				e.Dir = true
			case d.Type()&fs.ModeSymlink != 0:
				if e.Link, err = os.Readlink(p); err != nil {
					return err
				}
				e.Link = filepath.ToSlash(e.Link)
			case d.Type().IsRegular():
				info, err := d.Info()
				if err != nil {
					return err
				}
				b, err := os.ReadFile(p)
				if err != nil {
					return err
				}
				sum := sha256.Sum256(b)
				e.Mode, e.Size, e.SHA256 = uint32(info.Mode().Perm()), int64(len(b)), hex.EncodeToString(sum[:])
				o := old[e.Path]
				if o.sum == e.SHA256 {
					e.From = e.Path
				} else if src, ok := bySum[e.SHA256]; ok {
					e.From = src // moved, or copied
				} else {
					blob := deflate(b)
					if o.size > 0 && o.size <= maxDiff && len(b) <= maxDiff {
						prev, err := os.ReadFile(filepath.Join(oldDir, filepath.FromSlash(e.Path)))
						if err != nil {
							return err
						}
						if patch := Diff(prev, b); len(patch) < len(blob) {
							e.From, blob = e.Path, patch
						}
					}
					e.Data = int64(len(blob))
					data = append(data, blob)
				}
			default:
				return fmt.Errorf("%s is not a file, directory or link", p)
			}
			index.Entries = append(index.Entries, e)
			return nil
		})
		if err != nil {
			return err
		}
	}

	js, err := json.Marshal(index)
	if err != nil {
		return err
	}
	js = deflate(js)
	head := binary.AppendUvarint([]byte(deltaMagic), uint64(len(js)))
	if _, err := w.Write(append(head, js...)); err != nil {
		return err
	}
	for _, b := range data {
		if _, err := w.Write(b); err != nil {
			return err
		}
	}
	return nil
}

func relPath(dir, p string) string {
	rel, _ := filepath.Rel(dir, p)
	return filepath.ToSlash(rel)
}

// maxDeltaIndex limits the index of a delta that an app reads.
const maxDeltaIndex = 16 << 20

var errDamagedDelta = errors.New("damaged delta update")

// ApplyDelta makes in newDir, which must not exist, the app that the delta
// of size bytes read from r makes of the app in oldDir. The delta must
// update version from to version to, and every file it makes must have
// the size and SHA-256 its index gives.
func ApplyDelta(r io.ReaderAt, size int64, from, to, oldDir, newDir string) error {
	var head [len(deltaMagic) + binary.MaxVarintLen64]byte
	n, _ := r.ReadAt(head[:min(size, int64(len(head)))], 0)
	if !bytes.HasPrefix(head[:n], []byte(deltaMagic)) {
		return errors.New("not a delta update")
	}
	var indexLen uint64
	k, ok := uvarints(head[len(deltaMagic):n], &indexLen)
	start := int64(len(deltaMagic)) + k
	if !ok || indexLen > uint64(size-start) {
		return errDamagedDelta
	}
	js, err := io.ReadAll(io.LimitReader(flate.NewReader(io.NewSectionReader(r, start, int64(indexLen))), maxDeltaIndex+1))
	if err != nil || len(js) > maxDeltaIndex {
		return errDamagedDelta
	}
	var index deltaIndex
	if err := json.Unmarshal(js, &index); err != nil {
		return errDamagedDelta
	}
	if index.From != from || index.Version != to {
		return fmt.Errorf("the delta updates %s to %s, not %s to %s", index.From, index.Version, from, to)
	}

	// Check the index before making anything.
	offset := start + int64(indexLen)
	seen := map[string]bool{}
	for _, e := range index.Entries {
		if !fs.ValidPath(e.Path) || e.Path == "." || seen[e.Path] || e.Size < 0 || e.Data < 0 || e.Data > size-offset ||
			e.Dir && e.Link != "" || !e.isFile() && (e.Size != 0 || e.Data != 0 || e.From != "") ||
			e.From != "" && !fs.ValidPath(e.From) || e.isFile() && e.From == "" && e.Data == 0 {
			return errDamagedDelta
		}
		if e.Link != "" {
			if target := path.Join(path.Dir(e.Path), e.Link); path.IsAbs(e.Link) || !fs.ValidPath(target) {
				return fmt.Errorf("delta update link %q points outside the app", e.Path)
			}
		}
		seen[e.Path] = true
		offset += e.Data
	}
	if offset != size {
		return errDamagedDelta
	}

	if err := os.Mkdir(newDir, 0o755); err != nil {
		return err
	}
	oldRoot, err := os.OpenRoot(oldDir)
	if err != nil {
		return err
	}
	defer oldRoot.Close()
	newRoot, err := os.OpenRoot(newDir)
	if err != nil {
		return err
	}
	defer newRoot.Close()
	offset = start + int64(indexLen)
	for _, e := range index.Entries {
		name := filepath.FromSlash(e.Path)
		if dir := filepath.Dir(name); dir != "." {
			if err := newRoot.MkdirAll(dir, 0o755); err != nil {
				return err
			}
		}
		switch {
		case e.Dir:
			err = newRoot.MkdirAll(name, 0o755)
		case e.Link != "":
			err = newRoot.Symlink(filepath.FromSlash(e.Link), name)
		default:
			err = applyFile(e, io.NewSectionReader(r, offset, e.Data), oldRoot, newRoot)
			offset += e.Data
		}
		if err != nil {
			return err
		}
	}
	return nil
}

// applyFile makes the file of e from data, its data in the delta.
func applyFile(e deltaEntry, data *io.SectionReader, oldRoot, newRoot *os.Root) error {
	f, err := newRoot.OpenFile(filepath.FromSlash(e.Path), os.O_CREATE|os.O_EXCL|os.O_WRONLY, fs.FileMode(e.Mode).Perm()|0o200)
	if err != nil {
		return err
	}
	defer f.Close()
	h := sha256.New()
	out := &limitWriter{w: io.MultiWriter(f, h), n: e.Size}
	if e.From == "" {
		_, err = io.Copy(out, flate.NewReader(data))
	} else {
		var src *os.File
		if src, err = oldRoot.Open(filepath.FromSlash(e.From)); err != nil {
			return err
		}
		defer src.Close()
		var info fs.FileInfo
		switch info, err = src.Stat(); {
		case err != nil:
		case !info.Mode().IsRegular():
			err = fmt.Errorf("%s is not a file", e.From)
		case e.Data == 0:
			_, err = io.Copy(out, src)
		default:
			err = Patch(out, e.Size, src, info.Size(), data, e.Data)
		}
	}
	if err == nil {
		err = f.Close()
	}
	if err == errDamagedPatch || err == errTooLong || err == nil && (out.n != 0 || hex.EncodeToString(h.Sum(nil)) != e.SHA256) {
		// Most likely, the app is not the one the delta was made from.
		return fmt.Errorf("the delta update made %s wrong", e.Path)
	}
	return err
}

var errTooLong = errors.New("too long")

// limitWriter fails writes past n bytes.
type limitWriter struct {
	w io.Writer
	n int64
}

func (l *limitWriter) Write(b []byte) (int, error) {
	if int64(len(b)) > l.n {
		return 0, errTooLong
	}
	l.n -= int64(len(b))
	return l.w.Write(b)
}
