package family

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

// WriteFileAtomic replaces path with data (spec §5.2): a temporary file
// ".<name>.tmp-<pid>-<random>" in the same directory, written and synced,
// then renamed over path, then the directory synced. Readers see the old
// file or the new one, never a mix. The directory is created (0700) when
// missing.
func WriteFileAtomic(path string, data []byte, perm os.FileMode) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	tmp, err := writeTemp(path, data, perm)
	if err != nil {
		return err
	}
	if err := renameFile(tmp, path); err != nil {
		os.Remove(tmp)
		return err
	}
	syncDir(filepath.Dir(path))
	return nil
}

func renameFile(oldPath, newPath string) error {
	for attempt := 0; ; attempt++ {
		err := os.Rename(oldPath, newPath)
		if err == nil || runtime.GOOS != "windows" || attempt == 99 {
			return err
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// writeTemp writes data to a new temporary file beside path and returns
// its name.
func writeTemp(path string, data []byte, perm os.FileMode) (string, error) {
	var rnd [6]byte
	_, _ = rand.Read(rnd[:])
	name := filepath.Join(filepath.Dir(path), fmt.Sprintf(".%s.tmp-%d-%s", filepath.Base(path), os.Getpid(), hex.EncodeToString(rnd[:])))
	f, err := os.OpenFile(name, os.O_WRONLY|os.O_CREATE|os.O_EXCL, perm)
	if err != nil {
		return "", err
	}
	_, err = f.Write(data)
	if err == nil {
		err = f.Sync()
	}
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		os.Remove(name)
		return "", err
	}
	return name, nil
}

func syncDir(dir string) {
	if d, err := os.Open(dir); err == nil {
		_ = d.Sync()
		d.Close()
	}
}

// CleanTemps removes the temporary files crashed writers left in dir
// (".*.tmp-*" older than olderThan; spec §5.2 says an hour). It returns
// how many it removed.
func CleanTemps(dir string, olderThan time.Duration) int {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 0
	}
	n := 0
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasPrefix(name, ".") || !strings.Contains(name, ".tmp-") {
			continue
		}
		info, err := e.Info()
		if err != nil || time.Since(info.ModTime()) < olderThan {
			continue
		}
		if os.Remove(filepath.Join(dir, name)) == nil {
			n++
		}
	}
	return n
}

// CleanAllTemps runs CleanTemps over the family directory's own
// directories; apps call it at start-up.
func (d Dir) CleanAllTemps() int {
	n := 0
	for _, dir := range []string{d.Root, d.AppsDir(), d.ModelsDir(), d.ClaimsDir(), d.RuntimeDir(), d.AccountDir()} {
		n += CleanTemps(dir, time.Hour)
	}
	return n
}
