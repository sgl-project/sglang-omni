package models

import (
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// ResolveBlob returns the blob that a snapshot entry resolves to, whether the entry is a symlink, a hard link, or a copy.
func ResolveBlob(repoDir, link string) (string, error) {
	target, err := filepath.EvalSymlinks(link)
	if err != nil {
		return "", err
	}
	blobsDir := filepath.Join(repoDir, "blobs")
	if filepath.Dir(target) == blobsDir {
		return target, nil
	}
	info, err := os.Stat(target)
	if err != nil {
		return "", err
	}
	entries, err := os.ReadDir(blobsDir)
	if err != nil {
		return "", err
	}
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		candidate := filepath.Join(blobsDir, entry.Name())
		candidateInfo, err := os.Stat(candidate)
		if err == nil && os.SameFile(info, candidateInfo) {
			return candidate, nil
		}
	}
	lfsID, gitID, err := blobIDs(target, info.Size())
	if err != nil {
		return "", err
	}
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		if name := entry.Name(); name == lfsID || name == gitID {
			return filepath.Join(blobsDir, name), nil
		}
	}
	return "", fmt.Errorf("models: no blob for %s in %s", link, blobsDir)
}

func blobIDs(path string, size int64) (string, string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", "", err
	}
	defer file.Close()
	lfsHash := sha256.New()
	gitHash := sha1.New()
	fmt.Fprintf(gitHash, "blob %d\x00", size)
	if _, err := io.Copy(io.MultiWriter(lfsHash, gitHash), file); err != nil {
		return "", "", err
	}
	return hex.EncodeToString(lfsHash.Sum(nil)), hex.EncodeToString(gitHash.Sum(nil)), nil
}
