// Package update holds what the updater of package mygo and `mygo build`
// share: the update manifest, signatures, versions, and the formats of
// archives and delta updates.
package update

import (
	"archive/tar"
	"compress/gzip"
	"crypto/ed25519"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
)

// Manifest describes the latest version of an app for one target (such as
// darwin-arm64), published as update-<target>.json next to the archive.
type Manifest struct {
	Version string `json:"version"`
	// Notes are the release notes, in Markdown.
	Notes string `json:"notes,omitempty"`
	// Date is when the version was built, in RFC 3339 format.
	Date string `json:"date,omitempty"`
	// URL of the archive, its size in bytes and the Ed25519 signature of
	// its SHA-256 digest, in base64.
	URL       string `json:"url"`
	Size      int64  `json:"size"`
	Signature string `json:"signature"`
	// Deltas update earlier versions with smaller downloads (delta.go).
	Deltas []Delta `json:"deltas,omitempty"`
	// Previous are the archives of the versions before, newest first,
	// which `mygo build` makes the deltas of the next version from.
	Previous []Archive `json:"previous,omitempty"`
}

// Delta is a delta update of the app of version From, signed like the
// archive.
type Delta struct {
	From      string `json:"from"`
	URL       string `json:"url"`
	Size      int64  `json:"size"`
	Signature string `json:"signature"`
}

// Archive is the archive of an earlier version.
type Archive struct {
	Version   string `json:"version"`
	URL       string `json:"url"`
	Size      int64  `json:"size"`
	Signature string `json:"signature"`
}

// Delta returns the delta update of version from, or nil.
func (m *Manifest) Delta(from string) *Delta {
	for i, d := range m.Deltas {
		if d.From == from {
			return &m.Deltas[i]
		}
	}
	return nil
}

// ManifestName is the file name of the manifest of target.
func ManifestName(target string) string { return "update-" + target + ".json" }

// Limits of what an app downloads.
const (
	MaxManifestSize = 1 << 20
	MaxArchiveSize  = 1 << 30
)

// Keys are stored in base64: the 32-byte public key and the 64-byte
// private key of Ed25519.

// ParsePublicKey decodes a public key.
func ParsePublicKey(s string) (ed25519.PublicKey, error) {
	b, err := base64.StdEncoding.DecodeString(strings.TrimSpace(s))
	if err != nil || len(b) != ed25519.PublicKeySize {
		return nil, errors.New("invalid update public key")
	}
	return ed25519.PublicKey(b), nil
}

// ParsePrivateKey decodes a private key.
func ParsePrivateKey(s string) (ed25519.PrivateKey, error) {
	b, err := base64.StdEncoding.DecodeString(strings.TrimSpace(s))
	if err != nil || len(b) != ed25519.PrivateKeySize {
		return nil, errors.New("invalid update private key")
	}
	return ed25519.PrivateKey(b), nil
}

// Sign signs the SHA-256 digest of an archive.
func Sign(key ed25519.PrivateKey, digest []byte) string {
	return base64.StdEncoding.EncodeToString(ed25519.Sign(key, digest))
}

// Verify checks the signature of the SHA-256 digest of an archive.
func Verify(key ed25519.PublicKey, digest []byte, signature string) error {
	sig, err := base64.StdEncoding.DecodeString(signature)
	if err != nil || !ed25519.Verify(key, digest, sig) {
		return errors.New("the update is not signed with the app's key")
	}
	return nil
}

// Compare compares versions such as 1.2.3 and 1.3.0-beta.1 like semantic
// versioning does: numbers numerically, a pre-release before its release.
func Compare(a, b string) int {
	a, b = strings.TrimPrefix(a, "v"), strings.TrimPrefix(b, "v")
	a, _, _ = strings.Cut(a, "+")
	b, _, _ = strings.Cut(b, "+")
	ar, apre, _ := strings.Cut(a, "-")
	br, bpre, _ := strings.Cut(b, "-")
	if c := compareFields(ar, br); c != 0 {
		return c
	}
	switch {
	case apre == bpre:
		return 0
	case apre == "":
		return 1
	case bpre == "":
		return -1
	}
	return compareFields(apre, bpre)
}

func compareFields(a, b string) int {
	af, bf := strings.Split(a, "."), strings.Split(b, ".")
	for i := 0; i < len(af) || i < len(bf); i++ {
		var x, y string
		if i < len(af) {
			x = af[i]
		}
		if i < len(bf) {
			y = bf[i]
		}
		xn, xerr := strconv.Atoi(x)
		yn, yerr := strconv.Atoi(y)
		switch {
		case x == y:
			continue
		case x == "":
			return -1
		case y == "":
			return 1
		case xerr == nil && yerr == nil:
			if xn < yn {
				return -1
			}
			if xn > yn {
				return 1
			}
		case xerr == nil:
			return -1 // numbers sort before words
		case yerr == nil:
			return 1
		default:
			return strings.Compare(x, y)
		}
	}
	return 0
}

// WriteArchive writes the entries of dir (files, directories and symbolic
// links, with their permissions) as a gzip compressed tar archive.
func WriteArchive(w io.Writer, dir string, entries []string) error {
	gz, err := gzip.NewWriterLevel(w, gzip.BestCompression)
	if err != nil {
		return err
	}
	tw := tar.NewWriter(gz)
	for _, entry := range entries {
		err := filepath.WalkDir(filepath.Join(dir, entry), func(p string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			info, err := d.Info()
			if err != nil {
				return err
			}
			rel, err := filepath.Rel(dir, p)
			if err != nil {
				return err
			}
			link := ""
			if info.Mode()&fs.ModeSymlink != 0 {
				if link, err = os.Readlink(p); err != nil {
					return err
				}
			}
			hdr, err := tar.FileInfoHeader(info, link)
			if err != nil {
				return err
			}
			hdr.Name = filepath.ToSlash(rel)
			if d.IsDir() {
				hdr.Name += "/"
			}
			hdr.Uid, hdr.Gid, hdr.Uname, hdr.Gname = 0, 0, "", ""
			if err := tw.WriteHeader(hdr); err != nil {
				return err
			}
			if info.Mode().IsRegular() {
				f, err := os.Open(p)
				if err != nil {
					return err
				}
				_, err = io.Copy(tw, f)
				f.Close()
				return err
			}
			return nil
		})
		if err != nil {
			return err
		}
	}
	if err := tw.Close(); err != nil {
		return err
	}
	return gz.Close()
}

// ExtractArchive unpacks an archive of WriteArchive into dir. It refuses
// entries outside dir, links pointing out of it and anything but files,
// directories and symbolic links.
func ExtractArchive(r io.Reader, dir string) error {
	gz, err := gzip.NewReader(r)
	if err != nil {
		return err
	}
	defer gz.Close()
	tr := tar.NewReader(gz)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		name := path.Clean(hdr.Name)
		if !fs.ValidPath(name) || name == "." {
			return fmt.Errorf("invalid path in update archive: %q", hdr.Name)
		}
		target := filepath.Join(dir, filepath.FromSlash(name))
		mode := fs.FileMode(hdr.Mode).Perm()
		switch hdr.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0o755); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
				return err
			}
			f, err := os.OpenFile(target, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode|0o200)
			if err != nil {
				return err
			}
			_, err = io.Copy(f, tr)
			if cerr := f.Close(); err == nil {
				err = cerr
			}
			if err != nil {
				return err
			}
		case tar.TypeSymlink:
			resolved := path.Join(path.Dir(name), hdr.Linkname)
			if path.IsAbs(hdr.Linkname) || !fs.ValidPath(resolved) {
				return fmt.Errorf("update archive link %q points outside the app", hdr.Name)
			}
			if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
				return err
			}
			if err := os.Symlink(hdr.Linkname, target); err != nil {
				return err
			}
		default:
			return fmt.Errorf("unsupported entry in update archive: %q", hdr.Name)
		}
	}
}

// ReleaseNotes returns the section of a Markdown changelog under the
// "## <version>" heading, or "## [<version>]" as in Keep a Changelog (a date
// or other text may follow the version), or "" when there is none.
func ReleaseNotes(changelog, version string) string {
	var notes []string
	in := false
	version = strings.TrimPrefix(version, "v")
	for _, line := range strings.Split(strings.ReplaceAll(changelog, "\r\n", "\n"), "\n") {
		if rest, ok := strings.CutPrefix(line, "## "); ok {
			if in {
				break
			}
			fields := strings.Fields(rest)
			in = len(fields) > 0 && strings.TrimPrefix(headingVersion(fields[0]), "v") == version
			continue
		}
		if in {
			notes = append(notes, line)
		}
	}
	return strings.TrimSpace(strings.Join(notes, "\n"))
}

// headingVersion returns the version a changelog heading starts with, from
// "1.2.0", or from "[1.2.0]" and "[1.2.0](link)" as Keep a Changelog and
// release tools write it.
func headingVersion(s string) string {
	if rest, ok := strings.CutPrefix(s, "["); ok {
		s, _, _ = strings.Cut(rest, "]")
	}
	return s
}
