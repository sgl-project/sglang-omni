package main

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
)

// Windows configures Windows packaging.
type Windows struct {
	// Certificate is a code signing certificate (.pfx) to sign the
	// executable, the installer and its uninstaller with, which keeps
	// SmartScreen from warning users; signing the uninstaller needs NSIS
	// 3.08 or later. Its password comes from
	// MYGO_WINDOWS_CERTIFICATE_PASSWORD. signtool signs on Windows,
	// osslsigncode elsewhere.
	Certificate string `json:"certificate"`
	// SignCommand signs a file instead, with %1 standing for its path, e.g.
	// for Azure Trusted Signing or a certificate on a hardware token:
	//   "signtool sign /fd sha256 /tr http://timestamp.digicert.com /td sha256 /a %1"
	SignCommand string `json:"signCommand"`
	// TimestampURL is the RFC 3161 time stamping server of Certificate
	// (default http://timestamp.digicert.com).
	TimestampURL string `json:"timestampUrl"`
}

func (w *Windows) signs() bool { return w.Certificate != "" || w.SignCommand != "" }

// signWindows signs an executable or installer.
func signWindows(c *Config, file string) error {
	t := con.start("Signing " + filepath.Base(file))
	return t.end(signWindowsFile(c, file), "Signed "+filepath.Base(file))
}

func signWindowsFile(c *Config, file string) error {
	w := c.Windows
	if w.SignCommand != "" {
		// Not shown: the command may hold credentials.
		cmd := shellCommand(c.root, strings.ReplaceAll(w.SignCommand, "%1", shellQuote(file)))
		out, errOut := labeledOutput(cmd, "sign")
		err := waited(cmd.Run())
		out.flush()
		errOut.flush()
		if err != nil {
			return fmt.Errorf("windows.signCommand: %w", err)
		}
		return nil
	}
	password := os.Getenv("MYGO_WINDOWS_CERTIFICATE_PASSWORD")
	cert := c.path(w.Certificate)
	ts := w.TimestampURL
	if ts == "" {
		ts = "http://timestamp.digicert.com"
	}
	if runtime.GOOS == "windows" {
		tool := signtool()
		if tool == "" {
			return errors.New("signing needs signtool from the Windows SDK")
		}
		args := []string{"sign", "/f", cert, "/fd", "sha256", "/tr", ts, "/td", "sha256"}
		if password != "" {
			args = append(args, "/p", password)
		}
		if out, err := exec.Command(tool, append(args, file)...).CombinedOutput(); err != nil {
			return fmt.Errorf("signtool: %v\n%s", err, out)
		}
		return nil
	}
	tool, err := exec.LookPath("osslsigncode")
	if err != nil {
		return errors.New("signing Windows apps on this system needs osslsigncode (brew install osslsigncode, apt install osslsigncode)")
	}
	signed := file + ".signed"
	args := []string{"sign", "-pkcs12", cert, "-h", "sha256", "-ts", ts, "-in", file, "-out", signed}
	if password != "" {
		args = append(args, "-pass", password)
	}
	if out, err := exec.Command(tool, args...).CombinedOutput(); err != nil {
		os.Remove(signed)
		return fmt.Errorf("osslsigncode: %v\n%s", err, out)
	}
	return os.Rename(signed, file)
}

// The environment of the mygo that makensis runs to sign the uninstaller
// (see uninstallerSigning): the executable, and the signSettings in JSON.
const (
	signerEnv       = "MYGO_SIGNER"
	signSettingsEnv = "MYGO_SIGN_SETTINGS"
)

// signSettings are what `mygo sign-uninstaller` signs with: the Windows
// configuration of the build, and the project directory its paths are
// relative to.
type signSettings struct {
	Root    string  `json:"root"`
	Windows Windows `json:"windows"`
}

// runSignUninstaller signs the file makensis names, the uninstaller it
// made, for the build that runs makensis.
func runSignUninstaller(args []string) error {
	var s signSettings
	if len(args) != 1 || json.Unmarshal([]byte(os.Getenv(signSettingsEnv)), &s) != nil {
		return errors.New("sign-uninstaller is for makensis, which mygo build runs")
	}
	return signWindows(&Config{root: s.Root, Windows: s.Windows}, args[0])
}

// signWindowsResources signs the executables and libraries among the
// resources copied into dir, such as helper programs, that carry no
// signature: those signed by their publishers keep their signatures.
func signWindowsResources(c *Config, dir string, res []resource) error {
	for _, r := range res {
		if r.src == "" {
			continue // a directory whose contents are resources of their own
		}
		copied := resource{src: filepath.Join(dir, filepath.FromSlash(r.name)), nested: r.nested}
		err := copied.walk(func(path string, info fs.FileInfo) error {
			if !info.Mode().IsRegular() {
				return nil
			}
			if image, signed := peImage(path); image && !signed {
				return signWindows(c, path)
			}
			return nil
		})
		if err != nil {
			return err
		}
	}
	return nil
}

// peImage reports whether the file at path is a PE image, an executable or
// a DLL, and whether it carries an Authenticode signature.
func peImage(path string) (image, signed bool) {
	f, err := os.Open(path)
	if err != nil {
		return false, false
	}
	defer f.Close()
	var dos [64]byte
	if _, err := io.ReadFull(f, dos[:]); err != nil || string(dos[:2]) != "MZ" {
		return false, false
	}
	// The PE signature, the file header, then the optional header.
	pe := int64(binary.LittleEndian.Uint32(dos[0x3c:]))
	var h [26]byte
	if _, err := f.ReadAt(h[:], pe); err != nil || string(h[:4]) != "PE\x00\x00" {
		return false, false
	}
	optionalSize := int64(binary.LittleEndian.Uint16(h[20:]))
	var dirs int64 // where the data directories start in the optional header
	switch binary.LittleEndian.Uint16(h[24:]) {
	case 0x10b: // PE32
		dirs = 96
	case 0x20b: // PE32+
		dirs = 112
	default:
		return false, false
	}
	// NumberOfRvaAndSizes, then the directories: the certificate table is
	// the fifth, and its size follows its offset.
	const security = 4
	if optionalSize < dirs+(security+1)*8 {
		return true, false
	}
	var d [4 + (security+1)*8]byte
	if _, err := f.ReadAt(d[:], pe+24+dirs-4); err != nil {
		return true, false
	}
	count := binary.LittleEndian.Uint32(d[:])
	return true, count > security && binary.LittleEndian.Uint32(d[4+security*8+4:]) != 0
}

// signtool finds signtool.exe of the newest Windows SDK.
func signtool() string {
	if p, err := exec.LookPath("signtool"); err == nil {
		return p
	}
	arch := "x64"
	if runtime.GOARCH == "arm64" {
		arch = "arm64"
	}
	matches, _ := filepath.Glob(filepath.Join(os.Getenv("ProgramFiles(x86)"), "Windows Kits", "10", "bin", "10.*", arch, "signtool.exe"))
	if len(matches) == 0 {
		return ""
	}
	sort.Strings(matches)
	return matches[len(matches)-1]
}

// shellQuote quotes an argument for the shell that run uses.
func shellQuote(s string) string {
	if runtime.GOOS == "windows" {
		return `"` + s + `"`
	}
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}
