package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	fetch "github.com/egoist/mygo/internal/download"
	"github.com/egoist/mygo/internal/nsis"
)

// The Windows installer is built with NSIS: a per-user install in
// %LOCALAPPDATA%\Programs, where the app can update itself, with a Start
// menu shortcut, a desktop shortcut the finish page offers, and an
// uninstaller that Settings > Apps lists. Windows rarely has NSIS
// installed, so there mygo build downloads it when it needs it, as Tauri
// and electron-builder do; elsewhere makensis must be installed.

// makensis finds the NSIS compiler: an installed one or, on Windows, the
// one an earlier build downloaded.
func makensis() string {
	if p, err := exec.LookPath("makensis"); err == nil {
		return p
	}
	if runtime.GOOS == "windows" {
		for _, env := range []string{"ProgramFiles(x86)", "ProgramFiles"} {
			if p := filepath.Join(os.Getenv(env), "NSIS", "makensis.exe"); fileExists(p) {
				return p
			}
		}
		if dir := nsis.Dir(); dir != "" {
			return nsis.Executable(dir)
		}
	}
	return ""
}

// nsisCompiler returns the NSIS compiler, which it downloads on Windows
// when there is none, or "" on other systems without NSIS.
func nsisCompiler() (string, error) {
	if p := makensis(); p != "" || runtime.GOOS != "windows" {
		return p, nil
	}
	dir := nsis.Dir()
	if dir == "" {
		return "", errors.New("no cache directory to download NSIS into: install NSIS (makensis)")
	}
	return downloadNSIS(dir)
}

// downloadNSIS downloads nsis.Release into dir, unless it is there
// already, and returns its makensis.
func downloadNSIS(dir string) (string, error) {
	if p := nsis.Executable(dir); p != "" {
		return p, nil
	}
	v := nsis.Release.Version
	t := con.start("Downloading NSIS " + v + " for the Windows installer")
	p, err := nsis.Download(context.Background(), dir, func(url string) fetch.File { return downloadFile(t, url, "") })
	if t.end(err, "Downloaded NSIS "+v) != nil {
		return "", fmt.Errorf("downloading NSIS %s (or install it: https://nsis.sourceforge.io):\n%w", v, err)
	}
	return p, nil
}

// downloadHeaderTimeout is how long the CLI's downloads wait for a host to
// answer, so that one that takes connections and never answers gives way
// to the next copy.
var downloadHeaderTimeout = 30 * time.Second

// downloadFile is a download by the CLI of url, whose SHA-256 is sum: its
// progress shows on t, and a host that does not answer within
// downloadHeaderTimeout fails it.
func downloadFile(t *task, url, sum string) fetch.File {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.ResponseHeaderTimeout = downloadHeaderTimeout
	return fetch.File{
		URL:    url,
		SHA256: sum,
		Client: &http.Client{Transport: transport, Timeout: 10 * time.Minute},
		Header: http.Header{"User-Agent": {"mygo/" + version}},
		Progress: func(total int64) io.Writer {
			return &downloadProgress{t: t, total: total}
		},
	}
}

// installerFileName is "<Name> Setup <version> <arch>.exe", as "My App
// Setup 1.2.0 amd64.exe": each architecture's installer is named apart, so
// that the builds of two, uploaded to one release or bucket, keep both.
func installerFileName(c *Config, goarch string) string {
	return fsName(c.Name) + " Setup " + fsName(c.Version) + " " + goarch + ".exe"
}

// writeInstaller builds installerFileName in stage from the app files
// installed, built for goarch, the executable exe among them, and returns
// its path, or "" without NSIS.
func writeInstaller(c *Config, stage, work, exe, goarch string, installed []string) (string, error) {
	tool, err := nsisCompiler()
	if err != nil {
		return "", err
	}
	if tool == "" {
		warnf("Skipping the Windows installer: install NSIS (makensis)")
		return "", nil
	}
	out := filepath.Join(stage, installerFileName(c, goarch))
	var files strings.Builder
	for _, name := range installed {
		p := filepath.Join(stage, name)
		if isDir(p) {
			fmt.Fprintf(&files, "  File /r %s\n", nsisString(p))
		} else {
			fmt.Fprintf(&files, "  File %s\n", nsisString(p))
		}
	}
	icon := ""
	if c.Icon != "" {
		src, err := os.ReadFile(c.path(c.Icon))
		if err != nil {
			return "", err
		}
		ico, err := pngToICO(src)
		if err != nil {
			return "", err
		}
		path := filepath.Join(work, "installer.ico")
		if err := os.WriteFile(path, ico, 0o644); err != nil {
			return "", err
		}
		icon = "!define MUI_ICON " + nsisString(path) + "\n!define MUI_UNICON " + nsisString(path) + "\n"
	}
	signing, env, err := uninstallerSigning(c, tool)
	if err != nil {
		return "", err
	}
	uninstallKey := `Software\Microsoft\Windows\CurrentVersion\Uninstall\` + c.Identifier
	shortcut := nsisEscape(fsName(c.Name)) + ".lnk"
	register, unregister := nsisAssociations(c, exe)
	langs, err := c.languages()
	if err != nil {
		return "", err
	}
	languages, err := nsisLanguageMacros(langs)
	if err != nil {
		return "", err
	}
	script := `Unicode true
ManifestDPIAware true
SetCompressor /SOLID lzma
Name ` + nsisString(c.Name) + `
OutFile ` + nsisString(out) + `
` + signing + `InstallDir "$LOCALAPPDATA\Programs\` + nsisEscape(fsName(c.Name)) + `"
RequestExecutionLevel user
BrandingText " "
` + icon + `!define MUI_FINISHPAGE_RUN "$INSTDIR\` + nsisEscape(exe) + `"
!define MUI_FINISHPAGE_SHOWREADME
!define MUI_FINISHPAGE_SHOWREADME_TEXT "Create a desktop shortcut"
!define MUI_FINISHPAGE_SHOWREADME_FUNCTION CreateDesktopShortcut
!include "MUI2.nsh"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
` + languages + `
Section
  SetOutPath "$INSTDIR"
` + files.String() + `  WriteUninstaller "$INSTDIR\Uninstall.exe"
  CreateShortCut "$SMPROGRAMS\` + shortcut + `" "$INSTDIR\` + nsisEscape(exe) + `"
  WriteRegStr HKCU ` + nsisString(uninstallKey) + ` "DisplayName" ` + nsisString(c.Name) + `
  WriteRegStr HKCU ` + nsisString(uninstallKey) + ` "DisplayVersion" ` + nsisString(c.Version) + `
  WriteRegStr HKCU ` + nsisString(uninstallKey) + ` "DisplayIcon" "$INSTDIR\` + nsisEscape(exe) + `"
  WriteRegStr HKCU ` + nsisString(uninstallKey) + ` "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU ` + nsisString(uninstallKey) + ` "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegStr HKCU ` + nsisString(uninstallKey) + ` "QuietUninstallString" '"$INSTDIR\Uninstall.exe" /S'
  WriteRegDWORD HKCU ` + nsisString(uninstallKey) + ` "NoModify" 1
  WriteRegDWORD HKCU ` + nsisString(uninstallKey) + ` "NoRepair" 1
` + register + `SectionEnd

; The finish page offers it, so silent installs (/S) make none.
Function CreateDesktopShortcut
  SetOutPath "$INSTDIR"
  CreateShortCut "$DESKTOP\` + shortcut + `" "$INSTDIR\` + nsisEscape(exe) + `"
FunctionEnd

; Deletes the shortcut on the stack, trying again for 5 s while it is in
; use: Explorer opens a new shortcut a few seconds after it appears, not
; sharing it for deletion, and Delete fails meanwhile. It succeeds when
; there is no shortcut.
Function un.DeleteShortcut
  Exch $0
  Push $1
  StrCpy $1 50
  retry:
    ClearErrors
    Delete $0
    IfErrors 0 done
    IntOp $1 $1 - 1
    IntCmp $1 0 done
    Sleep 100
    Goto retry
  done:
  Pop $1
  Pop $0
FunctionEnd

Section "Uninstall"
  Push "$SMPROGRAMS\` + shortcut + `"
  Call un.DeleteShortcut
  Push "$DESKTOP\` + shortcut + `"
  Call un.DeleteShortcut
  DeleteRegKey HKCU ` + nsisString(uninstallKey) + `
` + unregister + `  ; Last, so that the app's folder is gone once the uninstall is done:
  ; the uninstaller runs from a copy of itself that nothing waits for.
  RMDir /r "$INSTDIR"
SectionEnd
`
	nsi := filepath.Join(work, "installer.nsi")
	if err := os.WriteFile(nsi, []byte(script), 0o644); err != nil {
		return "", err
	}
	t := con.start("Creating " + filepath.Base(out))
	if signing != "" {
		t.set("signing Uninstall.exe")
	}
	cmd := exec.Command(tool, "-V2", nsi)
	cmd.Env = append(os.Environ(), env...)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.fail()
		return "", fmt.Errorf("makensis: %v\n%s", err, out)
	}
	t.done("Created " + filepath.Base(out))
	return out, nil
}

// uninstallerSigning returns the NSIS command that signs the uninstaller
// of a signed app, and the environment makensis needs to run it. makensis
// writes the uninstaller to a temporary file, runs the command with it and
// puts the signed file into the installer. The command runs mygo, which
// signs the uninstaller as it signs the app (runSignUninstaller); its path
// goes through the environment, since NSIS reads "$" in the script as its
// own syntax.
func uninstallerSigning(c *Config, tool string) (script string, env []string, err error) {
	if !c.Windows.signs() {
		return "", nil, nil
	}
	if major, minor := nsisVersion(tool); major < 3 || major == 3 && minor < 8 {
		warnf("Not signing Uninstall.exe: that needs NSIS 3.08 or later, not %s", tool)
		return "", nil, nil
	}
	self, err := os.Executable()
	if err != nil {
		return "", nil, err
	}
	settings, err := json.Marshal(signSettings{Root: c.root, Windows: c.Windows})
	if err != nil {
		return "", nil, err
	}
	signer := `"%` + signerEnv + `%"`
	if runtime.GOOS != "windows" {
		signer = `"$` + signerEnv + `"`
	}
	// "= 0" stops the build when signing fails; NSIS ignores the exit code
	// otherwise.
	return "!uninstfinalize '" + signer + ` sign-uninstaller "%1"' = 0` + "\n",
		[]string{signerEnv + "=" + self, signSettingsEnv + "=" + string(settings)}, nil
}

// nsisVersion returns the version of the NSIS compiler tool, or zeros when
// it does not tell.
func nsisVersion(tool string) (major, minor int) {
	out, err := exec.Command(tool, "-VERSION").Output()
	if err == nil {
		// "v3.13", or "v3.08-3" for a distribution's build.
		fmt.Sscanf(strings.TrimPrefix(strings.TrimSpace(string(out)), "v"), "%d.%d", &major, &minor)
	}
	return major, minor
}

// nsisAssociations returns the installer commands that register, and
// unregister, the file associations and URL schemes of the app for the
// user, opening them with exe.
func nsisAssociations(c *Config, exe string) (register, unregister string) {
	var r, u strings.Builder
	open := `'"$INSTDIR\` + nsisEscape(exe) + `" "%1"'`
	prefix := strings.Map(func(r rune) rune {
		if r < 0x80 && (r == '.' || r >= '0' && r <= '9' || r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z') {
			return r
		}
		return -1
	}, c.Identifier)
	for _, fa := range c.FileAssociations {
		for _, ext := range fa.Ext {
			progID := prefix + "." + strings.ToLower(ext)
			class := `Software\Classes\` + progID
			fmt.Fprintf(&r, "  WriteRegStr HKCU %s \"\" %s\n", nsisString(class), nsisString(fa.Name))
			fmt.Fprintf(&r, "  WriteRegStr HKCU %s \"\" \"$INSTDIR\\%s,0\"\n", nsisString(class+`\DefaultIcon`), nsisEscape(exe))
			fmt.Fprintf(&r, "  WriteRegStr HKCU %s \"\" %s\n", nsisString(class+`\shell\open\command`), open)
			fmt.Fprintf(&r, "  WriteRegStr HKCU %s %s \"\"\n", nsisString(`Software\Classes\.`+ext+`\OpenWithProgids`), nsisString(progID))
			fmt.Fprintf(&u, "  DeleteRegKey HKCU %s\n", nsisString(class))
			fmt.Fprintf(&u, "  DeleteRegValue HKCU %s %s\n", nsisString(`Software\Classes\.`+ext+`\OpenWithProgids`), nsisString(progID))
		}
	}
	for _, scheme := range c.URLSchemes {
		key := `Software\Classes\` + scheme
		fmt.Fprintf(&r, "  WriteRegStr HKCU %s \"\" %s\n", nsisString(key), nsisString("URL:"+c.Name))
		fmt.Fprintf(&r, "  WriteRegStr HKCU %s \"URL Protocol\" \"\"\n", nsisString(key))
		fmt.Fprintf(&r, "  WriteRegStr HKCU %s \"\" %s\n", nsisString(key+`\shell\open\command`), open)
		fmt.Fprintf(&u, "  DeleteRegKey HKCU %s\n", nsisString(key))
	}
	if r.Len() > 0 {
		// Tell Explorer the associations changed.
		notify := "  System::Call 'shell32::SHChangeNotify(i 0x08000000, i 0, p 0, p 0)'\n"
		r.WriteString(notify)
		u.WriteString(notify)
	}
	return r.String(), u.String()
}

// nsisEscape escapes text for an NSIS string in double quotes.
func nsisEscape(s string) string {
	return strings.NewReplacer(`$`, `$$`, `"`, `$\"`, "\n", `$\n`, "\r", `$\r`, "\t", `$\t`).Replace(s)
}

func nsisString(s string) string { return `"` + nsisEscape(s) + `"` }

// downloadProgress shows how much of a download arrived on t.
type downloadProgress struct {
	t     *task
	n     int64
	total int64 // or -1
}

func (p *downloadProgress) Write(b []byte) (int, error) {
	p.n += int64(len(b))
	if p.total > 0 {
		p.t.set(fmt.Sprintf("%s of %s", formatSize(p.n), formatSize(p.total)))
	} else {
		p.t.set(formatSize(p.n))
	}
	return len(b), nil
}
