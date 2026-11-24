package main

// Settings → About's controller: installation diagnostics, self-checks,
// diagnostic bundles and bug reports.

import (
	"archive/zip"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strings"
	"time"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

type aboutServices interface {
	WriteClipboard(text string)
	OpenURL(url string)
}

// The project's addresses: issuesURL is where bugs are reported; the
// licence, the source and the third-party notices are linked from
// Settings → About.
const (
	repoURL    = "https://github.com/sgl-project/sglang-omni"
	sourceURL  = repoURL + "/tree/main/Voxt"
	issuesURL  = repoURL + "/issues/new"
	licenceURL = repoURL + "/blob/main/Voxt/LICENSE"
	noticeURL  = repoURL + "/blob/main/Voxt/NOTICE"
)

// aboutState is the Diagnostics page's state.
type aboutState struct {
	sys        *sysInfo
	sysLoading bool

	checks       []selfCheck
	selfChecking bool
}

// selfCheck is one check's outcome.
type selfCheck struct {
	id     string
	title  string
	status string // "pass", "warn", "fail"
	detail string
}

// diagnosticsText describes this install, without anything the user
// said or wrote: for bug reports.
func (a *app) diagnosticsText() string {
	var b strings.Builder
	line := func(k, v string) { fmt.Fprintf(&b, "%s: %s\n", k, uitext.OrDash(v)) }
	ver := a.version
	if ver == "" {
		ver = "dev"
	}
	st := a.helperStatus()
	line("Voxt", ver)
	hv := string(st.State)
	if st.Hello != nil {
		hv = st.Hello.Version + " (" + hv + ")"
	}
	if st.Err != "" {
		hv += ": " + st.Err
	}
	line("Helper", hv)
	line("Helper protocol", fmt.Sprint(proto.Version))
	if s := a.sys; s != nil {
		line("System", s.OS)
		line("Chip", s.Chip+" · "+s.Arch+fmt.Sprintf(" · %d cores", s.Cores))
		line("Memory", fmt.Sprintf("%d GB", s.MemoryGB()))
	}
	line("Go", runtime.Version())
	line("MyGo", moduleVersion("github.com/egoist/mygo"))
	line("Data folder", fmt.Sprintf("%s (free %s)", a.dataDir(), uitext.HumanBytes(diskFree(a.dataDir()))))
	line("Language", i18n.Locale())
	a.modelDiagnostics(line) // settings_about_models.go
	for _, c := range a.checks {
		line("Check "+c.id, c.status+" "+c.detail)
	}
	return b.String()
}

// moduleVersion is a dependency's version from the build.
func moduleVersion(path string) string {
	if bi, ok := debug.ReadBuildInfo(); ok {
		for _, m := range bi.Deps {
			if m.Path == path {
				return m.Version
			}
		}
	}
	return ""
}

// runSelfCheck checks what VoxtNew needs, in the background.
func (a *app) runSelfCheck() {
	if a.selfChecking {
		return
	}
	a.selfChecking = true
	a.recheckPermissionsThen(a.selfCheckWith) // the permissions as read now
}

// selfCheckWith runs the self-check once the permissions were read.
func (a *app) selfCheckWith() {
	st, hs, data := a.lib.store, a.helperStatus(), a.dataDir()
	perms := a.perms
	mods := a.modelCheckInput() // settings_about_models.go
	translator := i18n.CurrentTranslator()
	a.async(func() {
		var out []selfCheck
		add := func(id, title, status, detail string) {
			out = append(out, selfCheck{id: id, title: title, status: status, detail: detail})
		}
		// The library.
		if s, err := st.Stats(context.Background()); err != nil {
			add("library", translator.T("Library database"), "fail", err.Error())
		} else {
			add("library", translator.T("Library database"), "pass", translator.T("%d history entries, %d terms", s.History, s.Dictionary))
		}
		// The helper.
		switch hs.State {
		case helper.Ready:
			add("helper", translator.T("Helper"), "pass", translator.T("Ready (%s)", hs.Hello.Version))
		case helper.Starting:
			add("helper", translator.T("Helper"), "warn", translator.T("Starting…"))
		default:
			add("helper", translator.T("Helper"), "fail", uitext.OrDash(hs.Err))
		}
		// The data folder.
		os.MkdirAll(data, 0o755)
		if tmp, err := os.CreateTemp(data, ".voxtnew-check-*"); err != nil {
			add("data", translator.T("Data folder"), "fail", translator.T("Cannot write in %s: %v", data, err))
		} else {
			tmp.Close()
			os.Remove(tmp.Name())
			if free := diskFree(data); free > 0 && free < 1<<30 {
				add("data", translator.T("Data folder"), "warn", translator.T("Only %s free.", uitext.HumanBytes(free)))
			} else {
				add("data", translator.T("Data folder"), "pass", translator.T("Writable · %s free", uitext.HumanBytes(free)))
			}
		}
		// The models and their runtimes.
		modelChecksFor(translator, mods, add)
		// Permissions.
		if text, ok := permissionStatusText(translator, perms.mic); ok {
			add("microphone", translator.T("Microphone"), "pass", text)
		} else {
			add("microphone", translator.T("Microphone"), "warn", text)
		}
		if perms.access {
			add("accessibility", translator.T("Accessibility"), "pass", translator.T("Allowed"))
		} else {
			add("accessibility", translator.T("Accessibility"), "warn", translator.T("Not allowed, so results are copied instead of typed."))
		}
		a.post(func() { a.checks, a.selfChecking = out, false })
	})
}

// writeBundle writes the diagnostic bundle (a ZIP) to path: the
// diagnostics, the settings (they hold no secret) and the log files.
func writeBundle(path, diag string, prefs settings.Settings, logDir string) error {
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	zw := zip.NewWriter(f)
	put := func(name string, data []byte) error {
		w, err := zw.Create(name)
		if err != nil {
			return err
		}
		_, err = w.Write(data)
		return err
	}
	err = put("diagnostics.txt", []byte(diag))
	if err == nil {
		data, _ := json.MarshalIndent(prefs, "", "  ")
		err = put("settings.json", data)
	}
	if err == nil && logDir != "" {
		ents, _ := os.ReadDir(logDir)
		for _, e := range ents {
			if e.IsDir() || !strings.Contains(e.Name(), ".log") {
				continue
			}
			src, rerr := os.Open(filepath.Join(logDir, e.Name()))
			if rerr != nil {
				continue
			}
			w, cerr := zw.Create("logs/" + e.Name())
			if cerr == nil {
				_, cerr = io.Copy(w, src)
			}
			src.Close()
			if cerr != nil {
				err = cerr
				break
			}
		}
	}
	if cerr := zw.Close(); err == nil {
		err = cerr
	}
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, path)
}

// bundleDir is where tests save the bundle (no dialog).
var bundleDir string

// saveBundle asks where, then writes the diagnostic bundle.
func (a *app) saveBundle() {
	diag, prefs, logs := a.diagnosticsText(), a.prefs, a.logDir
	name := "Voxt diagnostics " + time.Now().Format("2006-01-02 1504") + ".zip"
	write := func(dst string) {
		a.async(func() {
			err := writeBundle(dst, diag, prefs, logs)
			a.post(func() {
				if err != nil {
					a.toast(i18n.T("Could not save the bundle: %v", err))
					return
				}
				a.toast(i18n.T("Saved %s", filepath.Base(dst)))
			})
		})
	}
	if a.win == nil {
		if bundleDir != "" {
			write(filepath.Join(bundleDir, name))
		}
		return
	}
	go func() {
		dst, err := mygo.Dialog.Save(mygo.SaveDialogOptions{Parent: a.win, Title: i18n.T("Save a diagnostic bundle"), DefaultPath: name,
			Filters: []mygo.FileFilter{{Name: "ZIP", Extensions: []string{"zip"}}}})
		if err != nil || dst == "" {
			return
		}
		a.post(func() { write(dst) })
	}()
}

// bugReportURL is a new GitHub issue with the diagnostics filled in.
func (a *app) bugReportURL() string {
	diag := a.diagnosticsText()
	if len(diag) > 5000 {
		diag = diag[:5000]
	}
	body := "### What happened\n\n\n### What you expected\n\n\n### Steps to reproduce\n\n1. \n\n### Diagnostics\n\n```\n" + diag + "```\n"
	q := url.Values{"body": {body}, "labels": {"bug"}}
	return issuesURL + "?" + q.Encode()
}

func (a *app) aboutSettingsModel() settingsAboutViewState {
	helperVersion := "—"
	if status := a.helperStatus(); status.Hello != nil {
		helperVersion = status.Hello.Version
	}
	return buildSettingsAboutViewState(i18n.CurrentTranslator(), settingsAboutStateInput{
		Version: a.version, HelperVersion: helperVersion, SystemInfo: a.sys,
		SelfChecking: a.selfChecking, Checks: a.checks, Announced: a.announceCurrent.Version,
	})
}

func (a *app) openWhatsNew() {
	if n := a.announceCurrent; n.Version != "" {
		a.announceShown, a.announceShownOpen = n, true
	}
}

func (a *app) copyDiagnostics(services aboutServices) {
	services.WriteClipboard(a.diagnosticsText())
	a.toast(i18n.T("Diagnostics copied"))
}

func (a *app) reportBug(services aboutServices) {
	reportURL := a.bugReportURL()
	if a.win == nil {
		services.OpenURL(reportURL)
	} else {
		go mygo.Shell.OpenExternal(reportURL)
	}
}
