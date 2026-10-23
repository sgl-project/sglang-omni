package main

// Settings → About (诊断与关于; AudioSmith's, for VoxtNew): versions
// and the machine, a self-check (library, helper, models, data folder,
// permissions), "Copy diagnostics", a diagnostic bundle (a ZIP
// of the logs, the settings and the diagnostics) and Report a bug, which
// opens a prefilled GitHub issue in the browser.

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
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/icons"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

// The project's addresses: the product page and the downloads are on
// actnow.dev (the ActNow family's site, which also serves the updates);
// issuesURL is where bugs are reported; the licence, the source and the
// third-party notices are linked from Settings → About.
const (
	websiteURL  = "https://actnow.dev/voxtnew"
	downloadURL = "https://actnow.dev/download/voxtnew"
	repoURL     = "https://github.com/sgl-project/sglang-omni"
	sourceURL   = repoURL + "/tree/main/VoxtNew"
	issuesURL   = repoURL + "/issues/new"
	licenceURL  = repoURL + "/blob/main/VoxtNew/LICENSE"
	noticeURL   = repoURL + "/blob/main/VoxtNew/NOTICE"
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
	line("VoxtNew", ver)
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
	st, hs, data := a.lib.store, a.helperStatus(), a.dataDir()
	a.recheckPermissions()
	perms := a.perms
	mods := a.modelCheckInput() // settings_about_models.go
	a.async(func() {
		var out []selfCheck
		add := func(id, title, status, detail string) {
			out = append(out, selfCheck{id: id, title: title, status: status, detail: detail})
		}
		// The library.
		if s, err := st.Stats(context.Background()); err != nil {
			add("library", i18n.T("Library database"), "fail", err.Error())
		} else {
			add("library", i18n.T("Library database"), "pass", i18n.T("%d history entries, %d terms", s.History, s.Dictionary))
		}
		// The helper.
		switch hs.State {
		case helper.Ready:
			add("helper", i18n.T("Helper"), "pass", i18n.T("Ready (%s)", hs.Hello.Version))
		case helper.Starting:
			add("helper", i18n.T("Helper"), "warn", i18n.T("Starting…"))
		default:
			add("helper", i18n.T("Helper"), "fail", uitext.OrDash(hs.Err))
		}
		// The data folder.
		os.MkdirAll(data, 0o755)
		if tmp, err := os.CreateTemp(data, ".voxtnew-check-*"); err != nil {
			add("data", i18n.T("Data folder"), "fail", i18n.T("Cannot write in %s: %v", data, err))
		} else {
			tmp.Close()
			os.Remove(tmp.Name())
			if free := diskFree(data); free > 0 && free < 1<<30 {
				add("data", i18n.T("Data folder"), "warn", i18n.T("Only %s free.", uitext.HumanBytes(free)))
			} else {
				add("data", i18n.T("Data folder"), "pass", i18n.T("Writable · %s free", uitext.HumanBytes(free)))
			}
		}
		// The models and their runtimes.
		modelChecks(mods, add)
		// Permissions.
		if t, ok := uitext.PermissionText(perms.mic); ok {
			add("microphone", i18n.T("Microphone"), "pass", t)
		} else {
			add("microphone", i18n.T("Microphone"), "warn", t)
		}
		if perms.access {
			add("accessibility", i18n.T("Accessibility"), "pass", i18n.T("Allowed"))
		} else {
			add("accessibility", i18n.T("Accessibility"), "warn", i18n.T("Not allowed: results are copied instead of typed."))
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
	name := "VoxtNew diagnostics " + time.Now().Format("2006-01-02 1504") + ".zip"
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

// aboutSettings is Settings → About.
func (a *app) aboutSettings(c *ui.Context) {
	t := c.Theme()
	a.loadSysInfo()
	sectionTitle(c, i18n.T("About"))
	ver := a.version
	if ver == "" {
		ver = i18n.T("development build")
	}
	hv := "—"
	if st := a.helperStatus(); st.Hello != nil {
		hv = st.Hello.Version
	}
	rows := [][2]string{
		{"VoxtNew", ver},
		{i18n.T("Helper"), hv},
		{i18n.T("Helper protocol"), fmt.Sprint(proto.Version)},
	}
	if s := a.sys; s != nil {
		rows = append(rows,
			[2]string{i18n.T("System"), uitext.OrDash(s.OS)},
			[2]string{i18n.T("Chip"), uitext.OrDash(s.Chip)},
			[2]string{i18n.T("Memory"), i18n.T("%d GB", s.MemoryGB())},
		)
	}
	ui.Grid(c.Key("about-grid")).ColumnTracks(ui.Fixed(px(t, 150)), ui.Fr(1)).GapX(px(t, sp3)).GapY(px(t, 6)).Children(func() {
		for _, r := range rows {
			ui.Text(c, r[0]).TextColor(t.TextMuted)
			ui.Text(c, r[1]).Selectable()
		}
	})
	// The legal notices: the copyright, the licence (Apache-2.0), the
	// source, and NOTICE (where the code came from, third parties).
	caption(c, i18n.T("Copyright © 2026 ActNow and the SGLang-Omni Team."))
	caption(c, i18n.T("VoxtNew is licensed under the Apache License 2.0 and comes with no warranty."))
	ui.Row(c.Key("about-licence")).Gap(px(t, sp3)).AlignItems(ui.Center).Wrap().Children(func() {
		ui.Link(c.Key("about-website"), i18n.T("Website"), websiteURL).FontSize(px(t, fsCaption))
		ui.Link(c.Key("about-download"), i18n.T("Download"), downloadURL).FontSize(px(t, fsCaption))
		ui.Link(c.Key("about-licence-apache"), i18n.T("Licence"), licenceURL).FontSize(px(t, fsCaption))
		ui.Link(c.Key("about-licence-source"), i18n.T("Source code"), sourceURL).FontSize(px(t, fsCaption))
		ui.Link(c.Key("about-licence-notices"), i18n.T("Third-party notices"), noticeURL).FontSize(px(t, fsCaption))
	})
	ui.Row(c.Key("about-icons")).Gap(px(t, 6)).AlignItems(ui.Center).Wrap().Children(func() {
		caption(c, i18n.T("Icons: Solar by 480 Design, under CC BY 4.0 (@iconify-json/solar %s).", icons.SolarVersion))
		ui.Link(c, i18n.T("Licence"), "https://creativecommons.org/licenses/by/4.0/").FontSize(px(t, fsCaption))
	})

	a.updateSettingsSection(c) // updates.go

	sectionTitle(c, i18n.T("Self-check"))
	settingRow(c, "about-check", i18n.T("Check VoxtNew"), i18n.T("The library, the helper, the models, the data folder and the permissions."), func() {
		if a.selfChecking {
			ui.Spinner(c)
		}
		if ui.Button(c.Key("about-check"), i18n.T("Run the check")).Disabled(a.selfChecking).Clicked() {
			a.runSelfCheck()
		}
	})
	for _, ch := range a.checks {
		ui.Row(c.Key("check-" + ch.id)).Gap(px(t, sp2)).AlignItems(ui.Start).Children(func() {
			mark, col := icons.CheckCircle, t.Success
			switch ch.status {
			case "warn":
				mark, col = icons.DangerTriangle, t.Warning
			case "fail":
				mark, col = icons.CloseCircle, t.Danger
			}
			ui.Icon(c, mark.Linear).Size(px(t, 14), px(t, 14)).TextColor(col).Shrink(0).Margin(px(t, 2), 0, 0)
			ui.Text(c, ch.title).Width(px(t, 150))
			ui.Text(c, ch.detail).FontSize(px(t, fsCaption)).TextColor(t.TextMuted).Grow(1).MinWidth(0)
		})
	}

	sectionTitle(c, i18n.T("Report a problem"))
	settingRow(c, "about-copy", i18n.T("Copy diagnostics"), i18n.T("Versions, the machine, the models, folders and the self-check: no text or audio."), func() {
		if ui.Button(c.Key("about-copy"), i18n.T("Copy")).Clicked() {
			c.WriteClipboard(a.diagnosticsText())
			a.toast(i18n.T("Diagnostics copied"))
		}
	})
	settingRow(c, "about-bundle", i18n.T("Save a diagnostic bundle"), i18n.T("A ZIP of the diagnostics, the logs and the settings, to attach to a report."), func() {
		if ui.Button(c.Key("about-bundle"), i18n.T("Save…")).Clicked() {
			a.saveBundle()
		}
	})
	settingRow(c, "about-bug", i18n.T("Report a bug"), i18n.T("Opens a new issue on GitHub in your browser, with the diagnostics filled in. Nothing is sent until you submit it."), func() {
		if ui.Button(c.Key("about-bug"), i18n.T("Report…")).Clicked() {
			u := a.bugReportURL()
			if a.win == nil {
				c.Services().OpenURL(u)
			} else {
				go mygo.Shell.OpenExternal(u)
			}
		}
	})
}
