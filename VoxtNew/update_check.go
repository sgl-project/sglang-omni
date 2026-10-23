//go:build updatecheck

package main

// The update check (scripts/mac-update-e2e.sh): with -tags updatecheck,
// `VoxtNew --update-check <report.json>` runs instead of the app and
// drives MyGo's updater with no window, no data, no lock: it checks the
// build's feed, installs what it finds (the delta of the running version
// when there is one, else the archive), reads the version of the bundle on
// disk afterwards, and starts the helper supervisor as the app does, which
// must see the update (appbundle.Updated) and start no helper. It writes
// what happened to <report.json> and quits. With --update-beta on it takes
// betas (mygo.Updater.SetPrereleases), as Beta updates does; without it,
// as the app does by default: betas when the running version is one.
// Release builds do not have it.

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"slices"
	"time"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/appbundle"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

func init() { updateCheck = runUpdateCheck }

// updateCheckReport is what the check found and did.
type updateCheckReport struct {
	Running  string `json:"running"`  // App.Version
	Beta     bool   `json:"beta"`     // the check took betas
	Enabled  bool   `json:"enabled"`  // Updater.Enabled
	Found    string `json:"found"`    // the version Check found, "" when up to date
	Notes    string `json:"notes"`    // its release notes
	CheckErr string `json:"checkErr"` // Check's error
	// NotPublished: Check's error is mygo.ErrNotPublished (no release on
	// the channel yet), which the update window shows as up to date.
	NotPublished bool `json:"notPublished"`
	// Totals are the sizes the progress reported, one per download: the
	// delta's alone, or the archive's (after the delta's when it failed).
	Totals     []int64 `json:"totals"`
	Downloaded int64   `json:"downloaded"`
	InstallErr string  `json:"installErr"`
	// After the install: the bundle on disk, and the supervisor's view.
	DiskVersion  string `json:"diskVersion"`
	Stale        bool   `json:"stale"`
	HelperState  string `json:"helperState"`
	HelperReason string `json:"helperReason"`
	HelperPID    int    `json:"helperPID"`
}

func runUpdateCheck(out string) {
	rep := updateCheckReport{Running: mygo.App.Version(), Enabled: mygo.Updater.Enabled()}
	var beta settings.Updates
	if v := argAfter("--update-beta"); v != "" {
		on := v == "on"
		beta.Beta = &on
	}
	rep.Beta = beta.BetaOn(rep.Running)
	mygo.Updater.SetPrereleases(rep.Beta)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	up, err := mygo.Updater.Check(ctx)
	if err != nil {
		rep.CheckErr, rep.NotPublished = err.Error(), errors.Is(err, mygo.ErrNotPublished)
	}
	if up != nil {
		rep.Found, rep.Notes = up.Version, up.Notes
		err := up.Install(ctx, func(downloaded, total int64) {
			if n := len(rep.Totals); n == 0 || rep.Totals[n-1] != total || downloaded < rep.Downloaded {
				rep.Totals = append(rep.Totals, total)
			}
			rep.Downloaded = downloaded
		})
		if err != nil {
			rep.InstallErr = err.Error()
		}
	}
	resources, _ := mygo.App.Path(mygo.PathResources)
	rep.DiskVersion, _ = appbundle.Version(resources)
	stale := func() bool { return appbundle.Updated(resources, rep.Running) }
	rep.Stale = stale()
	if rep.Stale {
		// As the app makes it (helperlink.go): it must start nothing.
		sup := helper.New(helper.Config{Bin: helper.Find(resources), AppVersion: rep.Running, Bus: core.NewBus(), Stale: stale,
			Logf: func(string, ...any) {}})
		sup.Start(ctx)
		st := sup.Status()
		sup.Stop()
		rep.HelperState, rep.HelperReason, rep.HelperPID = string(st.State), string(st.Reason), st.PID
	}
	rep.Totals = slices.Clip(rep.Totals)
	data, _ := json.MarshalIndent(rep, "", "  ")
	if err := os.WriteFile(out, append(data, '\n'), 0o644); err != nil {
		os.Stderr.WriteString(err.Error() + "\n")
		os.Exit(1)
	}
}
