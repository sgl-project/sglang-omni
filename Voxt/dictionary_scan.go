package main

// The Dictionary page's history scan (internal/learn.Scanner): Scan
// history… sends the Transcribe results not read yet to Transcribe's
// language model and puts the names and terms it finds among the
// suggestions. The dialog says how many results go to which model before
// anything is sent; the scan goes on with the dialog closed.

import (
	"context"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/learn"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// scanState is the dialog. Before a scan: model and count. During it:
// running, read and total, and found, which a scan that ended also shows
// with its outcome (stopped, or failed).
type scanState struct {
	open     bool
	model    local.Model
	counted  bool
	count    int
	running  bool
	stopping bool
	cancel   context.CancelFunc
	read     int
	total    int
	found    int
	ended    bool
	stopped  bool
	failed   string
}

// scanModel is the language model a history scan uses: Transcribe's, when
// one is chosen and installed. Memory only (the build pass calls it).
func (a *app) scanModel() (local.Model, bool) {
	id := a.prefs.Features.Transcribe.LLM
	m, ok := local.Choose(a.installed(local.LLM), id)
	return m, ok && m.ID == id // Choose falls back to the first model
}

// scanOpen opens the model id for a scan (off the main thread).
func (a *app) scanOpen(id string) func(context.Context) (local.Chat, local.Model, error) {
	services := a.local
	return func(ctx context.Context) (local.Chat, local.Model, error) { return services.LLM(ctx, id) }
}

// openScan opens the dialog for model m: on a running scan's progress, or
// counting what a new scan would read.
func (a *app) openScan(m local.Model) {
	s := &a.dv.scan
	if s.running {
		s.open = true
		return
	}
	*s = scanState{open: true, model: m}
	scanner := a.scanner
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		count, err := scanner.Pending(ctx)
		a.post(func() {
			s := &a.dv.scan
			s.counted, s.count = true, count
			if err != nil {
				s.failed, _ = uitext.ModelErrorText(err)
			}
		})
	})
}

func scanFoundText(found int) string {
	return i18n.N(found, "Found %d suggestion", "Found %d suggestions", found)
}

// startScan runs the scan on a goroutine of its own, also in tests,
// where a.async would run it at once and Stop could never be pressed.
func (a *app) startScan() {
	s := &a.dv.scan
	ctx, cancel := context.WithCancel(context.Background())
	s.running, s.cancel, s.total = true, cancel, s.count
	scanner, modelID, open := a.scanner, s.model.ID, a.scanOpen(s.model.ID)
	languages := learn.Languages{Main: session.MainLanguage(a.prefs, i18n.Locale()), Others: a.prefs.General.OtherLanguages}
	go func() {
		defer cancel()
		found, err := scanner.Scan(ctx, modelID, open, languages, func(read, total, found int) {
			a.post(func() { a.dv.scan.read, a.dv.scan.total, a.dv.scan.found = read, total, found })
		})
		stopped := ctx.Err() != nil
		a.post(func() {
			s := &a.dv.scan
			s.running, s.stopping, s.ended, s.stopped, s.found = false, false, true, stopped, found
			if err != nil && !stopped {
				s.failed, _ = uitext.ModelErrorText(err)
			}
			if !s.open {
				a.toast(scanFoundText(found))
			}
			a.loadSuggestions()
		})
	}()
}

func (a *app) scanDialog(c *ui.Context) {
	s := &a.dv.scan
	if !s.open {
		return
	}
	dialogFrame(c, "scan-dlg", &s.open, i18n.T("Scan history"), "", 460, func() {
		switch {
		case s.running:
			ui.Text(c, i18n.N(s.found, "Read %d/%d · %d suggestion", "Read %d/%d · %d suggestions", s.read, s.total, s.found))
		case s.ended:
			result := scanFoundText(s.found)
			if s.stopped {
				result = i18n.N(s.found, "Stopped, found %d suggestion", "Stopped, found %d suggestions", s.found)
			}
			ui.Text(c, result)
			if s.failed != "" {
				warnLine(c, s.failed)
			}
		case s.failed != "":
			warnLine(c, s.failed)
		case s.counted && s.count == 0:
			ui.Text(c, i18n.T("No new Transcribe results to scan."))
		case s.counted:
			ui.Text(c, i18n.N(s.count, "Voxt will send %d Transcribe result to %s to find names, products and terms. Nothing goes in the dictionary until you confirm it.",
				"Voxt will send %d Transcribe results to %s to find names, products and terms. Nothing goes in the dictionary until you confirm it.", s.count, s.model.Name))
		}
	}, func() {
		ui.Spacer(c)
		switch {
		case s.running:
			if primaryButton(c.Key("scan-stop"), i18n.T("Stop")).Disabled(s.stopping).Clicked() {
				s.stopping = true
				s.cancel()
			}
		case s.ended:
			if primaryButton(c.Key("scan-done"), i18n.T("Done")).Clicked() {
				s.open = false
			}
		default:
			if ui.Button(c.Key("scan-cancel"), i18n.T("Cancel")).Clicked() {
				s.open = false
			}
			if primaryButton(c.Key("scan-start"), i18n.T("Start")).Disabled(s.count == 0).Clicked() {
				a.startScan()
			}
		}
	})
}
