package main

// The announcement dialog (shell_announcement_controller.go reads them): a
// card's What's new, or its row in Settings → About → Updates, shows the
// announcement whole, under its date and the version it is about.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

// announcementDialog shows an announcement whole; state is nil when the
// dialog is closed. open binds the dialog's visibility.
func announcementDialog(c *ui.Context, state *announcementViewState, open *bool) {
	if state == nil {
		return
	}
	dialogFrame(c, "announcement", open, state.Title, state.Sub, 520, func() {
		ui.Text(c, state.Body).Selectable().Grow(1)
	}, func() {
		ui.Spacer(c)
		if primaryButton(c.Key("announcement-done"), i18n.T("Done")).Clicked() {
			*open = false
		}
	})
}
