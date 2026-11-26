package main

// The card's session row (#4005): one chip per session, its number and
// title (cut short on the others); the current one stands out and has
// "×" to delete it (the only one too: a new empty session takes its
// place). A click switches, "+" adds a session, a double click renames.
// It is a thin line where the card's rule is, and carries the rule
// after "+".

import (
	"strconv"
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
)

// sessionChipViewState is one session on the session row. Its number is
// its place in the row.
type sessionChipViewState struct {
	ID       int    `json:"id"`
	Title    string `json:"title,omitempty"`
	Full     string `json:"full,omitempty"` // the whole title, when Title is cut
	Current  bool   `json:"current,omitempty"`
	Renaming bool   `json:"renaming,omitempty"`
}

// sessionBarH is the room under the chips for the row's scroll bar.
const sessionChipH, sessionChipNameW, sessionBarH = 18, 140, 10

func sessionCardSessionsHeight(card *sessionCardViewState) int {
	if card != nil && len(card.Sessions) > 0 {
		return layout.CardSessionsH
	}
	return 0
}

// sessionRow draws the session row with the card's rule after it, top
// under what is over it. It says whether it drew: with no chips the
// caller draws the rule alone.
func sessionRow(context *ui.Context, chips []sessionChipViewState, top float32, actions sessionOverlayActions, local *answerCardUI) bool {
	if len(chips) == 0 {
		return false
	}
	ui.Row(context).Height(sessionChipH).Shrink(0).Margin(top, 0, 0, 0).Gap(8).AlignItems(ui.Start).Children(func() {
		// Taller than the chips: the scroll bar lies in the gap under
		// them, not over them.
		ui.ScrollHorizontal(context.Key("ov-sessions")).Height(sessionChipH + sessionBarH).AlignItems(ui.Start).Shrink(1).MinWidth(0).Label(i18n.T("Sessions")).Children(func() {
			ui.Row(context).Height(sessionChipH).Gap(4).AlignItems(ui.Center).Children(func() {
				for index, chip := range chips {
					sessionChip(context.Key(chip.ID), index+1, chip, actions, local)
				}
				if sessionChipButton(context.Key("ov-session-new"), i18n.T("New session"), icons.Add) {
					actions.newSession()
				}
			})
		})
		ui.Box(context).Height(1).Margin(sessionChipH/2, 0, 0, 0).Grow(1).MinWidth(0).Background(ovCardRule)
	})
	return true
}

// sessionChip draws one session.
func sessionChip(context *ui.Context, number int, chip sessionChipViewState, actions sessionOverlayActions, local *answerCardUI) {
	title := chip.Title
	if title == "" {
		title = i18n.T("New session")
	}
	if chip.Renaming {
		// The field starts empty over the automatic title, so typing needs
		// no clearing first; nothing typed gives that title back.
		input := ui.TextInputBase(context.Key("name"), &local.sessionName).Label(i18n.T("Session title")).Placeholder(title).
			Width(sessionChipNameW).Height(sessionChipH).Padding(0, 8).Radius(sessionChipH / 2).FontSize(11).
			TextColor(ui.RGBA(255, 255, 255, 0.94)).Background(ui.RGBA(255, 255, 255, 0.16)).Shrink(0).AutoFocus()
		if input.Submitted() {
			actions.nameSession(chip.ID, local.sessionName)
		}
		return
	}
	tip := i18n.T("Switch to session %d. Double-click to rename.", number)
	background, line, text := ui.RGBA(255, 255, 255, 0.06), ui.RGBA(255, 255, 255, 0.1), ovText2
	if chip.Current {
		tip = i18n.T("Session %d, the current one. Double-click to rename.", number)
		background, line, text = ui.RGBA(255, 255, 255, 0.2), ui.RGBA(255, 255, 255, 0.28), ui.RGBA(255, 255, 255, 0.94)
	}
	if chip.Full != "" {
		tip = chip.Full
	}
	// Only the current chip has the delete button: the others stay
	// narrow, and no chip changes its width under the pointer.
	removable := chip.Current
	row := ui.Row(context).Height(sessionChipH).Radius(sessionChipH/2).Background(background).Border(1, line).AlignItems(ui.Center).Shrink(0)
	if chip.Current && local.sessionShown != chip.ID {
		// Once, when the session becomes the current one: in every
		// frame it would keep the row from scrolling away from it.
		local.sessionShown = chip.ID
		row.ScrollIntoView()
	}
	row.Children(func() {
		right := float32(2)
		if !removable {
			right = 7
		}
		button := ui.ButtonBase(context.Key("go")).Height(sessionChipH).Padding(0, right, 0, 7).Center().Label(i18n.T("Session %d", number)).Tooltip(tip).Checked(chip.Current)
		button.Children(func() {
			ui.Text(context, strconv.Itoa(number)+" "+title).FontSize(11).FontWeight(600).TextColor(text).SingleLine()
		})
		switch {
		case button.DoubleClicked():
			local.sessionName = ""
			actions.renameSession(chip.ID)
		case button.Clicked() && !chip.Current:
			actions.switchSession(chip.ID)
		}
		if !removable {
			return
		}
		remove := ui.ButtonBase(context.Key("x")).Size(sessionChipH-4, sessionChipH-4).Margin(0, 2, 0, 0).Radius((sessionChipH - 4) / 2).Center().
			Label(i18n.T("Delete session %d", number)).Tooltip(i18n.T("Delete session %d", number))
		if remove.Hovered() {
			remove.Background(ui.RGBA(255, 255, 255, 0.16))
		}
		if remove.Children(func() {
			ui.Icon(context, icons.Close.Linear).Size(9, 9).TextColor(text)
		}).Clicked() {
			actions.deleteSession(chip.ID)
		}
	})
}

func sessionChipButton(context *ui.Context, label string, icon icons.Icon) bool {
	button := ui.ButtonBase(context).Size(sessionChipH, sessionChipH).Radius(sessionChipH / 2).Center().Shrink(0).Label(label).Tooltip(label)
	background := ui.RGBA(255, 255, 255, 0.06)
	if button.Hovered() {
		background = ui.RGBA(255, 255, 255, 0.16)
	}
	return button.Background(background).Children(func() {
		ui.Icon(context, icon.Linear).Size(12, 12).TextColor(ovText2)
	}).Clicked()
}

// sessionMarked is the pill's scene text with the session's number
// before it ("#2 · Mail"); number 0: the scene alone.
func sessionMarked(number int, scene string) string {
	switch {
	case number == 0:
		return scene
	case scene == "":
		return "#" + strconv.Itoa(number)
	}
	return "#" + strconv.Itoa(number) + " · " + scene
}

func pillSceneLabel(pill *sessionPillViewState) string {
	var parts []string
	if pill.Session > 0 {
		parts = append(parts, i18n.T("Session %d", pill.Session))
	}
	if pill.SceneLabel != "" {
		parts = append(parts, i18n.T("App enhancement: %s", pill.SceneLabel))
	}
	return strings.Join(parts, ", ")
}
