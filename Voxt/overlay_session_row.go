package main

// The card's session row (#4005): one chip per session, its number and
// title; the current one stands out. A click switches, "×" deletes, "+"
// adds one, a double click renames. It has its own line under the
// header in every state of the card, and in the Rewrite card.

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
	Current  bool   `json:"current,omitempty"`
	Renaming bool   `json:"renaming,omitempty"`
}

const sessionChipH, sessionChipNameW = 22, 140

func sessionCardSessionsHeight(card *sessionCardViewState) int {
	if card != nil && len(card.Sessions) > 0 {
		return layout.CardSessionsH
	}
	return 0
}

// sessionRow draws the session row, top under what is over it.
func sessionRow(context *ui.Context, chips []sessionChipViewState, top float32, actions sessionOverlayActions, local *answerCardUI) {
	if len(chips) == 0 {
		return
	}
	ui.ScrollHorizontal(context.Key("ov-sessions")).Height(sessionChipH).Shrink(0).Margin(top, 0, 0, 0).Label(i18n.T("Sessions")).Children(func() {
		ui.Row(context).Gap(6).AlignItems(ui.Center).Children(func() {
			for index, chip := range chips {
				sessionChip(context.Key(chip.ID), index+1, chip, actions, local)
			}
			if sessionChipButton(context.Key("ov-session-new"), i18n.T("New session"), icons.Add) {
				actions.newSession()
			}
		})
	})
}

func sessionChip(context *ui.Context, number int, chip sessionChipViewState, actions sessionOverlayActions, local *answerCardUI) {
	if chip.Renaming {
		input := ui.TextInputBase(context.Key("name"), &local.sessionName).Label(i18n.T("Session title")).Placeholder(i18n.T("Session title")).
			Width(sessionChipNameW).Height(sessionChipH).Padding(0, 8).Radius(sessionChipH / 2).FontSize(11).
			TextColor(ui.RGBA(255, 255, 255, 0.94)).Background(ui.RGBA(255, 255, 255, 0.16)).Shrink(0).AutoFocus()
		if input.Submitted() {
			actions.nameSession(chip.ID, local.sessionName)
		}
		return
	}
	label := strconv.Itoa(number)
	if chip.Title != "" {
		label += " " + chip.Title
	}
	tip := i18n.T("Switch to session %d. Double-click to rename.", number)
	background, line, text := ui.RGBA(255, 255, 255, 0.06), ui.RGBA(255, 255, 255, 0.1), ovText2
	if chip.Current {
		tip = i18n.T("Session %d, the current one. Double-click to rename.", number)
		background, line, text = ui.RGBA(255, 255, 255, 0.2), ui.RGBA(255, 255, 255, 0.28), ui.RGBA(255, 255, 255, 0.94)
	}
	ui.Row(context).Height(sessionChipH).Radius(sessionChipH/2).Background(background).Border(1, line).AlignItems(ui.Center).Shrink(0).Children(func() {
		button := ui.ButtonBase(context.Key("go")).Height(sessionChipH).Padding(0, 4, 0, 9).Center().Label(i18n.T("Session %d", number)).Tooltip(tip).Checked(chip.Current)
		if chip.Current {
			button.ScrollIntoView()
		}
		button.Children(func() {
			ui.Text(context, label).FontSize(11).FontWeight(600).TextColor(text).SingleLine()
		})
		switch {
		case button.DoubleClicked():
			local.sessionName = chip.Title
			actions.renameSession(chip.ID)
		case button.Clicked() && !chip.Current:
			actions.switchSession(chip.ID)
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
