package main

import (
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meetdetect"
)

// Menu items of the card that offers to record a detected meeting.
const (
	meetingAskSnooze   = "snooze"
	meetingAskAuto     = "auto"
	meetingAskSettings = "settings"
)

// meetingAskViewState is the card shown when a meeting is detected: what
// was found, the button that records it, and the menu beside the button.
// Compact is the card after a while without the pointer: one line, no
// button.
type meetingAskViewState struct {
	Visible    bool                 `json:"visible"`
	Title      string               `json:"title"`
	App        string               `json:"app"`
	StartLabel string               `json:"start_label"`
	MenuLabel  string               `json:"menu_label"`
	CloseLabel string               `json:"close_label"`
	Compact    bool                 `json:"compact"`
	End        bool                 `json:"end,omitempty"`
	MenuOpen   bool                 `json:"menu_open"`
	Menu       []meetingAskMenuItem `json:"menu"`
}

type meetingAskMenuItem struct {
	ID    string `json:"id"`
	Label string `json:"label"`
	// Rule draws a line under the item: what follows changes a setting.
	Rule bool `json:"rule,omitempty"`
}

// meetingAskInput is what the card shows: the meeting app (a meetdetect
// name, "" for no card unless End) and the card's own state.
type meetingAskInput struct {
	App      string
	Compact  bool
	MenuOpen bool
	// End: the card asks whether the meeting Voxt records is over.
	End bool
}

func buildMeetingAskViewState(translator *i18n.Translator, input meetingAskInput) meetingAskViewState {
	if input.End {
		// One answer, no menu: End recording, or close the card.
		return meetingAskViewState{
			Visible: true, Title: translator.T("Meeting ended?"), App: translator.T("No sound for 5 minutes"),
			StartLabel: translator.T("End recording"), CloseLabel: translator.T("Dismiss"),
			Compact: input.Compact, End: true, Menu: []meetingAskMenuItem{},
		}
	}
	if input.App == "" {
		return meetingAskViewState{Menu: []meetingAskMenuItem{}}
	}
	return meetingAskViewState{
		Visible:    true,
		Title:      translator.T("Meeting detected"),
		App:        meetingAppNameFor(translator, input.App),
		StartLabel: translator.T("Start recording"),
		MenuLabel:  translator.T("More options"),
		CloseLabel: translator.T("Dismiss"),
		Compact:    input.Compact,
		MenuOpen:   input.MenuOpen && !input.Compact,
		Menu: []meetingAskMenuItem{
			{ID: meetingAskSnooze, Label: translator.T("Ask again in 2 minutes"), Rule: true},
			{ID: meetingAskAuto, Label: translator.T("Start recording automatically")},
			{ID: meetingAskSettings, Label: translator.T("Detection settings")},
		},
	}
}

// meetingAppNameFor is a meeting app's name as the user knows it.
func meetingAppNameFor(translator *i18n.Translator, app string) string {
	switch app {
	case meetdetect.Zoom:
		return "Zoom"
	case meetdetect.Teams:
		return "Microsoft Teams"
	case meetdetect.Tencent:
		return translator.T("Tencent Meeting")
	case meetdetect.Feishu:
		return translator.T("Lark")
	case meetdetect.DingTalk:
		return translator.T("DingTalk")
	case meetdetect.WeCom:
		return translator.T("WeCom")
	case meetdetect.Slack:
		return "Slack"
	case meetdetect.Discord:
		return "Discord"
	case meetdetect.Webex:
		return "Webex"
	case meetdetect.Meet:
		return "Google Meet"
	}
	return app
}
