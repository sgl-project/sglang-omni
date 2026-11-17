package main

// The shell's actions: the user intents its views can trigger. The app
// implements them (the shell's controllers); the fixture demo records
// them.

// noticeActions are the intents of the notification cards
// (shell_notices.go).
type noticeActions interface {
	// dismissNotice removes one card (its ×).
	dismissNotice(id string)
	// clearNotices removes every card.
	clearNotices()
	// noticeCardAction runs a card's action (its link).
	noticeCardAction(id string)
	// openMainSettings opens the Settings dialog (the rail popover's row).
	openMainSettings()
}
