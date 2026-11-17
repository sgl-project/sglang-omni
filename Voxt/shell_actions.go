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

// logsActions are the intents of the logs page and panel (shell_logs.go).
type logsActions interface {
	// copyLogLines copies the shown lines with these seqs (in the log's
	// order), says how many, and clears the selection.
	copyLogLines(seqs []uint64)
	// openLogFile opens the log file in the Finder.
	openLogFile()
	// closeLogsPanel closes the panel under the page.
	closeLogsPanel()
}
