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

// navActions are the intents of the sidebar and the icon rail
// (shell_nav.go).
type navActions interface {
	// navigate goes to a page (a sidebar or rail item).
	navigate(path string)
	// setRail is the user's choice of the layout: icons only.
	setRail(on bool)
	// openPalette opens the command palette (the search field).
	openPalette()
	// sidebarWidthChanged is a resize of the sidebar, once the drag
	// rested: the settings keep the width.
	sidebarWidthChanged(width float64)
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
