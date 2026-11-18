package main

// The shell's fixture-demo parts: the local state's reset and the
// recorded actions (meeting_demo.go has the host).

// shellDemoState is the shell's part of the demo host: the shell views'
// local state and its own values.
type shellDemoState struct {
	navPath          string
	noticesOpen      bool
	narrow           bool
	navUI            navLocalUI
	logsSel          logSelection
	logsFind         finder
	logsQuery        string
	logsSource       string
	logsUI           logsLocalUI
	paletteOpen      bool
	paletteQuery     string
	paletteSel       int
	paletteUI        paletteLocalUI
	sheetOpen        bool
	announcementOpen bool
	shellUI          shellLocalUI
}

// resetShellDemo resets the shell's local state and seeds it from the
// fixture.
func (host *meetingDemoHost) resetShellDemo(fixture meetingDemoFixture) {
	host.navPath, host.noticesOpen, host.narrow = "/home", false, false
	if state := fixture.state.Nav; state != nil && state.Current != "" {
		host.navPath = state.Current
	}
	host.navUI = navLocalUI{nav: &host.navPath, noticesOpen: &host.noticesOpen, narrow: &host.narrow}
	host.logsSel, host.logsFind = logSelection{}, finder{}
	host.logsQuery, host.logsSource = "", ""
	if state := fixture.state.Logs; state != nil {
		host.logsQuery = state.Query
		for _, source := range state.Sources {
			if source.Label == state.Source {
				host.logsSource = source.Code
			}
		}
	}
	host.logsUI = logsLocalUI{sel: &host.logsSel, find: &host.logsFind, query: &host.logsQuery, source: &host.logsSource}
	host.paletteOpen, host.paletteQuery, host.paletteSel = false, "", 0
	host.sheetOpen, host.announcementOpen = false, false
	if state := fixture.state.ShellUI; state != nil {
		host.paletteOpen, host.paletteQuery, host.paletteSel = state.PaletteOpen, state.PaletteQuery, state.PaletteSel
		host.sheetOpen = state.SheetOpen
		host.announcementOpen = state.AnnouncementOpen
	}
	host.paletteUI = paletteLocalUI{open: &host.paletteOpen, query: &host.paletteQuery, sel: &host.paletteSel}
	host.shellUI = shellLocalUI{nav: &host.navUI, logs: &host.logsUI, palette: &host.paletteUI, sheetOpen: &host.sheetOpen}
}

// The shell's actions, recorded.

func (host *meetingDemoHost) setRail(on bool) { host.record("setRail %t", on) }
func (host *meetingDemoHost) openPalette()    { host.record("openPalette") }
func (host *meetingDemoHost) sidebarWidthChanged(width float64) {
	host.record("sidebarWidthChanged %v", width)
}
func (host *meetingDemoHost) dismissNotice(id string)    { host.record("dismissNotice %s", id) }
func (host *meetingDemoHost) clearNotices()              { host.record("clearNotices") }
func (host *meetingDemoHost) noticeCardAction(id string) { host.record("noticeCardAction %s", id) }
func (host *meetingDemoHost) openMainSettings()          { host.record("openMainSettings") }
func (host *meetingDemoHost) copyLogLines(seqs []uint64) { host.record("copyLogLines %v", seqs) }
func (host *meetingDemoHost) openLogFile()               { host.record("openLogFile") }
func (host *meetingDemoHost) closeLogsPanel()            { host.record("closeLogsPanel") }
func (host *meetingDemoHost) runCommandID(id string)     { host.record("runCommandID %s", id) }
func (host *meetingDemoHost) toggleLogsPanel()           { host.record("toggleLogsPanel") }
