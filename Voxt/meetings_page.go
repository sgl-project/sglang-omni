package main

// The Meetings page (/meetings): past meetings newest first (dev builds
// only until C1-10). While none is recording, Start begins one; each row
// can be deleted or shown in Finder when audio is still on disk.

import (
	"context"
	"os"
	"path/filepath"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

type meetingsView struct {
	list    []store.Meeting
	read    bool
	loading bool
	err     string
	seq     int
	listUI  ui.ListState
	detail  *meetingDetail
}

type meetingsState struct {
	mv meetingsView
}

func (a *app) loadMeetings() {
	n := &a.mv
	n.seq++
	n.loading = true
	seq := n.seq
	repo := a.lib.store.Meetings()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		list, err := repo.ListMeetings(ctx, store.MeetingQuery{})
		a.post(func() {
			n := &a.mv
			if seq != n.seq {
				return
			}
			n.loading, n.read = false, true
			if err != nil {
				n.err = err.Error()
				return
			}
			n.list, n.err = list, ""
		})
	})
}

func (a *app) meetingsPage(c *ui.Context) {
	if !a.meetingEnabled() {
		a.notFoundPage(c, "/meetings")
		return
	}
	t := c.Theme()
	n := &a.mv
	if !n.read && !n.loading {
		a.loadMeetings()
	}
	pageHeader(c, i18n.T("Meetings"), func() {
		if start := primaryButton(c.Key("mt-start"), i18n.T("Start meeting")).Disabled(a.meetingActive()); start.Clicked() {
			a.startMeeting()
		}
	})
	ui.Scroll(c.Key("meetings-scroll")).Grow(1).MinHeight(0).Children(func() {
		ui.Column(c).Padding(px(t, sp1), px(t, sp6), px(t, sp2)).Gap(px(t, sp2)).Children(func() {
			if n.err != "" {
				ui.Text(c, n.err)
				return
			}
			if n.loading {
				ui.Text(c, i18n.T("Loading…"))
				return
			}
			if len(n.list) == 0 {
				ui.Text(c, i18n.T("No meetings yet.")).TextColor(tok(t).Text2)
				return
			}
			list := n.list
			ui.List(c.Key("meetings-list"), &n.listUI, len(list)).Grow(1).MinHeight(0).Gap(px(t, 2)).Children(func() {
				for _, mt := range list {
					mt := mt
					ui.Row(c.Key("mt-"+mt.ID)).Padding(px(t, sp2), 0).AlignItems(ui.Center).Gap(px(t, sp2)).Children(func() {
						title := mt.CreatedAt.Format("2006-01-02 15:04")
						if mt.Title != "" {
							title = mt.Title
						}
						if ghostButton(c.Key("mt-open-"+mt.ID), i18n.T("Open")).Clicked() {
							a.openMeetingDetail(mt.ID)
						}
						ui.Column(c).Grow(1).Gap(px(t, 4)).Children(func() {
							ui.Text(c, title).FontWeight(600)
							line := string(mt.Status)
							if mt.DurationMS > 0 {
								line += " · " + (time.Duration(mt.DurationMS) * time.Millisecond).Round(time.Second).String()
							}
							ui.Text(c, line).TextColor(tok(t).Text2)
						})
						switch mt.Status {
						case store.MeetingInterrupted, store.MeetingFinishing:
							if primaryButton(c.Key("mt-continue-"+mt.ID), i18n.T("Continue final pass")).Clicked() {
								a.meetingReprocess(mt.ID)
							}
						case store.MeetingDone:
							if ghostButton(c.Key("mt-reprocess-"+mt.ID), i18n.T("Reprocess transcript")).Clicked() {
								a.meetingReprocess(mt.ID)
							}
						}
						if mt.Audio != "" {
							if ghostButton(c.Key("mt-show-"+mt.ID), i18n.T("Show in Finder")).Clicked() {
								dir := filepath.Join(a.dataDir(), "meetings", mt.Audio)
								revealInFinder(dir)
							}
						}
						if ghostButton(c.Key("mt-del-"+mt.ID), i18n.T("Delete")).Clicked() {
							id, audio := mt.ID, mt.Audio
							a.async(func() {
								_ = a.lib.store.Meetings().DeleteMeeting(context.Background(), id)
								if audio != "" {
									_ = os.RemoveAll(filepath.Join(a.dataDir(), "meetings", audio))
								}
								a.post(a.loadMeetings)
							})
						}
					})
				}
			})
			a.meetingDetailSection(c)
			ui.Box(c).Height(px(t, sp3))
			a.meetingSettings(c)
		})
	})
}
