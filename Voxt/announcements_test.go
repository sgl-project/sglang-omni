package main

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
)

// announceServer is Voxt's announcements server in Voxt 1.x's format:
// the manifest and the list of revision, list being its items' JSON, or
// status instead when it is not 200, or no answer for hang. It keeps
// every request (its URI and headers).
type announceServer struct {
	mu       sync.Mutex
	revision string
	list     string
	status   int
	hang     bool
	requests []*http.Request
}

func (s *announceServer) start(t *testing.T) string {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.mu.Lock()
		s.requests = append(s.requests, r)
		revision, list, status, hang := s.revision, s.list, s.status, s.hang
		s.mu.Unlock()
		switch {
		case hang:
			<-r.Context().Done()
		case status != 0:
			w.WriteHeader(status)
		case r.URL.Path == "/api/app/notifications/manifest":
			fmt.Fprintf(w, `{"schemaVersion":1,"revision":%q,"count":2,"latestNotificationID":"n2"}`, revision)
		default:
			fmt.Fprintf(w, `{"revision":%q,"notifications":[%s]}`, revision, list)
		}
	}))
	t.Cleanup(srv.Close)
	return srv.URL + "/api/app/notifications"
}

func (s *announceServer) set(f func(s *announceServer)) {
	s.mu.Lock()
	defer s.mu.Unlock()
	f(s)
}

func (s *announceServer) count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.requests)
}

const (
	oldAnnouncement = `{"id":"n1","title":"Welcome to Voxt","content":"Thanks for trying Voxt.","coverImage":null,"version":"2.0.0","publishedAt":"2026-11-01T08:00:00Z"}`
	newAnnouncement = `{"id":"n2","title":"Voxt 2.0.3 is out","content":"Meetings are coming.\nThe second line tells more, and the dialog shows it all.","coverImage":null,"version":"2.0.3","publishedAt":"2026-11-08T09:30:00.250Z"}`
)

// TestAnnouncements: as Voxt 1.x read its announcements. The newest is a
// card in the sidebar, whose Read shows it whole, with its date and the
// version it is about; dismissed, it does not
// come back, neither at the next check nor after a restart, and the older
// one is never shown. The requests carry nothing of the user's: the
// manifest's address, then the list's with the revision only, and no
// header but Go's own.
func TestAnnouncements(t *testing.T) {
	srv := &announceServer{revision: "r1", list: oldAnnouncement + "," + newAnnouncement}
	url := srv.start(t)
	dir := t.TempDir()
	a, _, tt := newTestAppIn(t, dir)
	a.announceURL = url
	a.checkAnnouncements()
	if !has(tt, "Voxt 2.0.3 is out") || !has(tt, "Meetings are coming.\nThe second line tells more, and the dialog shows it all.") || has(tt, "Welcome to Voxt") {
		t.Fatalf("no card for the newest, or one for the older: %q", tt.Texts())
	}
	click(t, tt, "Read")
	if !a.announceShownOpen || !has(tt, "Done") || !hasSuffix(tt, " · Version 2.0.3") {
		t.Fatalf("Read: %q", tt.Texts())
	}
	click(t, tt, "Done")
	for i, r := range srv.requests {
		want := []string{"/api/app/notifications/manifest", "/api/app/notifications?revision=r1"}[min(i, 1)]
		var headers []string
		for k := range r.Header {
			headers = append(headers, k)
		}
		slices.Sort(headers)
		if r.Method != http.MethodGet || r.URL.RequestURI() != want || !slices.Equal(headers, []string{"Accept-Encoding", "User-Agent"}) ||
			!strings.HasPrefix(r.UserAgent(), "Go-http-client/") || i > 1 {
			t.Errorf("request %d: %s %s %v", i, r.Method, r.URL.RequestURI(), r.Header)
		}
	}

	// Within 10 minutes no request; then the same revision reads only
	// the manifest and shows nothing again once the card is dismissed.
	before := srv.count()
	a.checkAnnouncements()
	if srv.count() != before {
		t.Errorf("checked again within 10 minutes")
	}
	if err := tt.ClickControl("Dismiss"); err != nil {
		t.Fatal(err)
	}
	frames(tt)
	a.announceAt = a.announceAt.Add(-announceEvery)
	a.checkAnnouncements()
	if srv.count() != before+1 || has(tt, "Voxt 2.0.3 is out") {
		t.Errorf("after dismissing: %d requests, %q", srv.count()-before, tt.Texts())
	}

	// A new revision of the same list, after a restart: nothing comes
	// back. A new announcement in it shows.
	a.quit()
	a.lib.close()
	srv.set(func(s *announceServer) { s.revision = "r2" })
	b, _, bt := newTestAppIn(t, dir)
	b.announceURL = url
	b.checkAnnouncements()
	if has(bt, "Voxt 2.0.3 is out") || has(bt, "Welcome to Voxt") {
		t.Errorf("seen ones again: %q", bt.Texts())
	}
	srv.set(func(s *announceServer) {
		s.revision = "r3"
		s.list = strings.Replace(newAnnouncement, `"id":"n2","title":"Voxt 2.0.3 is out"`, `"id":"n3","title":"Voxt 2.0.4 is out"`, 1) + "," + oldAnnouncement
	})
	b.announceAt = time.Time{}
	b.checkAnnouncements()
	if !has(bt, "Voxt 2.0.4 is out") {
		t.Errorf("no card for the new one: %q", bt.Texts())
	}
}

// TestAnnouncementsFail: a server error, a broken answer or no answer
// shows nothing, no card, toast or error, and the next check tries again.
func TestAnnouncementsFail(t *testing.T) {
	old := announceTimeout
	announceTimeout = 200 * time.Millisecond
	t.Cleanup(func() { announceTimeout = old })
	srv := &announceServer{revision: "r1", list: newAnnouncement}
	url := srv.start(t)
	a, _, tt := newTestApp(t)
	a.announceURL = url
	notices := len(a.notices)
	for _, fail := range []func(s *announceServer){
		func(s *announceServer) { s.status = http.StatusServiceUnavailable },
		func(s *announceServer) { s.status, s.list = 0, `{"id":` },
		func(s *announceServer) { s.list, s.hang = newAnnouncement, true },
	} {
		srv.set(fail)
		a.announceAt = time.Time{}
		a.checkAnnouncements()
		frames(tt)
		if len(a.notices) != notices || len(a.toasts) != 0 || has(tt, "Voxt 2.0.3 is out") || a.announceBusy {
			t.Errorf("a failure showed something: %d notices, toasts %q, busy %v", len(a.notices), a.toasts, a.announceBusy)
		}
	}
	srv.set(func(s *announceServer) { s.hang = false })
	a.announceAt = time.Time{}
	a.checkAnnouncements()
	if !has(tt, "Voxt 2.0.3 is out") {
		t.Errorf("not shown once the server answers: %q", tt.Texts())
	}
}

// TestAnnouncementsFor: the card is the newest announcement for Voxt 2 in
// the UI's language. Voxt 1.x's server still lists its 1.13 announcement
// of meetings and notes, in Chinese, which Voxt 2 showed to every user,
// in the English UI too.
func TestAnnouncementsFor(t *testing.T) {
	v1 := `{"id":"v1","title":"会议 & 笔记","content":"- 会议功能经过重构回来了，支持 会议、字幕、录音 三种模式","coverImage":null,"version":"1.13.0-beta.3","publishedAt":"2026-11-09T13:09:00Z"}`
	srv := &announceServer{revision: "r1", list: v1 + "," + oldAnnouncement}
	url := srv.start(t)
	a, _, tt := newTestApp(t)
	a.announceURL = url
	a.checkAnnouncements()
	if has(tt, "会议 & 笔记") || !has(tt, "Welcome to Voxt") {
		t.Errorf("want the 2.0.0 card, not the 1.x one: %q", tt.Texts())
	}
}
