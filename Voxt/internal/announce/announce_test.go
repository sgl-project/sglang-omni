package announce

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"
)

// server answers the manifest and the list in Voxt 1.x's format, and
// keeps the requests' URIs.
func server(t *testing.T, manifest, list string) (*httptest.Server, *[]string) {
	var got []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = append(got, r.URL.RequestURI())
		switch r.URL.Path {
		case "/api/app/notifications/manifest":
			w.Write([]byte(manifest))
		case "/api/app/notifications":
			w.Write([]byte(list))
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	return srv, &got
}

// TestFetch: the list comes newest first, read only when the manifest's
// revision is not the one read last; the server's dates may have
// fractions of seconds, as Voxt 1.x's decoder took.
func TestFetch(t *testing.T) {
	srv, got := server(t, `{"schemaVersion":1,"revision":"r 2","count":2,"latestNotificationID":"b"}`,
		`{"revision":"r 2","notifications":[
		 {"id":"a","title":"Old","content":"First","coverImage":null,"version":"1.0.0","publishedAt":"2026-07-12T08:00:00Z"},
		 {"id":"b","title":"New","content":"Second","coverImage":"https://example.com/b.png","version":"2.0.3","publishedAt":"2026-11-08T09:30:00.250Z"}]}`)
	base := srv.URL + "/api/app/notifications"
	rev, list, err := Fetch(context.Background(), srv.Client(), base, "")
	if err != nil || rev != "r 2" || len(list) != 2 || list[0].ID != "b" || list[0].Content != "Second" || list[0].Version != "2.0.3" || list[1].ID != "a" {
		t.Fatalf("%q %+v %v", rev, list, err)
	}
	rev, list, err = Fetch(context.Background(), srv.Client(), base, rev)
	if err != nil || rev != "r 2" || list != nil {
		t.Fatalf("same revision: %q %+v %v", rev, list, err)
	}
	if want := []string{"/api/app/notifications/manifest", "/api/app/notifications?revision=r+2", "/api/app/notifications/manifest"}; !slices.Equal(*got, want) {
		t.Errorf("requests %q", *got)
	}
}

// TestFetchFails: a manifest of another schema, a list of another
// revision than the manifest's, or an error status keep the revision
// read last and give no list.
func TestFetchFails(t *testing.T) {
	for _, c := range []struct{ name, manifest, list string }{
		{"schema", `{"schemaVersion":2,"revision":"r2","count":1}`, `{"revision":"r2","notifications":[]}`},
		{"revision", `{"schemaVersion":1,"revision":"r2","count":1}`, `{"revision":"r3","notifications":[]}`},
		{"status", `not found`, ``},
	} {
		srv, _ := server(t, c.manifest, c.list)
		base := srv.URL + "/api/app/notifications"
		if c.name == "status" {
			base = srv.URL + "/missing"
		}
		rev, list, err := Fetch(context.Background(), srv.Client(), base, "r1")
		if err == nil || rev != "r1" || list != nil {
			t.Errorf("%s: %q %+v %v", c.name, rev, list, err)
		}
		if c.name == "revision" && !errors.Is(err, ErrChanged) {
			t.Errorf("revision: %v", err)
		}
	}
}
