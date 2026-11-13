// Package announce reads Voxt's announcements, as Voxt 1.x did
// (VoxtNotificationStore): the manifest at <base>/manifest names the
// list's revision, and the list (<base>?revision=<revision>) is read only
// when that changed. The requests carry nothing but that revision: no
// version, language, identifier or anything else of the user's.
//
// Imports: nothing internal.
package announce

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"time"
	"unicode"
)

// URL is where the announcements are (Voxt 1.x's address).
const URL = "https://voxt.actnow.dev/api/app/notifications"

// Timeout bounds a check, both requests (Voxt 1.x waited 12 s for each).
const Timeout = 12 * time.Second

// maxBody bounds an answer: the list is a few short texts.
const maxBody = 1 << 20

// Announcement is one, as the server sends it (its cover image unused).
type Announcement struct {
	ID          string    `json:"id"`
	Title       string    `json:"title"`
	Content     string    `json:"content"`
	Version     string    `json:"version"`
	PublishedAt time.Time `json:"publishedAt"`
}

// ErrChanged is a list of another revision than the manifest's: it
// changed between the two requests, the next check reads it.
var ErrChanged = errors.New("the list changed while it was read")

// Fetch reads the announcements of base, newest first, unless revision
// (the one read last, "" none) is still the server's: then it returns
// that revision and no list.
func Fetch(ctx context.Context, client *http.Client, base, revision string) (string, []Announcement, error) {
	var manifest struct {
		SchemaVersion int    `json:"schemaVersion"`
		Revision      string `json:"revision"`
	}
	if err := get(ctx, client, base+"/manifest", &manifest); err != nil {
		return revision, nil, err
	}
	if manifest.SchemaVersion != 1 {
		return revision, nil, fmt.Errorf("manifest schema %d, want 1", manifest.SchemaVersion)
	}
	if manifest.Revision == revision {
		return revision, nil, nil
	}
	var list struct {
		Revision      string         `json:"revision"`
		Notifications []Announcement `json:"notifications"`
	}
	if err := get(ctx, client, base+"?revision="+url.QueryEscape(manifest.Revision), &list); err != nil {
		return revision, nil, err
	}
	if list.Revision != manifest.Revision {
		return revision, nil, ErrChanged
	}
	out := slices.DeleteFunc(list.Notifications, func(n Announcement) bool { return n.ID == "" || n.Title == "" })
	slices.SortStableFunc(out, func(x, y Announcement) int { return y.PublishedAt.Compare(x.PublishedAt) })
	return list.Revision, out, nil
}

// get decodes the JSON answer of a GET of address into v.
func get(ctx context.Context, client *http.Client, address string, v any) error {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, address, nil)
	if err != nil {
		return err
	}
	response, err := client.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode/100 != 2 {
		return fmt.Errorf("%s: HTTP %d", address, response.StatusCode)
	}
	if err := json.NewDecoder(io.LimitReader(response.Body, maxBody)).Decode(v); err != nil {
		return fmt.Errorf("%s: %w", address, err)
	}
	return nil
}

// For keeps the announcements a Voxt 2 user can use and read. Voxt 1.x's
// (a version below 2) go: the server is 1.x's, and they are about 1.x,
// such as its meetings and notes. Unless chinese, so do the ones written
// mostly in Chinese: the server says nothing of an announcement's
// language. A version that does not parse keeps it.
func For(list []Announcement, chinese bool) []Announcement {
	return slices.DeleteFunc(slices.Clone(list), func(n Announcement) bool {
		major, _, _ := strings.Cut(strings.TrimPrefix(strings.TrimSpace(n.Version), "v"), ".")
		if m, err := strconv.Atoi(major); err == nil && m < 2 {
			return true
		}
		han, words, inWord := 0, 0, false // a Han character against a Latin word
		for _, r := range n.Title + "\n" + n.Content {
			latin := r <= unicode.MaxLatin1 && unicode.IsLetter(r)
			if latin && !inWord {
				words++
			}
			if unicode.Is(unicode.Han, r) {
				han++
			}
			inWord = latin
		}
		return !chinese && han > words
	})
}
