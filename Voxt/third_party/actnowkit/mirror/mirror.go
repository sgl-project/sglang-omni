// Package mirror downloads the files of GitHub releases from github.com
// or from the family's mirror, https://mirror.actnow.dev/github, which
// answers GitHub's paths for the releases the apps need (spec §9.6):
//
//	/<owner>/<repo>/releases/latest                    GitHub: a redirect to …/releases/tag/<tag>;
//	                                                   the mirror: JSON {"tag": "<tag>"}
//	/<owner>/<repo>/releases/download/<tag>/<asset>    the file
//
// GitHub's release files reach many users in China at a few KB/s; the
// mirror (Cloudflare R2) at a few MB/s. Neither is trusted: the caller
// checks what it gets as it always did (a signature, a pinned SHA-256),
// and a file that fails its check is fetched from the next source.
//
// Which sources: Sources maps the family's download source setting
// (network.hfEndpoints): automatic → both, raced; "official" → github.com
// only; "mirror" → the mirror only.
//
//	c := &mirror.Client{Sources: mirror.Sources(hfEndpoints), HTTP: client}
//	tag, _, err := c.Latest(ctx, "yt-dlp/yt-dlp", validTag)
//	file, from, err := c.File(ctx, "/yt-dlp/yt-dlp/releases/download/"+tag+"/yt-dlp_macos.zip",
//		mirror.FileOptions{Dir: dir, SHA256: sum})
package mirror

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"slices"
	"strings"
	"time"

	"github.com/hehehai/actnowkit/settings"
)

// The sources.
const (
	// GitHub is github.com.
	GitHub = "https://github.com"
	// ActNow is the family's mirror of the releases the apps download.
	ActNow = "https://mirror.actnow.dev/github"
	// EnvSources, when set, replaces the sources: base URLs separated by
	// spaces (development and tests).
	EnvSources = "ACTNOW_GITHUB_SOURCES"
)

// Sources are the sources for a value of network.hfEndpoints (spec
// §9.6): "official" → github.com; "mirror" → the ActNow mirror;
// anything else ("", "auto", URLs of Hugging Face mirrors) → both.
// $ACTNOW_GITHUB_SOURCES replaces them.
func Sources(hfEndpoints string) []string {
	if env := strings.Fields(os.Getenv(EnvSources)); len(env) > 0 {
		return clean(env)
	}
	switch strings.TrimSpace(hfEndpoints) {
	case settings.EndpointsOfficial:
		return []string{GitHub}
	case settings.EndpointsMirror:
		return []string{ActNow}
	}
	return []string{GitHub, ActNow}
}

// Hosts are the origins of the sources (for a connection test).
func Hosts(sources []string) []string {
	var out []string
	for _, s := range sources {
		h := s
		if i := strings.Index(s, "://"); i >= 0 {
			if j := strings.IndexByte(s[i+3:], '/'); j >= 0 {
				h = s[:i+3+j]
			}
		}
		if !slices.Contains(out, h) {
			out = append(out, h)
		}
	}
	return out
}

func clean(in []string) []string {
	var out []string
	for _, s := range in {
		if s = strings.TrimRight(strings.TrimSpace(s), "/"); s != "" && !slices.Contains(out, s) {
			out = append(out, s)
		}
	}
	return out
}

// Defaults of the Client's tuning.
const (
	DefaultLead       = 256 << 10
	DefaultStall      = 20 * time.Second
	DefaultMinRate    = 32 << 10
	DefaultRateWindow = 30 * time.Second
)

// Client downloads from the sources. The zero Client uses both sources
// and http.DefaultClient. Safe for concurrent use.
type Client struct {
	// Sources are the base URLs, in order of preference when they tie
	// (nil: Sources("")).
	Sources []string
	// HTTP is the client requests go through, asked for each download so
	// that a changed proxy applies (nil, or returning nil:
	// http.DefaultClient).
	HTTP func() *http.Client
	// Logf, when set, is told which source served what and why one was
	// left.
	Logf func(format string, args ...any)

	// Lead is how many bytes of a file decide the race between sources
	// (0: DefaultLead).
	Lead int64
	// Stall: a source that sends nothing for this long is left for the
	// next one (0: DefaultStall).
	Stall time.Duration
	// MinRate: a source slower than this many bytes per second over
	// RateWindow is left for the next one (0: DefaultMinRate). Neither
	// applies to the last source left.
	MinRate    int64
	RateWindow time.Duration
}

func (c *Client) sources() []string {
	if len(c.Sources) > 0 {
		return clean(c.Sources)
	}
	return Sources("")
}

func (c *Client) client() *http.Client {
	if c.HTTP != nil {
		if h := c.HTTP(); h != nil {
			return h
		}
	}
	return http.DefaultClient
}

func (c *Client) logf(format string, args ...any) {
	if c.Logf != nil {
		c.Logf(format, args...)
	}
}

func or[T comparable](v, def T) T {
	var zero T
	if v == zero {
		return def
	}
	return v
}

// StatusError is a source's answer other than 200 (or a redirect where
// one is expected).
type StatusError struct {
	URL    string
	Status string
	Code   int
}

func (e *StatusError) Error() string { return fmt.Sprintf("%s: %s", e.URL, e.Status) }

// Try runs fn for every source at once and returns the first result
// without an error and its source; the other calls are cancelled. With
// one source fn runs alone. When every call fails the errors are joined,
// each prefixed by its source. Use it for what is checked as a whole
// from one source (a checksum file and its signature).
func Try[T any](ctx context.Context, c *Client, fn func(ctx context.Context, base string) (T, error)) (T, string, error) {
	srcs := c.sources()
	var zero T
	if len(srcs) == 1 {
		v, err := fn(ctx, srcs[0])
		if err != nil {
			return zero, "", fmt.Errorf("%s: %w", srcs[0], err)
		}
		return v, srcs[0], nil
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	type answer struct {
		v    T
		base string
		err  error
	}
	ch := make(chan answer, len(srcs))
	for _, b := range srcs {
		go func() {
			v, err := fn(ctx, b)
			ch <- answer{v, b, err}
		}()
	}
	var errs []error
	for range srcs {
		a := <-ch
		if a.err == nil {
			return a.v, a.base, nil
		}
		errs = append(errs, fmt.Errorf("%s: %w", a.base, a.err))
	}
	return zero, "", errors.Join(errs...)
}

// Latest is the tag of repo's ("owner/name") latest release, asked of
// every source at once: the first tag that ok accepts (nil: any) wins.
// GitHub answers /releases/latest with a redirect to …/releases/tag/<tag>;
// the ActNow mirror, files served by R2, which cannot redirect, with
// JSON {"tag": "<tag>"}. Either answer is read from any source.
func (c *Client) Latest(ctx context.Context, repo string, ok func(tag string) bool) (string, string, error) {
	return Try(ctx, c, func(ctx context.Context, base string) (string, error) {
		hc := *c.client()
		hc.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
		u := base + "/" + strings.Trim(repo, "/") + "/releases/latest"
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
		if err != nil {
			return "", err
		}
		resp, err := hc.Do(req)
		if err != nil {
			return "", err
		}
		defer resp.Body.Close()
		if resp.StatusCode == http.StatusOK {
			var v struct {
				Tag string `json:"tag"`
			}
			if err := json.NewDecoder(io.LimitReader(resp.Body, 64<<10)).Decode(&v); err != nil || v.Tag == "" {
				return "", fmt.Errorf("latest release of %s: not a redirect or {\"tag\"}", repo)
			}
			if strings.ContainsAny(v.Tag, "/\\") || v.Tag == "." || v.Tag == ".." || (ok != nil && !ok(v.Tag)) {
				return "", fmt.Errorf("latest release of %s: unexpected tag %q", repo, v.Tag)
			}
			return v.Tag, nil
		}
		loc := resp.Header.Get("Location")
		if resp.StatusCode/100 != 3 || loc == "" {
			return "", &StatusError{URL: u, Status: resp.Status, Code: resp.StatusCode}
		}
		tag := path.Base(strings.TrimRight(loc, "/"))
		if !strings.Contains(loc, "/releases/tag/") || tag == "" || tag == "." || tag == "/" || (ok != nil && !ok(tag)) {
			return "", fmt.Errorf("latest release of %s: unexpected %q", repo, loc)
		}
		return tag, nil
	})
}

// Small downloads base+p, at most limit bytes, from one source.
func (c *Client) Small(ctx context.Context, base, p string, limit int64) ([]byte, error) {
	u := base + p
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.client().Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, &StatusError{URL: u, Status: resp.Status, Code: resp.StatusCode}
	}
	b, err := io.ReadAll(io.LimitReader(resp.Body, limit+1))
	if err == nil && int64(len(b)) > limit {
		err = fmt.Errorf("%s is larger than %d bytes", path.Base(p), limit)
	}
	return b, err
}

// Bytes downloads p (at most limit bytes) from the first source that
// serves it and check accepts (nil: any).
func (c *Client) Bytes(ctx context.Context, p string, limit int64, check func([]byte) error) ([]byte, string, error) {
	return Try(ctx, c, func(ctx context.Context, base string) ([]byte, error) {
		b, err := c.Small(ctx, base, p, limit)
		if err == nil && check != nil {
			err = check(b)
		}
		return b, err
	})
}
