// The Hub API: repository info, and the race between endpoints (the official one and its mirrors).

package hfhub

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/hehehai/actnowkit/models"
)

// repoInfo is what /api/models/{repo}/revision/{rev}?blobs=true says.
type repoInfo struct {
	ID       string          `json:"id"`
	SHA      string          `json:"sha"`
	Gated    json.RawMessage `json:"gated"` // false, "auto" or "manual"
	Siblings []struct {
		Path   string `json:"rfilename"`
		BlobID string `json:"blobId"`
		Size   int64  `json:"size"`
		LFS    *struct {
			SHA256 string `json:"sha256"`
			Size   int64  `json:"size"`
		} `json:"lfs"`
	} `json:"siblings"`
}

func (r *repoInfo) gated() bool {
	g := strings.TrimSpace(string(r.Gated))
	return g != "" && g != "false" && g != "null"
}

// apiError is a failed request.
type apiError struct {
	status int
	err    error
}

func (e *apiError) Error() string { return e.err.Error() }

func (e *apiError) Unwrap() error { return e.err }

func revisionOf(e Entry) string {
	if e.Revision == "" {
		return "main"
	}
	return e.Revision
}

func isCommit(rev string) bool { return models.IsCommit(rev) }

func (c *Client) official(ep string) bool { return ep == c.opts.Official }

func (c *Client) newRequest(ctx context.Context, ep, rawURL, token string) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", c.opts.UserAgent)
	// Mirrors never see the token. Go drops it on redirects to other
	// hosts (the CDN).
	if token != "" && c.official(ep) {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	return req, nil
}

func (c *Client) fetchInfo(ctx context.Context, ep, repo, rev, token string) (*repoInfo, error) {
	u := ep + "/api/models/" + repo + "/revision/" + url.PathEscape(rev) + "?blobs=true"
	req, err := c.newRequest(ctx, ep, u, token)
	if err != nil {
		return nil, err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	switch {
	case resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden:
		return nil, &apiError{resp.StatusCode, fmt.Errorf("%s: %w (HTTP %d)", repo, ErrGated, resp.StatusCode)}
	case resp.StatusCode == http.StatusNotFound:
		return nil, &apiError{resp.StatusCode, fmt.Errorf("%s@%s: %w", repo, rev, ErrNotFound)}
	case resp.StatusCode != http.StatusOK:
		return nil, &apiError{resp.StatusCode, fmt.Errorf("%s: HTTP %d from %s", repo, resp.StatusCode, ep)}
	}
	var info repoInfo
	if err := json.NewDecoder(io.LimitReader(resp.Body, 32<<20)).Decode(&info); err != nil {
		return nil, fmt.Errorf("%s: %s: %w", repo, ep, err)
	}
	if info.SHA == "" {
		return nil, fmt.Errorf("%s: %s sent no commit", repo, ep)
	}
	return &info, nil
}

// info asks the endpoints for repo@rev, racing them unless a recent winner
// is known, and returns the answer and the endpoint that gave it. Gated
// repos are then asked of the official endpoint, with the token.
func (c *Client) info(ctx context.Context, repo, rev string) (*repoInfo, string, error) {
	token := c.token()
	c.mu.Lock()
	winner := c.winner
	if time.Since(c.wonAt) > winnerTTL {
		winner = ""
	}
	c.mu.Unlock()

	var info *repoInfo
	var ep string
	var err error
	if winner != "" {
		info, err = c.fetchInfo(ctx, winner, repo, rev, token)
		ep = winner
		if err != nil && !isDefinite(err) {
			info, winner = nil, "" // the winner failed: race again
		}
	}
	if winner == "" {
		info, ep, err = c.race(ctx, repo, rev, token)
	}
	if err != nil {
		return nil, "", err
	}
	if info.gated() {
		if token == "" {
			return nil, "", fmt.Errorf("%s: %w", repo, ErrGated)
		}
		if !c.official(ep) {
			ep = c.opts.Official
			if info, err = c.fetchInfo(ctx, ep, repo, rev, token); err != nil {
				return nil, "", err
			}
		}
	}
	return info, ep, nil
}

// isDefinite reports an answer that another endpoint would repeat (the
// repo is missing or gated), as opposed to an endpoint being down.
func isDefinite(err error) bool {
	var ae *apiError
	return errors.As(err, &ae) && (ae.status == 401 || ae.status == 403 || ae.status == 404)
}

// race asks every endpoint at once and takes the first good answer.
func (c *Client) race(ctx context.Context, repo, rev, token string) (*repoInfo, string, error) {
	eps := c.opts.Endpoints
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	type answer struct {
		info *repoInfo
		ep   string
		err  error
	}
	ch := make(chan answer, len(eps))
	for _, ep := range eps {
		go func() {
			info, err := c.fetchInfo(ctx, ep, repo, rev, token)
			ch <- answer{info, ep, err}
		}()
	}
	var errs []error
	var definite error
	for range eps {
		a := <-ch
		if a.err == nil {
			c.mu.Lock()
			c.winner, c.wonAt = a.ep, time.Now()
			c.mu.Unlock()
			return a.info, a.ep, nil
		}
		if isDefinite(a.err) && definite == nil {
			definite = a.err
		}
		errs = append(errs, a.err)
	}
	if ctx.Err() != nil {
		return nil, "", ctx.Err()
	}
	if definite != nil {
		return nil, "", definite
	}
	return nil, "", fmt.Errorf("%s: %w: %w", repo, ErrOffline, errors.Join(errs...))
}

func escapePath(p string) string {
	parts := strings.Split(p, "/")
	for i, s := range parts {
		parts[i] = url.PathEscape(s)
	}
	return strings.Join(parts, "/")
}
