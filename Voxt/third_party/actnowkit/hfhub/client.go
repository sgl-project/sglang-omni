package hfhub

import (
	"context"
	"errors"
	"net/http"
	"os"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/hehehai/actnowkit/family"
	"github.com/hehehai/actnowkit/models"
	"github.com/hehehai/actnowkit/settings"
)

// Errors of the Client. Code maps them to stable codes.
var (
	ErrGated     = errors.New("the repo is gated: accept its terms on huggingface.co and add a token")
	ErrNotFound  = errors.New("repo or revision not found")
	ErrDiskSpace = errors.New("not enough free disk space")
	ErrChecksum  = errors.New("checksum mismatch")
	ErrOffline   = errors.New("no endpoint answered")
)

// Stable error codes (Code), for the apps' own job errors.
const (
	CodeGated          = "gated"
	CodeDiskSpace      = "disk_space"
	CodeOffline        = "offline"
	CodeChecksum       = "checksum"
	CodeNotFound       = "not_found"
	CodeBusy           = "busy"      // a family lock is held (*models.BusyError)
	CodeCancelled      = "cancelled" // the context was cancelled
	CodeDownloadFailed = "download_failed"
)

// Code is a stable code for an error of the Client: CodeGated,
// CodeDiskSpace, CodeOffline, CodeChecksum, CodeNotFound, CodeBusy,
// CodeCancelled, else CodeDownloadFailed; "" for nil.
func Code(err error) string {
	switch {
	case err == nil:
		return ""
	case errors.Is(err, ErrGated):
		return CodeGated
	case errors.Is(err, ErrDiskSpace):
		return CodeDiskSpace
	case errors.Is(err, ErrOffline):
		return CodeOffline
	case errors.Is(err, ErrChecksum):
		return CodeChecksum
	case errors.Is(err, ErrNotFound):
		return CodeNotFound
	case errors.Is(err, family.ErrLocked):
		return CodeBusy
	case errors.Is(err, context.Canceled):
		return CodeCancelled
	default:
		return CodeDownloadFailed
	}
}

// DefaultEndpoints are raced when Options.Endpoints is empty and
// $HF_ENDPOINT unset: the official hub and its best-known mirror.
var DefaultEndpoints = []string{"https://huggingface.co", "https://hf-mirror.com"}

// DefaultUserAgent is the User-Agent of requests unless Options.UserAgent.
const DefaultUserAgent = "actnowkit-hfhub"

// winnerTTL is how long a race's winner is reused before racing again.
const winnerTTL = 24 * time.Hour

// Client implements Downloader over the Hugging Face HTTP API.
type Client struct {
	opts Options
	http *http.Client

	// progressEvery is the least time between progress reports.
	progressEvery time.Duration
	// waitPoll is how often a busy blob lock is tried again.
	waitPoll time.Duration

	mu       sync.Mutex
	winner   string // endpoint that won the last race
	wonAt    time.Time
	repoLock map[string]*sync.Mutex
}

var _ Downloader = (*Client)(nil)

// New returns a Client. Options left zero take their defaults.
func New(o Options) *Client {
	if o.Store != nil {
		o.CacheDir = o.Store.Hub
		if o.BundleID == "" {
			o.BundleID = o.Store.BundleID
		}
	}
	if o.CacheDir == "" {
		o.CacheDir = CacheDir()
	}
	if o.UserAgent == "" {
		o.UserAgent = DefaultUserAgent
	}
	env := strings.TrimRight(os.Getenv("HF_ENDPOINT"), "/")
	if o.Official == "" {
		o.Official = env
	}
	if o.Official == "" {
		o.Official = DefaultEndpoints[0]
	}
	o.Official = strings.TrimRight(o.Official, "/")
	if len(o.Endpoints) == 0 && env != "" {
		o.Endpoints = []string{env}
	}
	if len(o.Endpoints) == 0 {
		o.Endpoints = DefaultEndpoints
	}
	var eps []string
	for _, e := range o.Endpoints {
		if e = strings.TrimRight(strings.TrimSpace(e), "/"); e != "" && !slices.Contains(eps, e) {
			eps = append(eps, e)
		}
	}
	o.Endpoints = eps
	if o.Parallel <= 0 {
		o.Parallel = 4
	}
	if o.ChunkSize <= 0 {
		o.ChunkSize = 16 << 20
	}
	if o.ReserveBytes == 0 {
		o.ReserveBytes = 2e9
	}
	hc := o.HTTPClient
	if hc == nil {
		hc = NewHTTPClient(o.Proxy, 30*time.Second)
	}
	return &Client{
		opts: o, http: hc, repoLock: map[string]*sync.Mutex{},
		progressEvery: 250 * time.Millisecond, waitPoll: 200 * time.Millisecond,
	}
}

// Options returns the options in use, defaults filled in.
func (c *Client) Options() Options { return c.opts }

// lockRepo is the in-process lock of a repo, the first of the locks
// (spec §7.4): two downloads of one repo in this Client run in turn.
func (c *Client) lockRepo(repo string) func() {
	c.mu.Lock()
	l := c.repoLock[repo]
	if l == nil {
		l = &sync.Mutex{}
		c.repoLock[repo] = l
	}
	c.mu.Unlock()
	l.Lock()
	return l.Unlock
}

// who is this app as a lock owner doing op.
func (c *Client) who(op string) models.Owner {
	return models.Owner{BundleID: c.opts.BundleID, Op: op}
}

// post sends the family's models notification, when in a family.
func (c *Client) post() {
	if c.opts.Store != nil {
		_ = c.opts.Store.Family.Post(family.TopicModels)
	}
}

func (c *Client) token() string {
	if c.opts.Token == nil {
		return ""
	}
	return strings.TrimSpace(c.opts.Token())
}

// backup is the family's models.backup (spec §7.12): false, the default,
// keeps finished repos out of Time Machine.
func (c *Client) backup() bool {
	if c.opts.Store == nil {
		return false
	}
	s, _, _ := settings.Read(c.opts.Store.Family)
	return settings.ModelsBackup(s)
}
