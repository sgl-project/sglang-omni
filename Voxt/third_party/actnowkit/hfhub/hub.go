package hfhub

import (
	"context"
	"net/http"
	"strconv"
	"strings"

	"github.com/hehehai/actnowkit/models"
)

// Entry is a model to download: a repo at a revision, optionally only
// some of its files, and the auxiliary repos it needs (a tokenizer, a
// G2P model).
type Entry struct {
	ID       string `json:"id"`                 // the app's catalogue id; the claim's model
	Repo     string `json:"repo"`               // Hugging Face repo id, "owner/name"
	Revision string `json:"revision,omitempty"` // commit (spec §7.2: pin one); "" = main
	Files    []File `json:"files,omitempty"`    // only these files; none = the whole repo
	Aux      []Aux  `json:"aux,omitempty"`
	// Family and Role go into the claim (spec §7.3): the model series
	// ("qwen3-asr") and its use (models.RoleASR…). Optional.
	Family string `json:"family,omitempty"`
	Role   string `json:"role,omitempty"`
}

// File is a file of a repo.
type File struct {
	Path   string `json:"path"` // inside the repo
	Size   int64  `json:"size"`
	SHA256 string `json:"sha256,omitempty"` // LFS files: verified after download
}

// Aux is an auxiliary repo of an entry.
type Aux struct {
	Repo     string `json:"repo"`
	Revision string `json:"revision,omitempty"`
	Files    []File `json:"files,omitempty"`
}

// CacheDir is the family's default model library, models.DefaultHub
// ($FAMILY/models/hub; spec §7.1, v1.0.9). Family apps use
// models.ResolveHub, which puts the family's choice (models.root) first.
// Before v0.8.0 it was huggingface_hub's cache (models.EnvHub).
func CacheDir() string { return models.DefaultHub() }

// RepoDir is the cache directory of a repo id ("owner/name").
func RepoDir(cacheDir, repo string) string { return models.RepoDir(cacheDir, repo) }

// LocalState is how much of an entry is on disk.
type LocalState string

const (
	Missing    LocalState = "missing"    // nothing
	Partial    LocalState = "partial"    // some files, or interrupted downloads (resumable)
	Downloaded LocalState = "downloaded" // every file present (models.Installed)
)

// Local describes an entry on disk.
type Local struct {
	State    LocalState
	Snapshot string // snapshots/<commit> directory, for the engine; "" when Missing
	// Bytes on disk: the blobs the snapshots link (each once), plus the
	// partial downloads of repos that are not Downloaded.
	Bytes   int64
	Missing []string // files absent; "<repo>:" for an auxiliary repo
	// Damaged are files of the wrong size (truncated): the entry is
	// Partial; Download or Verify repairs them. "<repo>:<path>" for an
	// auxiliary repo.
	Damaged []string
}

// UnknownOwner is Progress.WaitingFor when the holder of a blob lock left
// no owner file (huggingface_hub, an app predating the spec).
const UnknownOwner = "unknown"

// Progress of a download.
type Progress struct {
	Repo       string
	File       string // current file
	Bytes      int64  // done, all files
	TotalBytes int64
	Mirror     string // endpoint in use (after racing)
	// WaitingFor is the bundle id of the app downloading a blob this
	// download waits for (UnknownOwner when unknown); "" when not
	// waiting. Show "Downloading in <app>…" (models.TextDownloadingIn).
	WaitingFor string
}

// Options configure a Client.
type Options struct {
	// CacheDir is $HUB; "" = CacheDir(). Ignored when Store is set: its
	// Hub is used.
	CacheDir string
	// Store makes downloads claim what they fetched and deletions follow
	// the family rules (spec §7.3, §7.6), and posts the family's models
	// notification. nil = no claims (a tool outside the family).
	Store *models.Store
	// BundleID names this app in lock owner files ("Downloading in
	// Dicta…"); "" = Store.BundleID.
	BundleID string
	// UserAgent of requests; "" = "actnowkit-hfhub".
	UserAgent string
	// Endpoints are raced on the first request; the fastest healthy one
	// wins ("https://huggingface.co", "https://hf-mirror.com"). Empty =
	// $HF_ENDPOINT, else DefaultEndpoints.
	Endpoints []string
	// Official is the endpoint trusted with the token: gated repos
	// download only from it, and mirrors never see the token. "" =
	// $HF_ENDPOINT, else "https://huggingface.co".
	Official string
	// Token returns the HF token for gated repos (see ResolveToken); nil
	// or "" = anonymous. Called for each request: keep it cheap.
	Token func() string
	// Parallel is the number of files (and ranges of big files) fetched at
	// once. 0 = 4.
	Parallel int
	// ChunkSize is the size of the ranges big files are fetched in, in
	// parallel; files under two chunks download whole. 0 = 16 MiB.
	ChunkSize int64
	// ReserveBytes is the free space a download must leave on the cache
	// volume. 0 = 2 GB; negative = no check.
	ReserveBytes int64
	// Proxy is "" (automatic: environment, then the macOS system proxy),
	// ProxyDirect, or a proxy URL; see ProxyFunc.
	Proxy string
	// HTTPClient makes the requests; nil = NewHTTPClient(Proxy, 30 s).
	HTTPClient *http.Client
}

// Downloader manages model files. Goroutine-safe; methods may block on I/O.
type Downloader interface {
	// Status reports what of e is on disk. Cheap enough for a page load,
	// but still I/O: not from a build pass.
	Status(ctx context.Context, e Entry) (Local, error)
	// Plan lists the files to fetch (from the HF API) and checks free
	// disk space; it errors before downloading anything when space is
	// short or the repo is gated without a token.
	Plan(ctx context.Context, e Entry) (Plan, error)
	// Download fetches what Plan lists, holding the family locks,
	// resuming partial files, verifying every file, writing refs/ last so
	// a snapshot is never half there, and claiming the result. progress
	// is called often; ctx cancels (partial files stay for resuming, by
	// this app or another).
	Download(ctx context.Context, e Entry, progress func(Progress)) error
	// Verify re-hashes the files and returns the bad or missing ones,
	// removing bad blobs so that a later Download repairs them. It needs
	// the HF API when e lists no files with hashes.
	Verify(ctx context.Context, e Entry) ([]string, error)
	// Delete removes e from this app, deleting the files nobody else
	// needs (spec §7.6).
	Delete(ctx context.Context, e Entry, scope models.Scope) ([]models.Result, error)
}

// Plan is what a download will do.
type Plan struct {
	Revision   string // resolved commit
	Files      []File
	TotalBytes int64 // every file
	Bytes      int64 // still to download
	FreeBytes  int64 // on the cache volume; -1 when unknown
	Endpoint   string
	Gated      bool
}

// HumanSize is a size for messages: "≈ 712 MB", "≈ 2.3 GB"; "" for none.
func HumanSize(bytes int64) string {
	switch {
	case bytes <= 0:
		return ""
	case bytes >= 1e9:
		return "≈ " + strings.TrimSuffix(strconv.FormatFloat(float64(bytes)/1e9, 'f', 1, 64), ".0") + " GB"
	default:
		return "≈ " + strconv.FormatInt((bytes+5e5)/1e6, 10) + " MB"
	}
}
