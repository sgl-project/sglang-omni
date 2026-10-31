package mygo

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// DownloadEvent is passed to Page.OnWillDownload listeners. Setting Path
// chooses where the file goes; preventing the event cancels the download.
type DownloadEvent struct {
	Preventable
	// URL of the download.
	URL string
	// SuggestedName is the file name the page or the server suggests.
	SuggestedName string
	// Path is where the file is saved: by default SuggestedName in the
	// Downloads directory, numbered when a file has that name. A file
	// already at a Path listeners set is replaced.
	Path string
}

// Download is a finished download, passed to Page.OnDownloadDone.
type Download struct {
	URL  string
	Path string
	// Err is why the download failed, or nil.
	Err error
}

// OnWillDownload is called when a page starts a download: a link with the
// download attribute, or a response the page cannot show, such as a file
// sent as an attachment. Without listeners, downloads are saved to the
// Downloads directory.
func (p *Page) OnWillDownload(fn func(e *DownloadEvent)) (off func()) {
	return p.w.onWillDownload.add(fn, false)
}

// OnDownloadDone is called when a download ended.
func (p *Page) OnDownloadDone(fn func(d *Download)) (off func()) {
	return p.w.onDownloadDone.add(fn, false)
}

func (h *windowHandler) DownloadStarted(url, suggested string) string {
	name := filepath.Base(filepath.Clean("/" + strings.ReplaceAll(suggested, `\`, "/")))
	if name == "/" || name == "." || name == "" {
		name = "download"
	}
	e := &DownloadEvent{URL: url, SuggestedName: name}
	if dir, err := App.Path(PathDownloads); err == nil && os.MkdirAll(dir, 0o755) == nil {
		e.Path = uniquePath(filepath.Join(dir, name))
	}
	fire1(&h.w.onWillDownload, e)
	if e.prevented || e.Path == "" {
		return ""
	}
	return e.Path
}

func (h *windowHandler) DownloadFinished(url, path string, err error) {
	fire1(&h.w.onDownloadDone, &Download{URL: url, Path: path, Err: err})
}

// uniquePath numbers path, like "name (2).ext", when a file has its name.
func uniquePath(path string) string {
	if _, err := os.Lstat(path); err != nil {
		return path
	}
	ext := filepath.Ext(path)
	base := strings.TrimSuffix(path, ext)
	for i := 2; ; i++ {
		p := base + " (" + strconv.Itoa(i) + ")" + ext
		if _, err := os.Lstat(p); err != nil {
			return p
		}
	}
}
