//go:build linux && (amd64 || arm64)

package linux

import (
	"fmt"
	"net/url"
	"unsafe"

	"github.com/ebitengine/purego"
)

// Downloads: the web context reports them, WebKit asks each where to save
// the file, and reports how it ended ("failed" comes before "finished").

type download struct {
	w         *window
	url, path string
	err       error
}

var (
	downloads       = map[ptr]*download{}
	downloadsHooked bool

	cbDownloadStarted, cbDownloadDestination, cbDownloadFinished, cbDownloadFailed ptr
)

func initDownloadCallbacks() {
	cbDownloadStarted = purego.NewCallback(func(ctx, d, data ptr) {
		w := theBackend.byWebView[webkitDownloadGetWebView(d)]
		if w == nil {
			return
		}
		downloads[d] = &download{w: w, url: goStr(webkitURIRequestGetURI(webkitDownloadGetRequest(d)))}
		connect(d, "decide-destination", cbDownloadDestination, 0)
		connect(d, "finished", cbDownloadFinished, 0)
		connect(d, "failed", cbDownloadFailed, 0)
	})
	cbDownloadDestination = purego.NewCallback(func(d, suggested, data ptr) bool {
		dl := downloads[d]
		if dl == nil {
			return false
		}
		dl.path = dl.w.h.DownloadStarted(dl.url, goStr(suggested))
		if dl.path == "" {
			delete(downloads, d)
			webkitDownloadCancel(d)
			return true
		}
		webkitDownloadSetAllowOverwrite(d, true)
		webkitDownloadSetDestination(d, cs((&url.URL{Scheme: "file", Path: dl.path}).String()))
		return true
	})
	cbDownloadFailed = purego.NewCallback(func(d, gerr, data ptr) {
		if dl := downloads[d]; dl != nil {
			// GError: domain, code, then the message.
			dl.err = fmt.Errorf("mygo: download failed: %s", goStr(*(*ptr)(unsafe.Add(*(*unsafe.Pointer)(unsafe.Pointer(&gerr)), 8))))
		}
	})
	cbDownloadFinished = purego.NewCallback(func(d, data ptr) {
		if dl := downloads[d]; dl != nil {
			delete(downloads, d)
			dl.w.h.DownloadFinished(dl.url, dl.path, dl.err)
		}
	})
}

// hookDownloads listens to the downloads of the web context once.
func hookDownloads() {
	if !downloadsHooked {
		downloadsHooked = true
		connect(webkitWebContextGetDefault(), "download-started", cbDownloadStarted, 0)
	}
}
