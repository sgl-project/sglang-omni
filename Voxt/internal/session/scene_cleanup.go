package session

import (
	"github.com/sgl-project/sglang-omni/Voxt/internal/appctx"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// cleanupCanVary reports whether a browser's pending address could change
// the global clean-up choice. Default-only groups never hold up model opening.
func cleanupCanVary(rules appctx.Rules, bundleID string, global bool) bool {
	for _, g := range rules.Groups {
		if g.Enabled && g.Features.Has(appctx.Transcribe) && g.TranscribeCleanup.Enabled(global) != global &&
			(g.HasApp(bundleID) || len(g.URLs) > 0) {
			return true
		}
	}
	return false
}

// transcribeCleanup is the same decision for model opening, warming and
// clean-up. A scene is decided once, with the existing bounded URL lookup.
// Notes and Try keep their global setting regardless of the foreground app.
func (r *run) transcribeCleanup() bool {
	global := r.settings.Features.Transcribe.Enhance
	if r.req.Kind != store.KindTranscribe || r.req.Note || r.req.Try {
		return global
	}
	m, known := r.knownScene()
	if !known {
		if !r.sc.cleanupPending {
			return global
		}
		m = r.scene()
	}
	if m.Matched() {
		return m.Group.TranscribeCleanup.Enabled(global)
	}
	return global
}

// windowCleanup is the choice for taking the window picture as the
// recording starts: the scene's when it is decided. While a browser's
// address is still pending, the picture goes when an override could
// turn clean-up on — the recording must not wait for the address, and
// a picture the ended-up-off choice does not need is never sent.
func (r *run) windowCleanup() bool {
	global := r.settings.Features.Transcribe.Enhance
	if r.req.Kind != store.KindTranscribe || r.req.Note || r.req.Try {
		return global
	}
	if m, known := r.knownScene(); known {
		if m.Matched() {
			return m.Group.TranscribeCleanup.Enabled(global)
		}
		return global
	}
	return global || r.sc.cleanupPending
}

// resolveServices passes the effective choice in a private settings copy.
// The recording starts alongside this work, including any URL wait.
// speech: as Resolver's.
func (r *run) resolveServices(speech bool) (Services, error) {
	s := r.settings
	if r.req.Kind == store.KindTranscribe {
		s.Features.Transcribe.Enhance = r.transcribeCleanup()
	}
	if err := r.ctx.Err(); err != nil {
		return Services{}, err
	}
	return r.m.cfg.Resolve(r.ctx, r.req.Kind, s, speech)
}
