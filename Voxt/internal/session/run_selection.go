package session

import (
	"context"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// readSelection reads the selected text of the app in front (Translate,
// Rewrite) and puts it in the snapshot; "" for none, or when it could
// not be read (the session goes on without one).
func (r *run) readSelection() string {
	defer r.m.update(r, func(s *Snapshot) { s.ReadingSelection = false })
	cfg := r.m.cfg
	timeout, params := cfg.ReadTimeout, proto.SelectionParams{
		AllowCopy:      AllowCopy(r.req.Target.BundleID),
		CopyTimeoutMS:  selectionCopyMS,
		CaretSettleMS:  intp(selectionSettleMS),
		CaretPollMS:    intp(selectionPollMS),
		ModifierWaitMS: intp(selectionModifierWaitMS),
	}
	if r.req.Kind == store.KindRewrite {
		timeout, params.CopyTimeoutMS = cfg.RewriteReadTimeout, rewriteCopyMS
	}
	ctx, cancel := context.WithTimeout(r.ctx, timeout)
	defer cancel()
	res, err := cfg.Select.Selection(ctx, params)
	if err != nil {
		if r.ctx.Err() == nil {
			r.logf("warning", "reading the selection: %v", err)
		}
		return ""
	}
	text := res.Text
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		// The helper's steps say which reading found nothing (#3835).
		r.logf("info", "no selection (%s, %.0f ms; %s)", res.Via, res.MS, strings.Join(res.Steps, ", "))
		return ""
	}
	if r.req.Kind == store.KindRewrite && !r.settings.Privacy.SendClipboardImage {
		r.clip = nil
	} else if r.req.Kind == store.KindRewrite {
		switch {
		case len(res.Image) > 0 && (trimmed == "[Image]" || r.settings.Privacy.SendClipboardImageWithSelection):
			// The copy fallback returns the newly selected image before it
			// restores the clipboard, never an earlier copied image.
			r.clip = res.Image
		case trimmed == "[Image]" || !r.settings.Privacy.SendClipboardImageWithSelection:
			// Text takes the text path by default. A placeholder without a
			// newly decoded image never revives the earlier clipboard image.
			r.clip = nil
		}
	}
	for _, t := range r.req.Previous {
		// A follow-up's: the text selected before, still selected, is in
		// the conversation already (#3816).
		if strings.TrimSpace(t.Selection) == trimmed {
			r.logf("info", "the selection is in the conversation already")
			return ""
		}
	}
	r.logf("info", "selection: %s via %s in %.0f ms", r.logText(text), res.Via, res.MS)
	r.m.update(r, func(s *Snapshot) { s.Selection, s.SelectionVia = text, res.Via })
	return text
}
