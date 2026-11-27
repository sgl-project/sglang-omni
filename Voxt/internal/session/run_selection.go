package session

import (
	"context"
	"slices"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// selectionSettleMS, selectionPollMS and selectionModifierWaitMS: the
// helper's waits around a selection read. A caret-only reading waits out
// a settle for a selection made with ⌘A in the shortcut's motion (#4467);
// the helper's ⌘C waits for the shortcut's held modifiers to lift, so the
// app reads ⌘C and not a key that copies nothing (#3835).
const (
	selectionSettleMS       = 600
	selectionPollMS         = 60
	selectionModifierWaitMS = 800
	selectionOverheadMS     = 300
	// selectionCopyMS is the selection read's ⌘C budget, sent explicitly
	// so the deadline arithmetic does not depend on the helper's default.
	selectionCopyMS = 300
)

// intp is v's address: the selection params carry their budgets as
// pointers, so an explicit 0 reaches the helper (nil is its default).
func intp(v int) *int { return &v }

// readSelection reads the selected text of the app in front (Translate,
// Rewrite) and puts it in the snapshot; "" for none, or when it could
// not be read (the session goes on without one). listened, Rewrite's,
// closes when the recording ends; nil reads once.
func (r *run) readSelection(listened <-chan struct{}) string {
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
	read := func() (proto.SelectionResult, error) {
		ctx, cancel := context.WithTimeout(r.ctx, timeout)
		defer cancel()
		return cfg.Select.Selection(ctx, params)
	}
	res, err := read()
	// Rewrite reads while the user speaks: with the shortcut's keys still
	// down (Option of a held Option+C) the helper skips the ⌘C, so read
	// again until they lift, the last time once the recording ends (VX-141).
	for last := false; listened != nil && err == nil && !last && copySkipped(res.Steps); res, err = read() {
		select {
		case <-r.ctx.Done():
			return ""
		case <-listened:
			last = true
		case <-time.After(selectionRetry):
		}
	}
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

// selectionRetry spaces Rewrite's reads while the shortcut's keys are
// down; each read already waits selectionModifierWaitMS for them.
const selectionRetry = 100 * time.Millisecond

// copySkipped: the helper skipped its ⌘C because keys were held.
func copySkipped(steps []string) bool {
	return slices.ContainsFunc(steps, func(s string) bool { return strings.HasPrefix(s, "copy: skipped") })
}
