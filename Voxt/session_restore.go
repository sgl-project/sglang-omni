package main

// Reopening a Rewrite or Translate card from its history, in the menu bar
// or the Dock menu. The history is the only copy: deleting it removes the
// menu entry, and a restart loses nothing that the menu can reopen.

import (
	"context"
	"slices"
	"time"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

const (
	reopenCardLimit  = 5
	reopenCardMaxAge = 24 * time.Hour
)

// reopenConversation is one history conversation as the menu lists it.
type reopenConversation struct {
	conversation string
	label        string
}

// currentCardEntries leaves the successful card answers that were not
// replaced by a successful redo, newest first as History.List returns them.
func currentCardEntries(entries []store.Entry) []store.Entry {
	replaced := map[string]bool{}
	for _, entry := range entries {
		if entry.Status == store.StatusDone && entry.Result != "" && entry.Replaces != "" {
			replaced[entry.Replaces] = true
		}
	}
	out := make([]store.Entry, 0, len(entries))
	for _, entry := range entries {
		cardKind := entry.Kind == store.KindRewrite || entry.Kind == store.KindTranslate
		if cardKind && entry.Status == store.StatusDone && entry.Result != "" && !replaced[entry.ID] {
			out = append(out, entry)
		}
	}
	return out
}

// cardFromHistory reconstructs one conversation, oldest turn first. A
// translation that later became a Rewrite conversation is its first turn.
func cardFromHistory(entries []store.Entry, conversation string) (answerCard, bool) {
	var turns []store.Entry
	for _, entry := range slices.Backward(currentCardEntries(entries)) {
		if entry.Conversation == conversation {
			turns = append(turns, entry)
		}
	}
	if len(turns) == 0 {
		return answerCard{}, false
	}
	hasRewrite := slices.ContainsFunc(turns, func(entry store.Entry) bool { return entry.Kind == store.KindRewrite })
	card := answerCard{open: true, conv: conversation, kind: store.KindTranslate}
	for _, entry := range turns {
		card.id, card.text, card.title, card.llmID = entry.ID, entry.Result, entry.Title, entry.LLMProvider
		card.target = proto.App{BundleID: entry.AppBundle, Name: entry.AppName}
		typed := ""
		if entry.Kind == store.KindRewrite {
			typed = entry.RawText
		}
		card.input = session.TextInput{Raw: entry.RawText, Selection: entry.SourceText, Typed: typed}
		if !hasRewrite {
			card.lang = entry.TargetLang
			continue
		}
		card.kind = store.KindRewrite
		instruction, selection := entry.PromptText, entry.SourceText
		if entry.Kind == store.KindTranslate {
			instruction = i18n.T("Translate into %s", uitext.TargetLanguageLabel(entry.TargetLang))
			selection = cmpOr(entry.SourceText, entry.RawText)
		}
		card.turns = append(card.turns, compose.Turn{Instruction: instruction, Selection: selection, Typed: typed,
			Answer: compose.Answer{Title: entry.Title, Content: entry.Result}})
		card.input.Raw = entry.PromptText
	}
	card.conversation = len(card.turns) > 1
	return card, true
}

// reopenLabel names a card in the menu by its app and first instruction;
// a translation without a Rewrite turn uses its translated text.
func reopenLabel(card answerCard) string {
	said := card.text
	if len(card.turns) > 0 {
		said = card.turns[0].Instruction
	}
	return uitext.TrayLabel(card.target.Name + " · " + said)
}

// reopenConversations lists the newest conversations whose current answer
// was not inserted and is under a day old.
func reopenConversations(entries []store.Entry, now time.Time) []reopenConversation {
	entries = currentCardEntries(entries)
	seen := map[string]bool{}
	var out []reopenConversation
	for _, entry := range entries {
		conversation := entry.Conversation
		if conversation == "" || seen[conversation] {
			continue
		}
		seen[conversation] = true
		if entry.Delivery == "insert" || now.Sub(entry.CreatedAt) >= reopenCardMaxAge {
			continue
		}
		if card, ok := cardFromHistory(entries, conversation); ok {
			out = append(out, reopenConversation{conversation: conversation, label: reopenLabel(card)})
		}
		if len(out) == reopenCardLimit {
			break
		}
	}
	return out
}

// reopenMenu lists the cards that can continue, for either menu.
func (a *app) reopenMenu(act func(func()) func(*mygo.MenuItem, *mygo.Window)) []*mygo.MenuItem {
	if !a.prefs.Privacy.KeepHistory || a.sess.State.Live() {
		return nil
	}
	items := []*mygo.MenuItem{{Label: i18n.T("Reopen card"), Disabled: true}}
	for _, conversation := range a.reopen {
		if a.answer.open && a.answer.conv == conversation.conversation {
			continue
		}
		conversation := conversation
		items = append(items, &mygo.MenuItem{Label: conversation.label, Click: act(func() {
			a.restoreAnswer(conversation.conversation)
		})})
	}
	if len(items) == 1 {
		return nil
	}
	return items
}

// restoreAnswer opens a history conversation and resolves its target's
// current process by bundle id; a stopped target still opens, and Insert's
// existing failure path explains that it cannot paste there.
func (a *app) restoreAnswer(conversation string) {
	if !a.prefs.Privacy.KeepHistory || a.sess.State.Live() || conversation == "" {
		return
	}
	a.restoreSeq++
	seq := a.restoreSeq
	history := a.lib.store.History()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		entries, err := history.List(ctx, store.HistoryQuery{Conversation: conversation, Finished: true})
		if err != nil {
			return
		}
		a.post(func() {
			card, ok := cardFromHistory(entries, conversation)
			if !ok || seq != a.restoreSeq || !a.prefs.Privacy.KeepHistory || a.sess.State.Live() {
				return
			}
			helperCall(a, proto.MethodAppList, proto.AppListParams{}, 5*time.Second, func(result proto.AppListResult, _ error) {
				if seq != a.restoreSeq || !a.prefs.Privacy.KeepHistory || a.sess.State.Live() {
					return
				}
				for _, running := range result.Apps {
					if running.Running && running.BundleID == card.target.BundleID {
						card.target.PID = running.PID
						break
					}
				}
				a.setAside(false)
				a.answer, a.followUp, a.notice = card, "", nil
				a.buildTrayMenu()
				a.syncHotkeys()
				a.overlaySync()
			})
		})
	})
}
