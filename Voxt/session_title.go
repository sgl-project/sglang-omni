package main

// A session's title from a language model (#4005): the head of the
// session's first instruction names it at once (titleSession); beside
// that, the model Rewrite uses is asked for a few words that sum the
// instruction up, and its answer replaces the head. The request runs
// after the session's own ended and outside it: it never waits on or
// joins the session's conversation, and only the instruction goes out.
// A failure, a timeout or no model leaves the head as the title.

import (
	"context"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// sessionTitleTimeout bounds the title request; sessionTitleInput is how
// much of the instruction goes with it.
var sessionTitleTimeout = 15 * time.Second

const sessionTitleInput = 1000

const sessionTitlePrompt = "Write a title of at most five words for the user's message, in the language of the message. Answer with the title only: no quotes, no period, no explanation."

// askSessionTitle asks Rewrite's language model for a title of text, the
// first instruction of session id, off the main thread. The answer is
// dropped when the session is gone or was named otherwise meanwhile.
func (a *app) askSessionTitle(id int, text string) {
	models, want := a.local, a.prefs.Features.Rewrite.LLM
	if models == nil {
		return
	}
	model, ok := local.Choose(local.ForFeature(models.Installed(local.LLM), string(store.KindRewrite)), want)
	if !ok {
		return
	}
	if runes := []rune(text); len(runes) > sessionTitleInput {
		text = string(runes[:sessionTitleInput])
	}
	request := llm.Request{Messages: []llm.Message{{Role: llm.System, Content: sessionTitlePrompt}, {Role: llm.User, Content: text}}}
	timeout := sessionTitleTimeout
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()
		chat, _, err := models.LLM(ctx, model.ID)
		if err != nil {
			return
		}
		response, err := chat.Chat(ctx, request)
		line, _, _ := strings.Cut(strings.TrimSpace(response.Text), "\n")
		title := sessionTitle(strings.Trim(line, " \t\"'“”‘’「」《》.。"))
		if err != nil || title == "" {
			return
		}
		a.post(func() { a.setSessionTitle(id, text, title) })
	})
}

// setSessionTitle gives session id the model's title for instruction,
// unless the session is gone or its title is no longer that
// instruction's head.
func (a *app) setSessionTitle(id int, instruction, title string) {
	n := a.slotIndex(id)
	if n < 0 || a.slots[n].titled == "" || !strings.HasPrefix(a.slots[n].titled, instruction) {
		return
	}
	a.slots[n].title, a.slots[n].titled = title, ""
	a.buildTrayMenu()
	a.overlaySync()
}

// sessionNameMax is how long a title the user gives a session can be.
const sessionNameMax = 40

// renameSession starts typing session id's title on the session row (a
// double click on it). The card's own keys (c, r) go to the text
// meanwhile, and the cancel shortcut ends the typing, not the card.
func (a *app) renameSession(id int) {
	if a.slotIndex(id) < 0 {
		return
	}
	a.renaming, a.renameID = true, id
	a.syncHotkeys()
	a.overlaySync()
}

// nameSession gives session id the title the user typed (Return). An
// empty one brings the automatic title back.
func (a *app) nameSession(id int, name string) {
	if n := a.slotIndex(id); n >= 0 {
		name = strings.Join(strings.Fields(name), " ")
		if runes := []rune(name); len(runes) > sessionNameMax {
			name = string(runes[:sessionNameMax])
		}
		a.slots[n].name = name
	}
	a.endRename()
	a.buildTrayMenu()
}

// endRename ends the typing and keeps the title as it was.
func (a *app) endRename() {
	a.renaming = false
	a.syncHotkeys()
	a.overlaySync()
}
