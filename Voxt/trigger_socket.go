package main

// The socket of `Voxt toggle` and `Voxt cancel` (trigger_cli.go,
// internal/triggerapi): open while the app runs, in the default data
// folder, 0600. An action arrives as one tap of its feature's shortcut in
// the app in front, so it starts, stops and cancels as the shortcut does
// (hotkey.Route), with no key needed.

import (
	"context"
	"errors"
	"fmt"
	"log"
	"path/filepath"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/historyapi"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/triggerapi"
)

type triggerSocketState struct {
	triggerStop context.CancelFunc
}

// openTriggerSocket runs once, when the app is ready.
func (a *app) openTriggerSocket() {
	listener, err := historyapi.Listen(filepath.Join(a.home(), triggerapi.SocketName))
	if err != nil {
		log.Printf("trigger socket: %v", err)
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	server := &triggerapi.Server{Fire: a.fireForCommand}
	go func() {
		if err := server.Serve(ctx, listener); err != nil {
			log.Printf("trigger socket: %v", err)
		}
	}()
	a.triggerStop = cancel
}

// closeTriggerSocket removes the socket, for quitting.
func (a *app) closeTriggerSocket() {
	if a.triggerStop != nil {
		a.triggerStop()
	}
}

// fireForCommand runs action on the main thread, from the socket's
// goroutine, and returns once it has.
func (a *app) fireForCommand(ctx context.Context, action string) error {
	done := make(chan error, 1)
	a.post(func() { done <- a.triggerFired(action) })
	select {
	case err := <-done:
		return err
	case <-ctx.Done():
		return errors.New("Voxt did not answer in time")
	}
}

// triggerFired acts on a command's action as on a tap of its feature's
// shortcut, except a hidden feature, whose shortcuts are off too.
func (a *app) triggerFired(action string) error {
	binding := proto.HotkeyCancel
	switch action {
	case triggerapi.ActionNote:
		if a.prefs.Features.IsHidden(settings.HideNotes) {
			return errors.New("Notes are hidden in Settings → General")
		}
		binding = hotkey.Note
	case triggerapi.ActionTranslate, triggerapi.ActionRewrite:
		if a.prefs.Features.IsHidden(action) { // HideTranslate and HideRewrite are the shortcuts' ids
			return fmt.Errorf("%s is hidden in Settings → General", strings.ToUpper(action[:1])+action[1:])
		}
		binding = hotkey.ID(action, 0)
	case triggerapi.ActionTranscribe:
		binding = hotkey.ID(action, 0)
	}
	front := a.front
	a.logs.Append(core.LogEntry{Time: time.Now(), Level: core.SeverityDebug, Source: sessionSource, Message: "command line: " + action + " in " + front.BundleID})
	a.hotkeyFired(proto.HotkeyEventParams{Binding: binding, Phase: proto.PhaseTrigger, Behavior: proto.BehaviorTap, Front: &front})
	return nil
}
