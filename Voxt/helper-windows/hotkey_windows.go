//go:build windows

package main

import (
	"fmt"
	"sync"
	"syscall"
	"time"
	"unsafe"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"golang.org/x/sys/windows"
)

const (
	hotkeyKeyboardLL       = 13
	hotkeyMouseLL          = 14
	hotkeyWindowMessage    = 0x0312
	windowKeyDown          = 0x0100
	windowKeyUp            = 0x0101
	windowSystemKeyDown    = 0x0104
	windowSystemKeyUp      = 0x0105
	windowLeftButtonDown   = 0x0201
	windowRightButtonDown  = 0x0204
	windowMiddleButtonDown = 0x0207
	windowMiddleButtonUp   = 0x0208
	windowXButtonDown      = 0x020b
	windowXButtonUp        = 0x020c
	windowLeftButtonUp     = 0x0202
	windowRightButtonUp    = 0x0205
	hotkeyModifierAlt      = 0x0001
	hotkeyModifierControl  = 0x0002
	hotkeyModifierShift    = 0x0004
	hotkeyModifierWindows  = 0x0008
	hotkeyModifierNoRepeat = 0x4000
)

var (
	procSetWindowsHookEx    = user32.NewProc("SetWindowsHookExW")
	procUnhookWindowsHookEx = user32.NewProc("UnhookWindowsHookEx")
	procCallNextHookEx      = user32.NewProc("CallNextHookEx")
	procRegisterHotKey      = user32.NewProc("RegisterHotKey")
	procUnregisterHotKey    = user32.NewProc("UnregisterHotKey")
	procGetDoubleClickTime  = user32.NewProc("GetDoubleClickTime")
	procGetModuleHandle     = windows.NewLazySystemDLL("kernel32.dll").NewProc("GetModuleHandleW")
)

type lowLevelKeyboardData struct {
	virtualKey uint32
	scanCode   uint32
	flags      uint32
	time       uint32
	extraInfo  uintptr
}

type lowLevelMouseData struct {
	pointX    int32
	pointY    int32
	mouseData uint32
	flags     uint32
	time      uint32
	extraInfo uintptr
}

type windowsHotkeys struct {
	server            *helperServer
	loop              *windowsEventLoop
	recognizer        *hotkeyRecognizer
	keyboardHook      uintptr
	mouseHook         uintptr
	keyboardCallback  uintptr
	mouseCallback     uintptr
	registered        map[int]string
	physicalKeys      map[uint16]bool
	physicalModifiers map[modifierMask]bool
	timer             *time.Timer
	recording         *windowsHotkeyRecording
	recordGeneration  uint64
	events            int
	maxCallbackMicros float64
	lastError         string
	outgoing          chan proto.HotkeyEventParams
	done              chan struct{}
	closeOnce         sync.Once
}

func newWindowsHotkeys(server *helperServer) *windowsHotkeys {
	manager := &windowsHotkeys{
		server:            server,
		loop:              server.events,
		recognizer:        newHotkeyRecognizer(),
		registered:        make(map[int]string),
		physicalKeys:      make(map[uint16]bool),
		physicalModifiers: make(map[modifierMask]bool),
		outgoing:          make(chan proto.HotkeyEventParams, 256),
		done:              make(chan struct{}),
	}
	manager.keyboardCallback = syscall.NewCallback(manager.keyboardHookCallback)
	manager.mouseCallback = syscall.NewCallback(manager.mouseHookCallback)
	server.events.hotkeys = manager
	go manager.writeEvents()
	return manager
}

func (manager *windowsHotkeys) writeEvents() {
	for {
		select {
		case event := <-manager.outgoing:
			manager.server.notify(proto.MethodHotkeyEvent, event)
		case <-manager.done:
			return
		}
	}
}

func (manager *windowsHotkeys) configure(params proto.HotkeyConfigureParams) (proto.HotkeyStatus, error) {
	doubleClickTicks, _, _ := procGetDoubleClickTime.Call()
	doubleClickMS := float64(doubleClickTicks)
	if doubleClickMS == 0 {
		doubleClickMS = 500
	}
	doubleClickMS = min(doubleClickMS, 800)
	config, err := makeRecognizerConfig(params, doubleClickMS)
	if err != nil {
		return proto.HotkeyStatus{}, err
	}
	var status proto.HotkeyStatus
	manager.loop.callAndWait(func() {
		manager.configureOnThread(config)
		status = manager.statusOnThread()
	})
	return status, nil
}

func (manager *windowsHotkeys) configureOnThread(config recognizerConfig) {
	wasInstalled := manager.installedOnThread()
	manager.uninstallOnThread()
	manager.recognizer.configure(config)
	if len(config.bindings) == 0 && config.cancel == nil {
		manager.recognizer.reset()
	} else {
		manager.installOnThread()
		manager.armTimer()
	}
	if manager.recording != nil && !manager.ensureRecordingHooks() {
		manager.finishRecording(hotkeyRecordOutcome{err: deniedRecordError()})
	}
	if wasInstalled != manager.installedOnThread() {
		manager.server.notify(proto.MethodHotkeyState, manager.statusOnThread())
	}
}

func (manager *windowsHotkeys) installOnThread() {
	manager.lastError = ""
	nextID := 1
	needsKeyboard := manager.recognizer.config.hasModifierChords() || manager.recognizer.config.cancel != nil
	for index := range manager.recognizer.config.bindings {
		binding := &manager.recognizer.config.bindings[index]
		if !eligibleForRegisteredHotkey(binding) {
			needsKeyboard = true
			continue
		}
		modifiers := registeredModifierFlags(binding.Modifiers)
		result, _, callErr := procRegisterHotKey.Call(0, uintptr(nextID), uintptr(modifiers|hotkeyModifierNoRepeat), uintptr(binding.virtualKey))
		if result == 0 {
			needsKeyboard = true
			manager.lastError = fmt.Sprintf("RegisterHotKey for %s failed: %v; using the low-level hook", binding.ID, callErr)
			continue
		}
		manager.registered[nextID] = binding.ID
		nextID++
	}
	if needsKeyboard {
		module, _, _ := procGetModuleHandle.Call(0)
		hook, _, callErr := procSetWindowsHookEx.Call(hotkeyKeyboardLL, manager.keyboardCallback, module, 0)
		if hook == 0 {
			manager.lastError = fmt.Sprintf("SetWindowsHookEx(keyboard): %v", callErr)
		} else {
			manager.keyboardHook = hook
		}
		hook, _, callErr = procSetWindowsHookEx.Call(hotkeyMouseLL, manager.mouseCallback, module, 0)
		if hook == 0 {
			if manager.lastError == "" {
				manager.lastError = fmt.Sprintf("SetWindowsHookEx(mouse): %v", callErr)
			}
		} else {
			manager.mouseHook = hook
		}
	}
}

func (manager *windowsHotkeys) uninstallOnThread() {
	if manager.timer != nil {
		manager.timer.Stop()
		manager.timer = nil
	}
	for id := range manager.registered {
		procUnregisterHotKey.Call(0, uintptr(id))
	}
	manager.registered = make(map[int]string)
	if manager.keyboardHook != 0 {
		procUnhookWindowsHookEx.Call(manager.keyboardHook)
		manager.keyboardHook = 0
	}
	if manager.mouseHook != 0 {
		procUnhookWindowsHookEx.Call(manager.mouseHook)
		manager.mouseHook = 0
	}
}

func (manager *windowsHotkeys) close() {
	manager.closeOnce.Do(func() {
		if manager.recording != nil {
			manager.finishRecording(hotkeyRecordOutcome{err: recordError(proto.CodeCancelled, "Cancelled")})
		}
		manager.uninstallOnThread()
		close(manager.done)
	})
}

func (manager *windowsHotkeys) status() proto.HotkeyStatus {
	var status proto.HotkeyStatus
	manager.loop.callAndWait(func() { status = manager.statusOnThread() })
	return status
}

func (manager *windowsHotkeys) statusOnThread() proto.HotkeyStatus {
	return proto.HotkeyStatus{
		Installed:         manager.installedOnThread(),
		Accessibility:     true,
		Bindings:          len(manager.recognizer.config.bindings),
		Events:            manager.events,
		MissedEvents:      manager.recognizer.missedEvents,
		MaxCallbackMicros: manager.maxCallbackMicros,
		Error:             manager.lastError,
	}
}

func (manager *windowsHotkeys) installedOnThread() bool {
	return manager.keyboardHook != 0 || manager.mouseHook != 0 || len(manager.registered) > 0
}

func eligibleForRegisteredHotkey(binding *recognizedBinding) bool {
	if binding.Key == nil || binding.Behavior != proto.BehaviorTap || binding.Swallow == nil || *binding.Swallow {
		return false
	}
	for family, side := range binding.Modifiers {
		if family == proto.ModFn || side != proto.SideAny {
			return false
		}
	}
	return true
}

func registeredModifierFlags(modifiers map[string]string) uint32 {
	var flags uint32
	for family := range modifiers {
		switch family {
		case proto.ModOption:
			flags |= hotkeyModifierAlt
		case proto.ModControl:
			flags |= hotkeyModifierControl
		case proto.ModShift:
			flags |= hotkeyModifierShift
		case proto.ModCommand:
			flags |= hotkeyModifierWindows
		}
	}
	return flags
}

func (manager *windowsHotkeys) registeredHotkey(id int) {
	if manager.recording != nil {
		return
	}
	bindingID, found := manager.registered[id]
	if !found {
		return
	}
	manager.events++
	for index := range manager.recognizer.config.bindings {
		binding := &manager.recognizer.config.bindings[index]
		if binding.ID == bindingID {
			manager.queueEvents([]hotkeyFired{fire(binding, proto.PhaseTrigger, 0, manager.nowMS())})
			return
		}
	}
}

func (manager *windowsHotkeys) registeredMatch(press hotkeyPress) bool {
	if manager.recognizer.config.active && manager.recognizer.config.cancel != nil && bindingMatchesPress(manager.recognizer.config.cancel, press, manager.recognizer.pressed) {
		return false
	}
	for _, bindingID := range manager.registered {
		for index := range manager.recognizer.config.bindings {
			binding := &manager.recognizer.config.bindings[index]
			if binding.ID == bindingID && bindingMatchesPress(binding, press, manager.recognizer.pressed) {
				return true
			}
		}
	}
	return false
}

func (manager *windowsHotkeys) keyboardHookCallback(code int32, message uintptr, dataPointer unsafe.Pointer) uintptr {
	started := time.Now()
	if code < 0 || dataPointer == nil {
		return manager.callNext(code, message, dataPointer)
	}
	data := (*lowLevelKeyboardData)(dataPointer)
	manager.events++
	if data.extraInfo == helperInputMarker {
		return manager.callNext(code, message, dataPointer)
	}
	virtualKey := uint16(data.virtualKey)
	down := message == windowKeyDown || message == windowSystemKeyDown
	up := message == windowKeyUp || message == windowSystemKeyUp
	if !down && !up {
		return manager.callNext(code, message, dataPointer)
	}
	if manager.recording != nil {
		recorded := manager.recording.chords.keyboard(virtualKey, down)
		if recorded.finished {
			manager.finishRecording(hotkeyRecordOutcome{keys: recorded.chord})
		}
		if recorded.swallow {
			return 1
		}
		return manager.callNext(code, message, dataPointer)
	}
	var result recognizerResult
	if modifier, isModifier := modifierForVirtualKey(virtualKey); isModifier {
		wasDown := manager.physicalModifiers[modifier]
		if down && wasDown {
			return manager.callNext(code, message, dataPointer)
		}
		manager.physicalModifiers[modifier] = down
		pressed := manager.recognizer.pressed
		if down {
			pressed |= modifier
		} else {
			pressed &^= modifier
		}
		result = manager.recognizer.modifier(virtualKey, pressed, manager.nowMS())
	} else if down {
		repeat := manager.physicalKeys[virtualKey]
		manager.physicalKeys[virtualKey] = true
		press := hotkeyPress{virtualKey: virtualKey}
		if manager.registeredMatch(press) {
			result.events = manager.recognizer.tick(manager.nowMS())
			if manager.recognizer.pressed != 0 {
				manager.recognizer.spoiled = true
				manager.recognizer.longDeadline = nil
			}
		} else {
			result = manager.recognizer.keyDown(press, repeat, manager.nowMS())
		}
	} else {
		delete(manager.physicalKeys, virtualKey)
		result = manager.recognizer.keyUp(hotkeyPress{virtualKey: virtualKey}, manager.nowMS())
	}
	manager.finishCallback(result, started)
	if result.swallow {
		return 1
	}
	return manager.callNext(code, message, dataPointer)
}

func (manager *windowsHotkeys) mouseHookCallback(code int32, message uintptr, dataPointer unsafe.Pointer) uintptr {
	started := time.Now()
	if code < 0 || dataPointer == nil {
		return manager.callNext(code, message, dataPointer)
	}
	data := (*lowLevelMouseData)(dataPointer)
	if data.extraInfo == helperInputMarker {
		return manager.callNext(code, message, dataPointer)
	}
	press, down, up, ok := mousePress(message, data.mouseData)
	if !ok {
		return manager.callNext(code, message, dataPointer)
	}
	if manager.recording != nil {
		recorded := manager.recording.chords.mouse(press.button, down)
		if recorded.finished {
			manager.finishRecording(hotkeyRecordOutcome{keys: recorded.chord})
		}
		if recorded.swallow {
			return 1
		}
		return manager.callNext(code, message, dataPointer)
	}
	var result recognizerResult
	if down {
		result = manager.recognizer.keyDown(press, false, manager.nowMS())
	} else if up {
		result = manager.recognizer.keyUp(press, manager.nowMS())
	}
	manager.finishCallback(result, started)
	if result.swallow {
		return 1
	}
	return manager.callNext(code, message, dataPointer)
}

func mousePress(message uintptr, mouseData uint32) (hotkeyPress, bool, bool, bool) {
	switch message {
	case windowLeftButtonDown:
		return hotkeyPress{button: 0, mouse: true}, true, false, true
	case windowRightButtonDown:
		return hotkeyPress{button: 1, mouse: true}, true, false, true
	case windowMiddleButtonDown:
		return hotkeyPress{button: 2, mouse: true}, true, false, true
	case windowXButtonDown:
		return hotkeyPress{button: 2 + int(mouseData>>16), mouse: true}, true, false, true
	case windowLeftButtonUp:
		return hotkeyPress{button: 0, mouse: true}, false, true, true
	case windowRightButtonUp:
		return hotkeyPress{button: 1, mouse: true}, false, true, true
	case windowMiddleButtonUp:
		return hotkeyPress{button: 2, mouse: true}, false, true, true
	case windowXButtonUp:
		return hotkeyPress{button: 2 + int(mouseData>>16), mouse: true}, false, true, true
	default:
		return hotkeyPress{}, false, false, false
	}
}

func (manager *windowsHotkeys) finishCallback(result recognizerResult, started time.Time) {
	manager.queueEvents(result.events)
	manager.armTimer()
	micros := float64(time.Since(started).Microseconds())
	manager.maxCallbackMicros = max(manager.maxCallbackMicros, micros)
}

func (manager *windowsHotkeys) armTimer() {
	if manager.timer != nil {
		manager.timer.Stop()
		manager.timer = nil
	}
	deadline, found := manager.recognizer.nextDeadline()
	if !found {
		return
	}
	delay := time.Duration(max(0, deadline-manager.nowMS()) * float64(time.Millisecond))
	manager.timer = time.AfterFunc(delay, func() {
		select {
		case <-manager.done:
			return
		default:
		}
		manager.loop.call(func() {
			manager.queueEvents(manager.recognizer.tick(manager.nowMS()))
			manager.armTimer()
		})
	})
}

func (manager *windowsHotkeys) queueEvents(events []hotkeyFired) {
	if len(events) == 0 {
		return
	}
	front := manager.server.cachedFront()
	var frontPointer *proto.App
	if front.PID != 0 {
		frontPointer = &front
	}
	for _, event := range events {
		params := proto.HotkeyEventParams{
			Binding: event.binding, Phase: event.phase, Behavior: event.behavior,
			HeldMS: event.heldMS, T: event.timeMS, Front: frontPointer,
		}
		select {
		case manager.outgoing <- params:
		case <-manager.done:
			return
		}
	}
}

func (manager *windowsHotkeys) nowMS() float64 {
	return float64(time.Since(manager.server.started).Microseconds())/1000 + 1
}

func (manager *windowsHotkeys) callNext(code int32, message uintptr, data unsafe.Pointer) uintptr {
	result, _, _ := procCallNextHookEx.Call(0, uintptr(code), message, uintptr(data))
	return result
}
