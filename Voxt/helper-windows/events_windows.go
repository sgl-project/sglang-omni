//go:build windows

package main

import (
	"fmt"
	"os"
	"runtime"
	"sync"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

const (
	eventSystemForeground = 0x0003
	winEventOutOfContext  = 0x0000
	windowMessageQuit     = 0x0012
	windowMessageApp      = 0x8000
	eventLoopRunAction    = windowMessageApp + 1
)

var (
	procGetCurrentThreadID = windows.NewLazySystemDLL("kernel32.dll").NewProc("GetCurrentThreadId")
	procGetMessage         = user32.NewProc("GetMessageW")
	procPostThreadMessage  = user32.NewProc("PostThreadMessageW")
	procSetWinEventHook    = user32.NewProc("SetWinEventHook")
	procUnhookWinEvent     = user32.NewProc("UnhookWinEvent")
)

type windowMessage struct {
	windowHandle uintptr
	message      uint32
	wParam       uintptr
	lParam       uintptr
	time         uint32
	pointX       int32
	pointY       int32
	private      uint32
}

type windowsEventLoop struct {
	threadID    uint32
	foreground  uintptr
	actions     chan func()
	done        chan struct{}
	closeOnce   sync.Once
	callback    uintptr
	startupDone chan error
	hotkeys     *windowsHotkeys
}

func startWindowsEventLoop(onForeground func(uintptr)) (*windowsEventLoop, error) {
	loop := &windowsEventLoop{
		actions:     make(chan func(), 16),
		done:        make(chan struct{}),
		startupDone: make(chan error, 1),
	}
	go loop.run(onForeground)
	if err := <-loop.startupDone; err != nil {
		return nil, err
	}
	return loop, nil
}

func (loop *windowsEventLoop) run(onForeground func(uintptr)) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	defer close(loop.done)
	threadID, _, _ := procGetCurrentThreadID.Call()
	loop.threadID = uint32(threadID)
	loop.callback = syscall.NewCallback(func(_ uintptr, windowHandle uintptr, _ uint32, _ uintptr, _ uint32, _ uint32, _ uint32) uintptr {
		if onForeground != nil {
			onForeground(windowHandle)
		}
		return 0
	})
	hook, _, callErr := procSetWinEventHook.Call(
		eventSystemForeground,
		eventSystemForeground,
		0,
		loop.callback,
		0,
		0,
		winEventOutOfContext,
	)
	if hook == 0 {
		loop.startupDone <- fmt.Errorf("SetWinEventHook: %w", callErr)
		return
	}
	loop.foreground = hook
	loop.startupDone <- nil
	defer procUnhookWinEvent.Call(loop.foreground)
	defer func() {
		if loop.hotkeys != nil {
			loop.hotkeys.close()
		}
	}()

	for {
		var message windowMessage
		result, _, callErr := procGetMessage.Call(uintptr(unsafe.Pointer(&message)), 0, 0, 0)
		switch int32(result) {
		case -1:
			fmt.Fprintln(os.Stderr, "GetMessageW:", callErr)
			return
		case 0:
			return
		}
		if message.message == hotkeyWindowMessage && loop.hotkeys != nil {
			loop.hotkeys.registeredHotkey(int(message.wParam))
		}
		if message.message == eventLoopRunAction {
			for {
				select {
				case action := <-loop.actions:
					action()
				default:
					goto actionsDone
				}
			}
		actionsDone:
		}
	}
}

func (loop *windowsEventLoop) call(action func()) {
	select {
	case <-loop.done:
		return
	case loop.actions <- action:
	}
	result, _, callErr := procPostThreadMessage.Call(uintptr(loop.threadID), eventLoopRunAction, 0, 0)
	if result == 0 {
		fmt.Fprintln(os.Stderr, "PostThreadMessageW:", callErr)
	}
}

func (loop *windowsEventLoop) callAndWait(action func()) bool {
	executed := make(chan struct{})
	loop.call(func() {
		action()
		close(executed)
	})
	select {
	case <-executed:
		return true
	case <-loop.done:
		return false
	}
}

func (loop *windowsEventLoop) close() {
	loop.closeOnce.Do(func() {
		result, _, _ := procPostThreadMessage.Call(uintptr(loop.threadID), windowMessageQuit, 0, 0)
		if result == 0 {
			return
		}
		<-loop.done
	})
}
