//go:build windows

package main

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"syscall"
	"time"

	"golang.org/x/sys/windows"
)

func main() {
	runtime.LockOSThread()
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	automation, err := newAutomation()
	if err != nil {
		return err
	}
	defer windows.CoUninitialize()
	defer automation.release()
	if err := initializeOLE(); err != nil {
		return err
	}
	defer procOleUninitialize.Call()

	var failures []error
	if err := probeNotepad(automation); err != nil {
		failures = append(failures, fmt.Errorf("notepad: %w", err))
	}
	if err := probeConsole(automation); err != nil {
		failures = append(failures, fmt.Errorf("console: %w", err))
	}
	return errors.Join(failures...)
}

func probeNotepad(automation *comObject) error {
	temporaryDirectory, err := os.MkdirTemp("", "voxt-uia-probe-")
	if err != nil {
		return err
	}
	fileName := fmt.Sprintf("voxt-uia-%d.txt", os.Getpid())
	filePath := filepath.Join(temporaryDirectory, fileName)
	if err := os.WriteFile(filePath, nil, 0o600); err != nil {
		os.RemoveAll(temporaryDirectory)
		return err
	}
	command := exec.Command("notepad.exe", filePath)
	if err := command.Start(); err != nil {
		os.RemoveAll(temporaryDirectory)
		return err
	}
	defer command.Process.Release()
	windowHandle, processID, err := waitForWindow(fileName, 20*time.Second)
	if err != nil {
		command.Process.Kill()
		command.Wait()
		os.RemoveAll(temporaryDirectory)
		return err
	}
	defer func() {
		procSetForegroundWindow.Call(windowHandle)
		if err := sendKeyWithControl(virtualKeyS); err != nil {
			fmt.Printf("notepad cleanup: Ctrl+S: %v\n", err)
		}
		time.Sleep(300 * time.Millisecond)
		closeWindow(windowHandle)
		if err := os.RemoveAll(temporaryDirectory); err != nil {
			fmt.Printf("notepad cleanup: %v\n", err)
		}
	}()
	return probeTarget(automation, "notepad", windowHandle, processID)
}

func probeConsole(automation *comObject) error {
	powerShellPath, err := exec.LookPath("powershell.exe")
	if err != nil {
		fmt.Println("TARGET console: skipped (powershell.exe is unavailable)")
		return nil
	}
	title := fmt.Sprintf("Voxt-UIA-Probe-Console-%d", os.Getpid())
	command := exec.Command(powerShellPath,
		"-NoLogo", "-NoProfile", "-NoExit", "-Command",
		fmt.Sprintf("$Host.UI.RawUI.WindowTitle = '%s'", title),
	)
	command.SysProcAttr = &syscall.SysProcAttr{CreationFlags: windows.CREATE_NEW_CONSOLE}
	if err := command.Start(); err != nil {
		return err
	}
	defer func() {
		command.Process.Kill()
		command.Wait()
	}()
	windowHandle, processID, err := waitForWindow(title, 15*time.Second)
	if err != nil {
		return err
	}
	return probeTarget(automation, "powershell console", windowHandle, processID)
}
