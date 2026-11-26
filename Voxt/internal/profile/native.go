package profile

import (
	"fmt"
	"os"
	"strings"
	"sync"
	"time"
)

// NativeLog saves omni.Config.Log without sharing the app's log files.
type NativeLog struct {
	mu       sync.Mutex
	file     *os.File
	recorder *Recorder
}

func OpenNativeLog(path string, recorder *Recorder) (*NativeLog, error) {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		return nil, err
	}
	return &NativeLog{file: file, recorder: recorder}, nil
}

func (l *NativeLog) Write(level, message string) {
	message = l.recorder.Clean(message)
	l.mu.Lock()
	defer l.mu.Unlock()
	_, _ = fmt.Fprintf(l.file, "%s %s %s\n", time.Now().UTC().Format(time.RFC3339Nano), level, message)
	if level == "error" || level == "warning" && invalidNativeWarning(message) {
		l.recorder.Warn("native " + level + ": " + message)
	}
}

func invalidNativeWarning(message string) bool {
	// Native stderr is labelled warning even for routine timing lines; only
	// failures that invalidate the requested run belong in report.Warnings.
	for _, problem := range []string{"starting without it", "does not know --cache-limit-mb", "stalled", "exited unexpectedly", "did not start"} {
		if strings.Contains(message, problem) {
			return true
		}
	}
	return false
}

func (l *NativeLog) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.file.Close()
}
