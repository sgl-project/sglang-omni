package core

import (
	"context"
	"time"
)

// Severity grades notifications and log lines.
type Severity string

const (
	SeverityDebug   Severity = "debug"
	SeverityInfo    Severity = "info"
	SeveritySuccess Severity = "success"
	SeverityWarning Severity = "warning"
	SeverityError   Severity = "error"
)

// Notification is a message for the notification centre (and a toast).
type Notification struct {
	ID       string    `json:"id"`
	Severity Severity  `json:"severity"`
	Title    string    `json:"title"`
	Body     string    `json:"body,omitempty"`
	Link     string    `json:"link,omitempty"` // router path or voxtnew:// deep link, e.g. "/settings/models"
	Time     time.Time `json:"time"`
	Read     bool      `json:"read,omitempty"`
}

// LogEntry is one line of the logs panel: from the app or from the helper
// (its "log" notifications and stderr).
type LogEntry struct {
	Time    time.Time `json:"time"`
	Level   Severity  `json:"level"`
	Source  string    `json:"source"` // "app", "helper"…
	Message string    `json:"message"`
	// Seq numbers the lines a log appended, from 1: a line's identity in
	// the logs view (its selection) while lines come and go.
	Seq uint64 `json:"-"`
}

// NotificationService is the notification centre. Post never blocks.
type NotificationService interface {
	Post(n Notification)
	List(ctx context.Context) ([]Notification, error)
	MarkRead(ctx context.Context, id string) error
	Dismiss(ctx context.Context, id string) error
	Clear(ctx context.Context) error
}

// LogService is the in-memory ring of log lines behind the logs panel,
// also written to rotated files in mygo.PathLogs. Append never blocks.
type LogService interface {
	Append(e LogEntry)
	// Tail returns up to n newest entries from source ("" = all), oldest first.
	Tail(n int, source string) []LogEntry
}
