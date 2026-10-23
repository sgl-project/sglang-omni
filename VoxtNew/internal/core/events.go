package core

// Events of the shell. Other packages define their own next to the code
// that publishes them, the topic prefix naming the owner:
//
//	settings.Changed       "settings.changed"  internal/settings
//	helper.StatusChanged   "helper.status"     internal/helper

// NotificationPosted: something for the notification centre and a toast.
type NotificationPosted struct{ Notification Notification }

// LogAppended: a line for the logs panel.
type LogAppended struct{ Entry LogEntry }

func (NotificationPosted) Topic() string { return "notification.posted" }
func (LogAppended) Topic() string        { return "log.appended" }
