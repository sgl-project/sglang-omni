// Package notify is the notification centre: core.NotificationService in
// memory. Posting stores the notification (newest first, at most Max) and
// publishes core.NotificationPosted; the shell shows the notifications as
// a stack of cards at the foot of the sidebar (a dot on the rail's
// Settings icon), where each is dismissed.
//
// A centre opened over Documents (Open) keeps its info, success and
// warning notifications across restarts, until they are cleared; errors
// belong to the session that had them and are not kept. Saving happens on
// a goroutine after each change (the newest list wins); Close waits for it.
//
// Imports: core.
package notify

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"sync"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
)

// Max is how many notifications the centre keeps; older ones drop off.
const Max = 200

// Center implements core.NotificationService.
type Center struct {
	bus  *core.Bus
	now  func() time.Time
	docs Documents // nil: memory only

	mu    sync.Mutex
	items []core.Notification // newest first

	saveMu sync.Mutex // serializes saves
	saving sync.WaitGroup
}

// Documents is where a centre keeps its notifications; store.Documents
// satisfies it.
type Documents interface {
	Get(ctx context.Context, key string) ([]byte, bool, error)
	Put(ctx context.Context, key string, doc []byte) error
}

// DocKey is the document the notifications are kept in.
const DocKey = "notifications"

// Open returns a centre that keeps its notifications in docs, with the
// ones kept last time. A document that does not decode is ignored (and
// replaced at the next change): Open then returns the empty centre and
// the error, for the log.
func Open(ctx context.Context, docs Documents, bus *core.Bus) (*Center, error) {
	c := New(bus)
	c.docs = docs
	data, ok, err := docs.Get(ctx, DocKey)
	if err != nil || !ok {
		return c, err
	}
	var items []core.Notification
	if err := json.Unmarshal(data, &items); err != nil {
		return c, fmt.Errorf("notifications: %w", err)
	}
	c.items = items
	return c, nil
}

// kept reports whether n survives a restart.
func kept(n core.Notification) bool { return n.Severity != core.SeverityError }

// save writes the kept notifications in the background; it never blocks.
func (c *Center) save() {
	if c.docs == nil {
		return
	}
	c.saving.Add(1)
	go func() {
		defer c.saving.Done()
		c.saveMu.Lock()
		defer c.saveMu.Unlock()
		c.mu.Lock()
		items := slices.DeleteFunc(slices.Clone(c.items), func(n core.Notification) bool { return !kept(n) })
		c.mu.Unlock()
		if items == nil {
			items = []core.Notification{}
		}
		data, err := json.Marshal(items)
		if err == nil {
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			c.docs.Put(ctx, DocKey, data)
			cancel()
		}
	}()
}

// Close waits for pending saves (call it before closing the store).
func (c *Center) Close() { c.saving.Wait() }

var _ core.NotificationService = (*Center)(nil)

// New returns an empty centre publishing on bus (which may be nil).
func New(bus *core.Bus) *Center { return &Center{bus: bus, now: time.Now} }

// Post stores n, filling its ID and Time when empty, and publishes
// NotificationPosted. A notification with the ID of one stored replaces it
// (as unread), so a repeated problem shows once. It never blocks.
func (c *Center) Post(n core.Notification) {
	if n.ID == "" {
		n.ID = core.NewID(c.now())
	}
	if n.Time.IsZero() {
		n.Time = c.now()
	}
	if n.Severity == "" {
		n.Severity = core.SeverityInfo
	}
	n.Read = false
	c.mu.Lock()
	c.items = slices.DeleteFunc(c.items, func(x core.Notification) bool { return x.ID == n.ID })
	c.items = slices.Insert(c.items, 0, n)
	if len(c.items) > Max {
		c.items = c.items[:Max]
	}
	c.mu.Unlock()
	if kept(n) {
		c.save()
	}
	c.bus.Publish(core.NotificationPosted{Notification: n})
}

// List returns the notifications, newest first.
func (c *Center) List(context.Context) ([]core.Notification, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return slices.Clone(c.items), nil
}

// MarkRead marks one notification read; "" marks them all.
func (c *Center) MarkRead(_ context.Context, id string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	found := id == ""
	for i := range c.items {
		if id == "" || c.items[i].ID == id {
			c.items[i].Read = true
			found = true
		}
	}
	if !found {
		return fmt.Errorf("notification %s: %w", id, core.ErrNotFound)
	}
	c.save()
	return nil
}

// Dismiss removes one notification; one not there is no error (a
// problem the shell withdraws may be gone already).
func (c *Center) Dismiss(_ context.Context, id string) error {
	c.mu.Lock()
	n := len(c.items)
	c.items = slices.DeleteFunc(c.items, func(x core.Notification) bool { return x.ID == id })
	gone := len(c.items) != n
	c.mu.Unlock()
	if gone {
		c.save()
	}
	return nil
}

// Clear removes every notification.
func (c *Center) Clear(context.Context) error {
	c.mu.Lock()
	c.items = nil
	c.mu.Unlock()
	c.save()
	return nil
}
