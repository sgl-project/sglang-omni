package notify

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
)

func TestCenter(t *testing.T) {
	ctx := context.Background()
	bus := core.NewBus()
	var posted []core.Notification
	core.On(bus, func(e core.NotificationPosted) { posted = append(posted, e.Notification) })
	c := New(bus)
	c.Post(core.Notification{Title: "first"})
	c.Post(core.Notification{ID: "disk", Severity: core.SeverityWarning, Title: "Disk low"})
	c.Post(core.Notification{Title: "third"})
	if len(posted) != 3 || posted[0].ID == "" || posted[0].Time.IsZero() || posted[0].Severity != core.SeverityInfo {
		t.Fatalf("posted %+v", posted)
	}
	list, _ := c.List(ctx)
	if len(list) != 3 || list[0].Title != "third" || list[2].Title != "first" {
		t.Fatalf("list %+v", list)
	}
	if err := c.MarkRead(ctx, "disk"); err != nil {
		t.Fatal(err)
	}
	// The same ID again replaces it, unread, on top.
	c.Post(core.Notification{ID: "disk", Severity: core.SeverityWarning, Title: "Disk still low"})
	list, _ = c.List(ctx)
	if len(list) != 3 || list[0].Title != "Disk still low" || list[0].Read {
		t.Fatalf("list %+v", list)
	}
	if err := c.MarkRead(ctx, "nope"); !errors.Is(err, core.ErrNotFound) {
		t.Error(err)
	}
	c.MarkRead(ctx, "")
	list, _ = c.List(ctx)
	for _, n := range list {
		if !n.Read {
			t.Error("unread after MarkRead(\"\")", n)
		}
	}
	// Dismiss removes one; again, or one never there, is no error.
	if err := c.Dismiss(ctx, "disk"); err != nil {
		t.Fatal(err)
	}
	if err := c.Dismiss(ctx, "disk"); err != nil {
		t.Error(err)
	}
	if list, _ = c.List(ctx); len(list) != 2 || list[0].Title != "third" {
		t.Fatalf("after Dismiss %+v", list)
	}
	for range Max + 5 {
		c.Post(core.Notification{Title: "x"})
	}
	if list, _ = c.List(ctx); len(list) != Max {
		t.Errorf("kept %d", len(list))
	}
	c.Clear(ctx)
	if list, _ = c.List(ctx); len(list) != 0 {
		t.Error("not cleared")
	}
}

type memDocs struct {
	mu sync.Mutex
	m  map[string][]byte
}

func (d *memDocs) Get(_ context.Context, key string) ([]byte, bool, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	b, ok := d.m[key]
	return b, ok, nil
}

func (d *memDocs) Put(_ context.Context, key string, doc []byte) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.m[key] = doc
	return nil
}

// Info, success and warning notifications survive a restart; errors do not.
func TestCenterPersists(t *testing.T) {
	ctx := context.Background()
	docs := &memDocs{m: map[string][]byte{}}
	c, err := Open(ctx, docs, nil)
	if err != nil {
		t.Fatal(err)
	}
	c.Post(core.Notification{Title: "info"})
	c.Post(core.Notification{Severity: core.SeverityError, Title: "error"})
	c.Post(core.Notification{ID: "w", Severity: core.SeverityWarning, Title: "warning"})
	c.MarkRead(ctx, "w")
	c.Close()

	c2, err := Open(ctx, docs, nil)
	if err != nil {
		t.Fatal(err)
	}
	list, _ := c2.List(ctx)
	if len(list) != 2 || list[0].Title != "warning" || !list[0].Read || list[1].Title != "info" {
		t.Fatalf("after restart %+v", list)
	}
	c2.Dismiss(ctx, "w")
	c2.Close()
	c2b, _ := Open(ctx, docs, nil)
	if list, _ := c2b.List(ctx); len(list) != 1 || list[0].Title != "info" {
		t.Fatalf("dismissed, yet %+v", list)
	}
	c2.Clear(ctx)
	c2.Close()
	c3, _ := Open(ctx, docs, nil)
	if list, _ := c3.List(ctx); len(list) != 0 {
		t.Errorf("cleared, yet %+v", list)
	}
	docs.m[DocKey] = []byte("{broken")
	if _, err := Open(ctx, docs, nil); err == nil {
		t.Error("a broken document opened silently")
	}
}
