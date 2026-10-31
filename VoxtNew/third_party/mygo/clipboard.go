package mygo

import (
	"errors"
	"slices"

	"github.com/egoist/mygo/internal/platform"
	"github.com/egoist/mygo/transfer"
)

// ClipboardOptions configures the lifetime of a clipboard write.
type ClipboardOptions struct {
	// OnRelease runs once on the main thread after native providers are no
	// longer needed: replacement, Clear, Flush, or application shutdown.
	// The system may also finish providers after fetching all their data.
	// It is scheduled after the native callback returns and may safely write
	// a new clipboard. It does not run when Write fails. Closing a window
	// does not release the application-owned clipboard.
	OnRelease func()
}

// ErrClipboardReentrant means a provider tried to access the clipboard.
// Providers may return prepared bytes, but must not read or replace the
// clipboard while the system is requesting their representation.
var ErrClipboardReentrant = errors.New("mygo: clipboard access from a clipboard provider")

// ErrClipboardChanged means ownership changed while a read was in progress.
var ErrClipboardChanged = platform.ErrClipboardChanged

// ErrClipboardPersistence means the desktop cannot store clipboard data
// after the owning application exits. Providers have still been materialized.
var ErrClipboardPersistence = platform.ErrClipboardPersistence

// Clipboard state is main-thread only. User release hooks are deferred until
// native callbacks unwind, including callbacks made during Write itself.
type clipboardLifetime struct {
	armed, released, notified bool
	onRelease                 func()
}

func (l *clipboardLifetime) release() {
	l.released = true
	if l.armed && !l.notified {
		l.notified = true
		if fn := l.onRelease; fn != nil {
			l.onRelease = nil
			if clipboardOperations != 0 {
				clipboardReleases = append(clipboardReleases, fn)
			} else {
				queueClipboardRelease(fn)
			}
		}
	}
}

var clipboardProviding, clipboardStopped bool
var clipboardOperations int
var clipboardReleases []func()

func queueClipboardRelease(fn func()) {
	if !postMain(fn) {
		// Native shutdown can end a nested operation before it unwinds.
		// The loop can no longer deliver hooks, but their resources still
		// need cleanup. Clipboard methods already reject stopped-loop work.
		fn()
	}
}

// Native reads/storage may pump nested event loops (notably GTK store).
// Release hooks wait for the complete outer clipboard operation, so a hook
// replacing the clipboard cannot interfere with an unfinished native flush.
func withClipboardOperation(fn func()) {
	clipboardOperations++
	defer func() {
		clipboardOperations--
		if clipboardOperations == 0 {
			pending := clipboardReleases
			clipboardReleases = nil
			for _, release := range pending {
				queueClipboardRelease(release)
			}
		}
	}()
	fn()
}

func clipboardNativeValue[T any](fn func() T) T {
	return onMainValue(func() T {
		var value T
		if !clipboardProviding && !clipboardStopped {
			withClipboardOperation(func() { value = fn() })
		}
		return value
	})
}

func clipboardOnMain(fn func() error) error {
	err := errLoopStopped
	onMain(func() {
		if clipboardStopped {
			return
		}
		withClipboardOperation(func() { err = fn() })
	})
	return err
}

// Write replaces the clipboard with items offering alternative serialized
// representations. It takes a fresh provider snapshot without invoking any
// providers. Providers run on the main thread at most once per write; errors
// and panics are cached. Native receivers can request them until ownership
// ends, even after the source window closes. Keep captured resources alive
// until OnRelease; prepare expensive data beforehand on a goroutine.
//
// Empty data clears the clipboard. At most one options value is allowed.
// It is safe from any goroutine. Go values from Drag(value) are never copied
// to the clipboard; offer an explicit encoding with transfer.Bytes or Lazy.
func (ClipboardModule) Write(data transfer.Data, options ...ClipboardOptions) error {
	needsApp("Clipboard.Write")
	if len(options) > 1 {
		panic("mygo: Clipboard.Write accepts at most one options value")
	}
	if slices.Contains(data.Formats(), platform.DragSessionFormat) {
		return errors.New("mygo: clipboard cannot contain a process-local drag session")
	}
	var opts ClipboardOptions
	if len(options) == 1 {
		opts = options[0]
	}
	return clipboardOnMain(func() error {
		if clipboardProviding {
			return ErrClipboardReentrant
		}
		l := &clipboardLifetime{onRelease: opts.OnRelease}
		// Guard native provider reentry without changing the shared data model.
		// Read() below also materializes this snapshot on the main thread.
		var items []transfer.Item
		for _, item := range data.Snapshot().Items() {
			if len(item.Formats()) == 0 {
				continue
			}
			var reps []transfer.Representation
			for _, f := range item.Formats() {
				reps = append(reps, transfer.Lazy(f, func() ([]byte, error) {
					clipboardProviding = true
					defer func() { clipboardProviding = false }()
					return item.Read(f)
				}))
			}
			items = append(items, transfer.NewItem(reps...))
		}
		err := backend().Clipboard().WriteData(transfer.New(items...), l.release)
		if err != nil {
			l.onRelease = nil
			return err
		}
		l.armed = true
		if l.released {
			l.release()
		}
		return nil
	})
}

// Read copies the requested representations from the current clipboard.
// With no formats it reads all advertised representations; an empty
// clipboard returns empty data. Explicit formats select alternatives without
// invoking other providers; no matching representation is transfer.ErrFormat.
// The returned data has no native providers and is safe to keep, read from
// any goroutine, or offer in a later drag. Ownership changes during a native
// read return an error, rather than mixing contents from different owners.
func (ClipboardModule) Read(formats ...transfer.Format) (transfer.Data, error) {
	needsApp("Clipboard.Read")
	var data transfer.Data
	err := clipboardOnMain(func() error {
		if clipboardProviding {
			return ErrClipboardReentrant
		}
		d, err := backend().Clipboard().ReadData(slices.Clone(formats))
		data = d
		return err
	})
	return data, err
}

// Formats discovers portable format names without requesting provider data.
// Use transfer.PreferredFormat to negotiate in receiver preference order.
// The older AvailableFormats method keeps its platform-specific naming convention.
func (ClipboardModule) Formats() []transfer.Format {
	needsApp("Clipboard.Formats")
	return clipboardNativeValue(func() []transfer.Format { return backend().Clipboard().Formats() })
}

// ReadFormat copies one representation, coalescing text/file items as
// Data.Read does. It returns transfer.ErrFormat when it is unavailable.
func (c ClipboardModule) ReadFormat(format transfer.Format) ([]byte, error) {
	needsApp("Clipboard.ReadFormat")
	d, err := c.Read(format)
	if err != nil {
		return nil, err
	}
	return d.Read(format)
}

// ReadFiles returns local paths from the clipboard's file/URI list.
// Remote URLs are ignored. No file is opened, created, moved or deleted.
func (c ClipboardModule) ReadFiles() ([]string, error) {
	needsApp("Clipboard.ReadFiles")
	d, err := c.Read(transfer.FileList, transfer.URIList)
	if err != nil {
		return nil, err
	}
	return d.Files()
}

// WriteFiles copies absolute paths to the clipboard as native file URLs/lists.
func (c ClipboardModule) WriteFiles(paths ...string) error {
	needsApp("Clipboard.WriteFiles")
	d, err := transfer.FileData(paths...)
	if err != nil {
		return err
	}
	return c.Write(d)
}

// Flush requests all representations of MyGo's current clipboard and hands
// them to native storage, releasing lazy-provider resources on success.
// Provider errors leave ownership intact. It does nothing to another app's
// clipboard. App quitting calls it automatically, on a best-effort basis.
// Linux persistence requires a clipboard manager; without one Flush returns
// ErrClipboardPersistence after materializing providers. Window closure needs
// no flush.
func (ClipboardModule) Flush() error {
	needsApp("Clipboard.Flush")
	return clipboardOnMain(func() error {
		if clipboardProviding {
			return ErrClipboardReentrant
		}
		return backend().Clipboard().Flush()
	})
}
