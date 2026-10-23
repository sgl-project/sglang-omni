package platform

import "github.com/egoist/mygo/transfer"

// DragSessionFormat carries an unguessable, process-local registry key.
// It never contains a pointer. Only the core resolves it while a drag lives.
const DragSessionFormat transfer.Format = "application/x-mygo-drag-session"

// DragRequest starts a native drag on the main thread during a pointer
// gesture. Done must be called exactly once on the main thread. Windows
// may run a native nested event loop; other backends return immediately.
type DragRequest struct {
	Data                      transfer.Data
	Operations                transfer.Operation
	Session                   string
	X, Y                      float64
	Preview                   []byte // premultiplied BGRA, one pixel per DIP
	Width, Height, HotX, HotY int
	Done                      func(transfer.Result)
}

// DataDragEvent is a synchronous destination query. The core resolves
// Session to Local and the original Data only for a live local source.
// Content sets Operation and Formats to accept. Backends read those
// formats on drop, never while hovering, and copy the bytes into Data.
type DataDragEvent struct {
	Offer     transfer.Offer
	Session   string
	Local     any
	Data      transfer.Data
	Operation transfer.Operation
	Formats   []transfer.Format
	HasFiles  bool
	Resolved  bool // the core substituted its process-local data
}
