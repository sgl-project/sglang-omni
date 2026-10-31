//go:build windows && (amd64 || arm64)

package windows

import (
	"encoding/binary"
	"errors"
	"os"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"
	"unicode/utf16"
	"unsafe"

	"github.com/egoist/mygo/internal/platform"
	"github.com/egoist/mygo/internal/watchdriver"
)

// directoryWatch watches directories with ReadDirectoryChangesW, every
// directory's I/O completing on one completion port that the goroutine of
// the watch waits on. A Subtree target watches a whole tree with one
// handle: Windows doesn't let a directory be renamed while a handle is
// open on something beneath it, so a handle per directory would keep the
// directories of the tree from being renamed.
type directoryWatch struct {
	port     syscall.Handle
	mu       sync.Mutex // guards closed, for Wake
	closed   bool
	next     uint32
	sessions map[uint32]*directorySession // by key, until their I/O completes
	targets  map[uint64]*directorySession
}

type directorySession struct {
	key     uint32
	target  uint64
	handle  syscall.Handle
	subtree bool
	removed bool
	overlap syscall.Overlapped
	buffer  [16384]uint32 // 64 KiB, the most a network share takes, aligned as the records need
	pin     runtime.Pinner
}

// Completion keys: 0 wakes the goroutine, sessions count from 1.
const watchWakeKey = 0

// errorNotifyEnumDir is what ReadDirectoryChangesW fails with when more
// changed than its buffer could hold.
const errorNotifyEnumDir = syscall.Errno(1022)

// leakedSessions keeps the buffers of I/O that never completed after Close
// from being reused while Windows can still write to them.
var leakedSessions struct {
	sync.Mutex
	sessions []*directorySession
}

func (*Backend) NewFileWatch(notify func(platform.FileWatchNotice)) (platform.FileWatch, error) {
	port, err := syscall.CreateIoCompletionPort(syscall.InvalidHandle, 0, 0, 1)
	if err != nil {
		return nil, err
	}
	d := &directoryWatch{port: port, sessions: map[uint32]*directorySession{}, targets: map[uint64]*directorySession{}}
	return watchdriver.New(d, notify), nil
}

func (d *directoryWatch) Add(t platform.FileWatchTarget) error {
	if !t.Directory {
		return platform.ErrUnsupported // files are watched through their directory
	}
	path, err := syscall.UTF16PtrFromString(t.Path)
	if err != nil {
		return err
	}
	share := uint32(syscall.FILE_SHARE_READ | syscall.FILE_SHARE_WRITE | syscall.FILE_SHARE_DELETE)
	h, err := syscall.CreateFile(path, syscall.FILE_LIST_DIRECTORY, share, nil, syscall.OPEN_EXISTING, syscall.FILE_FLAG_BACKUP_SEMANTICS|syscall.FILE_FLAG_OVERLAPPED|syscall.FILE_FLAG_OPEN_REPARSE_POINT, 0)
	if err != nil {
		return &os.PathError{Op: "watch", Path: t.Path, Err: err}
	}
	if err = sameDirectory(h, t.Path, t.Info); err != nil {
		syscall.CloseHandle(h)
		return &os.PathError{Op: "watch", Path: t.Path, Err: err}
	}
	d.next++
	s := &directorySession{key: d.next, target: t.ID, handle: h, subtree: t.Subtree}
	if _, err = syscall.CreateIoCompletionPort(h, d.port, s.key, 0); err != nil {
		syscall.CloseHandle(h)
		return err
	}
	s.pin.Pin(s)
	if err = s.arm(); err != nil {
		s.pin.Unpin()
		syscall.CloseHandle(h)
		return err
	}
	d.sessions[s.key] = s
	d.targets[t.ID] = s
	return nil
}

// sameDirectory checks that h is the directory info describes and not a
// link to one. The handle can't be given to the os package, which would
// tie it to the Go runtime's completion port, so a second handle does.
func sameDirectory(h syscall.Handle, name string, info os.FileInfo) error {
	path, err := syscall.UTF16PtrFromString(name)
	if err != nil {
		return err
	}
	var native syscall.ByHandleFileInformation
	if err = syscall.GetFileInformationByHandle(h, &native); err != nil {
		return err
	}
	if native.FileAttributes&syscall.FILE_ATTRIBUTE_DIRECTORY == 0 {
		return errors.New("not a directory")
	}
	share := uint32(syscall.FILE_SHARE_READ | syscall.FILE_SHARE_WRITE | syscall.FILE_SHARE_DELETE)
	const fileReadAttributes = 0x80
	check, err := syscall.CreateFile(path, fileReadAttributes, share, nil, syscall.OPEN_EXISTING, syscall.FILE_FLAG_BACKUP_SEMANTICS|syscall.FILE_FLAG_OPEN_REPARSE_POINT, 0)
	if err != nil {
		return err
	}
	f := os.NewFile(uintptr(check), name)
	defer f.Close()
	var other syscall.ByHandleFileInformation
	if err = syscall.GetFileInformationByHandle(check, &other); err != nil {
		return err
	}
	checked, err := f.Stat()
	if err != nil {
		return err
	}
	// A directory to Lstat, which a junction, a symlink or a mount point
	// is not, while a placeholder of cloud files (OneDrive) is: the
	// handle is that of the reparse point, which doesn't fetch anything.
	if !checked.IsDir() {
		return errors.New("not a directory")
	}
	if other.VolumeSerialNumber != native.VolumeSerialNumber || other.FileIndexHigh != native.FileIndexHigh || other.FileIndexLow != native.FileIndexLow || !os.SameFile(checked, info) {
		return errors.New("watch target identity changed")
	}
	return nil
}

func (s *directorySession) arm() error {
	// FILE_NOTIFY_CHANGE_FILE_NAME | DIR_NAME | SIZE | LAST_WRITE: not
	// ATTRIBUTES, which antivirus and indexers change, and which would be
	// taken for writes.
	return syscall.ReadDirectoryChanges(s.handle, (*byte)(unsafe.Pointer(&s.buffer[0])), uint32(len(s.buffer)*4), s.subtree, 0x1b, nil, &s.overlap, 0)
}

// release closes the handle of s, which cancels its I/O. s stays in
// sessions until that completes.
func (d *directoryWatch) release(s *directorySession) {
	if !s.removed {
		s.removed = true
		delete(d.targets, s.target)
		syscall.CloseHandle(s.handle)
	}
}

func (d *directoryWatch) Remove(id uint64) {
	if s := d.targets[id]; s != nil {
		d.release(s)
	}
}

func (d *directoryWatch) Wait() ([]platform.FileWatchNotice, error) {
	var size, key uint32
	var overlap *syscall.Overlapped
	err := syscall.GetQueuedCompletionStatus(d.port, &size, &key, &overlap, syscall.INFINITE)
	if overlap == nil {
		return nil, err // nil for Wake
	}
	s := d.sessions[key]
	if s == nil {
		return nil, nil
	}
	if s.removed {
		d.finish(s)
		return nil, nil
	}
	var notices []platform.FileWatchNotice
	switch {
	case err == errorNotifyEnumDir || err == nil && size == 0:
		notices = []platform.FileWatchNotice{{Err: platform.ErrFileWatchOverflow}}
	case err != nil:
		// ERROR_ACCESS_DENIED once the directory is deleted, for example.
		d.finish(s)
		return []platform.FileWatchNotice{{Target: s.target, Kind: platform.FileWatchLost}}, nil
	default:
		data := unsafe.Slice((*byte)(unsafe.Pointer(&s.buffer[0])), min(int(size), len(s.buffer)*4))
		if notices, err = parseDirectoryChanges(data, s.target); err != nil {
			notices = []platform.FileWatchNotice{{Err: platform.ErrFileWatchOverflow}}
		}
	}
	if err = s.arm(); err != nil {
		d.finish(s)
		notices = append(notices, platform.FileWatchNotice{Target: s.target, Kind: platform.FileWatchLost})
	}
	return notices, nil
}

// finish releases s, which has no I/O pending.
func (d *directoryWatch) finish(s *directorySession) {
	d.release(s)
	delete(d.sessions, s.key)
	s.pin.Unpin()
}

func (d *directoryWatch) Wake() {
	d.mu.Lock()
	defer d.mu.Unlock()
	if !d.closed {
		syscall.PostQueuedCompletionStatus(d.port, 0, watchWakeKey, nil)
	}
}

func (d *directoryWatch) Close() {
	d.mu.Lock()
	d.closed = true
	d.mu.Unlock()
	for _, s := range d.sessions {
		d.release(s)
	}
	// Wait for the canceled I/O, which still writes to the sessions.
	deadline := time.Now().Add(5 * time.Second)
	for len(d.sessions) > 0 {
		wait := time.Until(deadline)
		if wait <= 0 {
			break
		}
		var size, key uint32
		var overlap *syscall.Overlapped
		err := syscall.GetQueuedCompletionStatus(d.port, &size, &key, &overlap, uint32(wait/time.Millisecond)+1)
		if overlap == nil {
			if err != nil && err != syscall.Errno(syscall.WAIT_TIMEOUT) {
				break
			}
			continue
		}
		if s := d.sessions[key]; s != nil {
			delete(d.sessions, key)
			s.pin.Unpin()
		}
	}
	if len(d.sessions) > 0 {
		leakedSessions.Lock()
		for _, s := range d.sessions {
			leakedSessions.sessions = append(leakedSessions.sessions, s)
		}
		leakedSessions.Unlock()
	}
	syscall.CloseHandle(d.port)
}

// parseDirectoryChanges reads the FILE_NOTIFY_INFORMATION records in data.
// Names below a subtree's directory are paths, which it separates with
// slashes.
func parseDirectoryChanges(data []byte, target uint64) ([]platform.FileWatchNotice, error) {
	var result []platform.FileWatchNotice
	for {
		if len(data) < 12 {
			return nil, errors.New("truncated directory notification")
		}
		next := uint64(binary.LittleEndian.Uint32(data))
		action := binary.LittleEndian.Uint32(data[4:])
		size := uint64(binary.LittleEndian.Uint32(data[8:]))
		if size == 0 || size%2 != 0 || size > uint64(len(data)-12) {
			return nil, errors.New("invalid directory notification name length")
		}
		units := make([]uint16, int(size/2))
		for i := range units {
			units[i] = binary.LittleEndian.Uint16(data[12+2*i:])
		}
		for i := 0; i < len(units); i++ {
			u := units[i]
			if u == 0 {
				return nil, errors.New("NUL in directory notification")
			}
			if u >= 0xd800 && u <= 0xdbff {
				if i+1 >= len(units) || units[i+1] < 0xdc00 || units[i+1] > 0xdfff {
					return nil, errors.New("invalid UTF-16 directory notification")
				}
				i++
			} else if u >= 0xdc00 && u <= 0xdfff {
				return nil, errors.New("invalid UTF-16 directory notification")
			}
		}
		n := platform.FileWatchNotice{Target: target, Name: strings.ReplaceAll(string(utf16.Decode(units)), `\`, "/")}
		switch action {
		case 1:
			n.Kind = platform.FileWatchCreate
		case 2:
			n.Kind = platform.FileWatchRemove
		case 3:
			n.Kind = platform.FileWatchWrite
		case 4:
			n.Kind = platform.FileWatchMoveFrom
		case 5:
			n.Kind = platform.FileWatchMoveTo
		default:
			return nil, errors.New("invalid directory notification action")
		}
		result = append(result, n)
		if next == 0 {
			return result, nil
		}
		if next%4 != 0 || next < 12+size || next > uint64(len(data)-12) {
			return nil, errors.New("invalid directory notification offset")
		}
		data = data[int(next):]
	}
}
