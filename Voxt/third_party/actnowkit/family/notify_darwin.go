//go:build darwin

package family

import (
	"encoding/binary"
	"io"
	"os"
	"strings"
	"syscall"

	"github.com/hehehai/actnowkit/internal/darwin"
)

// validName rejects names notify(3) would misread.
func validName(name string) bool {
	return name != "" && !strings.ContainsRune(name, 0)
}

// systemNotify is true: notify_post reaches this process's registrations.
const systemNotify = true

func post(name string) error {
	if !validName(name) {
		return nil
	}
	return darwin.NotifyPost(name)
}

// watch registers names with notify(3) on one descriptor and reads the
// tokens it delivers. The descriptor is duplicated and made non-blocking
// so that the reader can be stopped by closing the duplicate (closing a
// descriptor another thread blocks in read(2) on would not wake it).
func watch(names []string, deliver func(string)) (func(), error) {
	if len(names) == 0 {
		return func() {}, nil
	}
	fd, tokens, err := darwin.NotifyRegister(names)
	if err != nil {
		return nil, err
	}
	dup, err := syscall.Dup(fd)
	if err != nil {
		darwin.NotifyCancel(tokens)
		return nil, err
	}
	syscall.CloseOnExec(dup)
	if err := syscall.SetNonblock(dup, true); err != nil {
		syscall.Close(dup)
		darwin.NotifyCancel(tokens)
		return nil, err
	}
	f := os.NewFile(uintptr(dup), "notify")
	done := make(chan struct{})
	go func() {
		defer close(done)
		var buf [4]byte
		for {
			if _, err := io.ReadFull(f, buf[:]); err != nil {
				return
			}
			if n, ok := tokens[int32(binary.BigEndian.Uint32(buf[:]))]; ok {
				deliver(n)
			}
		}
	}()
	return func() {
		f.Close()
		<-done
		darwin.NotifyCancel(tokens)
	}, nil
}
