//go:build !js

package mygo

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json/v2"
	"errors"
	"io"
	"log"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"syscall"
	"time"
)

var singleInstance struct {
	sync.Mutex
	listener net.Listener
	path     string
	socket   os.FileInfo // identifies our socket file at path
	locked   bool
	handlers listeners[func([]string, string)]
}

type secondInstanceMessage struct {
	Args       []string `json:"args"`
	WorkingDir string   `json:"workingDir"`
}

// RequestSingleInstanceLock makes sure a single instance of the app runs.
// It returns true in the first instance. In any later instance it returns
// false after forwarding its command line arguments and working directory to
// the first one (see OnSecondInstance); that instance should then exit:
//
//	if !mygo.App.RequestSingleInstanceLock() {
//		return
//	}
//
// The lock is identified by the application name and, in a packaged app,
// its identifier, and released on quit.
func (a *Application) RequestSingleInstanceLock() bool {
	if os.Getenv("MYGO_GENERATE") != "" {
		return true // Run will only write the TypeScript client.
	}
	singleInstance.Lock()
	defer singleInstance.Unlock()
	if singleInstance.locked {
		return true
	}
	path := singleInstanceSocket(singleInstanceKey(a.Name()))
	wd, _ := os.Getwd()
	msg, _ := json.Marshal(secondInstanceMessage{Args: os.Args[1:], WorkingDir: wd})
	for attempt := 0; attempt < 5; attempt++ {
		if conn, err := net.DialTimeout("unix", path, time.Second); err == nil {
			// Another instance owns the lock: hand over our arguments.
			_ = conn.SetDeadline(time.Now().Add(2 * time.Second))
			_, _ = conn.Write(msg)
			if uc, ok := conn.(*net.UnixConn); ok {
				_ = uc.CloseWrite()
			}
			_, _ = io.Copy(io.Discard, conn) // wait until it was read
			conn.Close()
			if launchedByDev() {
				// mygo dev would only see the app exit.
				log.Printf("mygo: %s is already running: handed the command line over to it (RequestSingleInstanceLock)", a.Name())
			}
			return false
		}
		// Nobody answers: remove a stale socket left by a crash and take
		// over. If another instance wins the race, Listen fails and the next
		// attempt connects to it.
		_ = os.Remove(path)
		ln, err := net.Listen("unix", path)
		if err == nil {
			// The socket file is removed on release, and only if it is
			// still ours: an instance that found it stale may have
			// replaced it.
			ln.(*net.UnixListener).SetUnlinkOnClose(false)
			singleInstance.socket, _ = os.Stat(path)
			singleInstance.listener, singleInstance.path, singleInstance.locked = ln, path, true
			go acceptSecondInstances(ln)
			a.OnQuit(releaseSingleInstanceLock)
			return true
		}
		// Windows reports WSAEADDRINUSE.
		if !errors.Is(err, syscall.EADDRINUSE) && !errors.Is(err, syscall.Errno(10048)) {
			// Locking is not possible here (e.g. read-only temp dir):
			// behave as if this were the only instance.
			return true
		}
		time.Sleep(time.Duration(attempt+1) * 20 * time.Millisecond)
	}
	return true
}

// OnSecondInstance is called in the first instance when another instance
// was started and called RequestSingleInstanceLock. Apps usually focus
// their main window here.
func (a *Application) OnSecondInstance(fn func(args []string, workingDir string)) (off func()) {
	return singleInstance.handlers.add(fn, false)
}

func acceptSecondInstances(ln net.Listener) {
	for {
		conn, err := ln.Accept()
		if err != nil {
			return
		}
		go func() {
			defer conn.Close()
			_ = conn.SetDeadline(time.Now().Add(2 * time.Second))
			data, err := io.ReadAll(io.LimitReader(conn, 1<<20))
			if err != nil {
				return
			}
			var msg secondInstanceMessage
			if json.Unmarshal(data, &msg) != nil {
				return
			}
			postMain(func() {
				for _, fn := range singleInstance.handlers.snapshot() {
					fn(msg.Args, msg.WorkingDir)
				}
				deliverArgs(msg.Args, msg.WorkingDir)
			})
		}()
	}
}

func releaseSingleInstanceLock() {
	singleInstance.Lock()
	defer singleInstance.Unlock()
	if singleInstance.listener != nil {
		singleInstance.listener.Close()
		if fi, err := os.Stat(singleInstance.path); err == nil && singleInstance.socket != nil && os.SameFile(fi, singleInstance.socket) {
			_ = os.Remove(singleInstance.path)
		}
		singleInstance.listener, singleInstance.socket, singleInstance.locked = nil, nil, false
	}
}

// singleInstanceKey identifies the lock of the app named name. The
// identifier keeps the development app of mygo dev ("<identifier>.dev")
// apart from the installed app when both call SetName with the same name.
func singleInstanceKey(name string) string {
	if info, ok := packageInfo(); ok && info.Identifier != "" {
		return info.Identifier + "\n" + name
	}
	return name
}

// singleInstanceSocket returns a short per-user socket path; Unix socket
// paths are limited to about 100 bytes.
func singleInstanceSocket(name string) string {
	sum := sha256.Sum256([]byte(name))
	file := "mygo-" + strconv.Itoa(os.Getuid()) + "-" + hex.EncodeToString(sum[:6]) + ".sock"
	for _, dir := range []string{os.Getenv("XDG_RUNTIME_DIR"), os.TempDir(), "/tmp"} {
		if dir != "" && len(filepath.Join(dir, file)) < 100 {
			return filepath.Join(dir, file)
		}
	}
	return filepath.Join("/tmp", file)
}
