// Package terminal is a terminal for MyGo's native UI: a view that runs a
// shell, or any program, in a pseudo-terminal, with the terminal emulator
// of Ghostty, libghostty-vt, and draws it with package ui.
//
//	term, err := terminal.New(terminal.Options{})
//	...
//	mygo.NewWindow(mygo.WindowOptions{Title: "Terminal", Content: ui.View(func(c ui.Frame) {
//		terminal.View(c, term).Fill()
//	})})
//
// It does what terminals do: colors, styles and Unicode (wide characters,
// emoji, grapheme clusters), the scrollback with reflow on resize, the
// alternate screen of full-screen programs, the keyboard as legacy
// sequences or the Kitty keyboard protocol, mouse reporting, bracketed
// paste, focus reports, synchronized output, hyperlinks (OSC 8), titles,
// the working directory (OSC 7) and the clipboard (OSC 52, OSC 5522),
// which programs write through by default and read as Options.ClipboardRead
// allows. Selecting with the pointer copies as Ghostty does: a drag selects
// cells, a double click words and a triple click lines; Option (macOS) or
// Alt drags a rectangle.
//
// libghostty-vt is a native library loaded at run time, with no cgo: `mygo
// build` and `mygo dev` put the build this package binds into the app, and
// programs run without them, as with `go run` and `go test`, download it
// once into the user's cache. See LibraryPath.
package terminal

import (
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/egoist/mygo/plugins/terminal/internal/pty"
	"github.com/egoist/mygo/plugins/terminal/internal/vt"
	"github.com/egoist/mygo/transfer"
	"github.com/egoist/mygo/ui"
)

// Options configure a terminal.
type Options struct {
	// Command is the program the terminal runs and its arguments. Empty
	// runs the user's shell as a login shell, as terminal apps do: $SHELL,
	// else /bin/zsh on macOS and /bin/sh elsewhere; PowerShell on Windows.
	Command []string
	// Dir is the working directory of the program: the user's home
	// directory when empty.
	Dir string
	// Env adds to the environment of the program, as "KEY=value". The
	// terminal sets TERM=xterm-256color and COLORTERM=truecolor, and LANG
	// to a UTF-8 locale when no locale is set, as when the app starts from
	// the Finder.
	Env []string
	// Conn connects the terminal to something other than a program, such
	// as a remote shell: what the terminal reads from it shows, and what
	// is typed is written to it. Command, Dir and Env are then ignored. A
	// Conn with a method Resize(cols, rows int) error is told the size.
	Conn io.ReadWriteCloser

	// Font is the font of the text: the system's monospaced font at 13
	// DIPs when zero.
	Font Font
	// Theme sets the colors, and DarkTheme those in dark windows when set,
	// as Ghostty's theme = light:…,dark:… does (GhosttyTheme returns
	// Ghostty's themes). Without either, the terminal follows the window's
	// light or dark appearance, with LightTheme and DarkTheme.
	Theme, DarkTheme *Theme
	// Cursor is the shape of the cursor, which programs may change, and
	// NoBlink keeps it from blinking unless a program asks it to.
	Cursor  CursorStyle
	NoBlink bool
	// Scrollback is about how many bytes of output the terminal keeps above
	// the screen: 10 MB when zero, none when negative.
	Scrollback int
	// OptionAsAlt makes Option on macOS the Alt key of programs (Meta),
	// rather than the key that types accented letters and symbols.
	OptionAsAlt bool
	// CopyOnSelect copies what the pointer selects as the selection
	// ends, and what Select All selects: CopyOnSelectClipboard to the
	// clipboard, as Ghostty's copy-on-select = clipboard. Ghostty's
	// primary, the selection clipboard of Linux, is not there: ui has
	// none. SetCopyOnSelect changes it.
	CopyOnSelect CopyOnSelect
	// Keybinds bind keys to actions over DefaultKeybinds, as Ghostty's
	// keybind option: a binding replaces the defaults of its key and
	// modifiers, and one to ActionNone unbinds them, so that the program
	// gets the key. NoDefaultKeybinds starts from none, as Ghostty's
	// keybind = clear. OnKey sees keys first; see View.
	Keybinds          []Keybind
	NoDefaultKeybinds bool
	// RightClick is what a secondary click does, as Ghostty's
	// right-click-action: show the context menu when zero. A program that
	// takes the mouse gets the click instead, unless Shift is held.
	RightClick RightClickAction
	// NoClipboardWrite keeps programs from writing to the clipboard (OSC
	// 52, OSC 5522), as Ghostty's clipboard-write = deny. Writes go
	// through unless it is set; reads are governed by ClipboardRead.
	NoClipboardWrite bool
	// ClipboardRead is what programs get when they read the clipboard
	// (OSC 52 with "?", OSC 5522): ClipboardDeny by default, so reads
	// stay refused — refused with EPERM for OSC 5522 — until an app opts
	// in, where Ghostty asks by default: a terminal that can ask needs a
	// view, and one nobody shows leaves reads pending. What a paste
	// offered a program is answered whatever the policy (see PasteData).
	ClipboardRead ClipboardAccess
	// Transparent leaves the terminal's background undrawn, so that what
	// is under the view shows through, as the window's material or an
	// element's translucent background, as Ghostty's background-opacity
	// does: cells with a background color of their own still draw it.
	// Give the theme the background the view shows on, which the text
	// under a block cursor takes.
	Transparent bool

	// OnTitle, OnExit, OnBell and OnNotify run on a goroutine of the
	// terminal when a program sets the title, when the program (or Conn)
	// ends, when it rings the bell, and when it asks for a desktop
	// notification (OSC 9, OSC 777). Calls to the terminal are safe in
	// them.
	OnTitle  func(title string)
	OnExit   func(code int)
	OnBell   func()
	OnNotify func(title, body string)
	// OnPasteStatus tells how a paste goes that a program reads itself
	// (mode 5522, see PasteData), on a goroutine of the terminal: offered,
	// then sending as the program reads, then sent; or not taken when the
	// program has not read it 5 seconds after the offer was written, as a
	// program that does not read pastes, or one far away over a slow link.
	// A paste of no text to a program that reads neither the clipboard nor
	// bracketed pastes, which pastes nothing, tells PasteNoText. An app
	// shows a large image's progress, or why nothing pasted, this way.
	OnPasteStatus func(p PasteProgress)
	// OnClipboardRequest asks the app before a program reads the
	// clipboard, when ClipboardRead is ClipboardAsk; without it the
	// terminal draws its own prompt in the view. It runs on the main
	// thread, with what the program asked for; Allow or Deny answers the
	// request, in the hook or later and from any goroutine, and the first
	// of them wins. A request the hook leaves unanswered stays pending,
	// as does a read of a terminal no view shows.
	OnClipboardRequest func(req ClipboardRequest)

	// OpenLink takes what a Command+click (Control+click elsewhere)
	// opens: a hyperlink (OSC 8), a URL printed as text, or, only with
	// OpenLink set, a file path printed as text (absolute, ~/, ./ or ../,
	// one with a slash, or a file name with a :line), as printed, a
	// :line:col suffix included; a relative one is the program's, as of
	// Dir. It runs on the main thread and reports whether it opened the
	// link; a hyperlink or a URL it leaves, or every one when it is nil,
	// opens with the system (Context.OpenURL), and a path it leaves does
	// nothing.
	OpenLink func(link string) bool
	// OnPaste sees a paste of the clipboard (ActionPaste's keys, a Paste
	// menu item, a secondary click with RightClickPaste) before the
	// program gets it, on the main thread, with every format the
	// clipboard holds, read as the paste starts. It reports whether it
	// took the paste; a paste taken sends nothing, as when an app uploads
	// the clipboard's files or image (d.Files, transfer.PNG) for a remote
	// program, then pastes their paths with Paste.
	OnPaste func(d transfer.Data) bool
	// OnKey sees a key press before Keybinds and before the terminal
	// encodes it, on the main thread, and reports whether it took the key;
	// a key taken sends nothing, neither the text it types nor its
	// release. Unlike a shortcut of an element around the view, which runs
	// with the next frame, it keeps its place among the keys typed after
	// it, as does what it sends with Send. A key that only types text may
	// come as its text alone, as on Linux, which OnKey does not see.
	OnKey func(mods ui.Modifiers, key ui.Key) bool
	// OnScroll sees the wheel scroll the view by whole rows, negative up,
	// before the terminal scrolls its scrollback, on the main thread, and
	// reports whether it took the scroll; a scroll taken moves nothing. It
	// is not asked while a program takes the wheel: as the mouse, unless
	// Shift is held, or, on a full-screen program's screen, as arrow keys
	// (alternate scroll, which "\x1b[?1007l" turns off). It suits a Conn
	// whose history lives at the other end, as a remote multiplexer
	// streaming only its screen.
	OnScroll func(rows int) bool
	// OnZoom sees a zoom, on the main thread: the wheel with Control held
	// (Command on macOS), before OnScroll and a program that takes the
	// mouse, or a key bound to a font size action. It reports whether it
	// took the zoom, as an app that sizes all its text at once does; a
	// zoom it leaves, or every one without it, changes the terminal's font
	// size, a DIP a step, as Ghostty's increase_font_size:1 does, and back
	// to Font's size on a reset. Returning true from every zoom turns
	// zooming off.
	OnZoom func(z Zoom) bool
}

// ClipboardAccess is what programs get when they read the clipboard
// (Options.ClipboardRead).
type ClipboardAccess int

const (
	// ClipboardDeny refuses a read: an empty clipboard for OSC 52, an
	// EPERM status for the Kitty protocol's OSC 5522. It is the zero
	// value, so reads stay refused until an app opts in.
	ClipboardDeny ClipboardAccess = iota
	// ClipboardAllow answers a read with what the clipboard holds.
	ClipboardAllow
	// ClipboardAsk asks the app first: Options.OnClipboardRequest when
	// set, else the terminal's own prompt in the view.
	ClipboardAsk
)

// CursorStyle is the shape of the cursor.
type CursorStyle int

const (
	CursorBlock CursorStyle = iota
	CursorBar
	CursorUnderline
)

// CopyOnSelect is where a selection is copied as it ends
// (Options.CopyOnSelect).
type CopyOnSelect int

const (
	// CopyOnSelectNone copies nothing, as Ghostty's none.
	CopyOnSelectNone CopyOnSelect = iota
	// CopyOnSelectClipboard copies to the clipboard, as Ghostty's
	// clipboard.
	CopyOnSelectClipboard
)

// RightClickAction is what a secondary click does (Options.RightClick), as
// Ghostty's right-click-action.
type RightClickAction int

const (
	// RightClickContextMenu shows the context menu: Copy, Paste, Select
	// All and Clear Scrollback.
	RightClickContextMenu RightClickAction = iota
	// RightClickPaste lets go of the selection and pastes.
	RightClickPaste
	// RightClickCopy copies the selection and lets go of it.
	RightClickCopy
	// RightClickCopyOrPaste copies the selection and lets go of it, or
	// pastes when there is none, as Windows Terminal's secondary click.
	RightClickCopyOrPaste
	// RightClickIgnore does nothing.
	RightClickIgnore
)

// Zoom is a change of the font's size the user asks for (Options.OnZoom).
type Zoom struct {
	// Steps is how many steps, larger positive: a notch of the wheel
	// rolled up, or a press of a key bound to ActionIncreaseFontSize, is
	// 1. A trackpad's moves add up to notches.
	Steps int
	// Reset asks for the size of Font back (ActionResetFontSize).
	Reset bool
}

// PasteStatus is how a paste a program reads itself goes (OnPasteStatus).
type PasteStatus int

const (
	// PasteOffered: the program was told what the paste holds.
	PasteOffered PasteStatus = iota
	// PasteSending: the program reads it, and Sent of Total went so far.
	PasteSending
	// PasteSent: all of it went.
	PasteSent
	// PasteNotTaken: the program has not read it 5 seconds after it was
	// offered. It may still, as long as no other paste comes.
	PasteNotTaken
	// PasteNoText: the paste had no text, as only an image, and the
	// program reads neither the clipboard nor bracketed pastes (mode
	// 2004), so nothing was pasted. A program that takes bracketed
	// pastes gets an empty one.
	PasteNoText
)

// PasteProgress is how a paste a program reads itself goes
// (OnPasteStatus).
type PasteProgress struct {
	Status PasteStatus
	// Sent and Total count the bytes of the reply written to the program,
	// encoded as the protocol sends them (base64 and its packets), for
	// PasteSending and PasteSent: Total is known once the program reads.
	Sent, Total int64
}

// ClipboardRequest is a program's read of the clipboard that the app is
// asked about (Options.OnClipboardRequest).
type ClipboardRequest struct {
	// Mimes are the MIME types the program asked for, in order of
	// preference; empty when it asked only for what is available.
	Mimes []string
	// List tells that it asked for the types available (the Kitty
	// protocol's targets listing), which is answered without asking.
	List bool
	// Primary is the selection clipboard of X11 (OSC 52's "s"/"p").
	Primary bool
	// Text is what the clipboard held as text when the request was seen,
	// for a prompt about it.
	Text string

	req *clipboardRequest
}

// Allow answers the request with what the clipboard holds.
func (r *ClipboardRequest) Allow() { r.answer(1) }

// Deny refuses the request.
func (r *ClipboardRequest) Deny() { r.answer(2) }

// answer records the first of Allow and Deny, from any goroutine, and
// asks for the frame that answers the request.
func (r *ClipboardRequest) answer(a int8) {
	if r == nil || r.req == nil {
		return
	}
	t := r.req.t
	t.mu.Lock()
	first := r.req.answer == 0
	if first {
		r.req.answer = a
	}
	t.mu.Unlock()
	if first {
		t.redraw()
	}
}

// Terminal is a terminal emulator and the program it runs. Its methods are
// safe from any goroutine.
type Terminal struct {
	opts Options
	in   inputQueue

	// mu guards the emulator, which the reader changes and frames read.
	mu      sync.Mutex
	term    *vt.Terminal
	size    gridSize
	events  events
	closed  bool
	clipOut []string // what programs copied, for the view to write
	// clipReads are reads of the clipboard waiting for the main thread to
	// answer them: a frame takes them, reads the clipboard and writes the
	// reply (see view.answerReads).
	clipReads []*clipboardRequest
	// offer is what the last paste offered a program that reads the
	// clipboard itself (mode 5522), until it reads it; replying tells
	// that what WritePTY gets next is the reply to its read, which it
	// sends in pieces that report their progress. The answers of reads
	// the main thread writes hold t.mu too (view.finishRead).
	offer    *pasteOffer
	replying bool
	// binds are the keys bound to actions; fontSize is the size zooms
	// set, 0 for the font's.
	binds    []Keybind
	fontSize float32
	// src is the program's pseudo-terminal, or Conn: nil until the
	// program starts, from launch, its options.
	src    io.ReadWriteCloser
	proc   *pty.PTY
	launch *pty.Options
	timer  *time.Timer // starts the program when no view does

	srcOnce sync.Once
	srcErr  error
	// connMu guards the size Conn is told, from a goroutine at a time.
	connMu   sync.Mutex
	connSize gridSize
	connBusy bool

	// rmu guards what frames draw from: the render state, which a
	// program's synchronized update (mode 2026) holds.
	rmu       sync.Mutex
	rs        *vt.RenderState
	held      bool
	heldSince time.Time

	draw     atomic.Pointer[func()] // asks for a frame of the view
	done     chan struct{}
	doneOnce sync.Once
	code     atomic.Int64

	// curTheme is the view's theme, for the program's questions; t.mu
	// guards it. v is the view's state, on the main thread.
	curTheme *Theme
	v        *view
}

// gridSize is the size of the screen in cells and of a cell in pixels.
type gridSize struct{ cols, rows, cellW, cellH int }

// events are what a program did while the reader wrote its output, for
// the callbacks, which run once the lock is released.
type events struct {
	title  bool
	bells  int
	notify [][2]string
}

// New makes a terminal running opts.Command, or connected to opts.Conn.
// The program starts once a view shows the terminal, at the size of the
// view, or half a second later at 80×24 when none does.
func New(opts Options) (*Terminal, error) {
	if err := Load(); err != nil {
		return nil, err
	}
	opts.Font.Features = append([]string(nil), opts.Font.Features...)
	t := &Terminal{opts: opts, done: make(chan struct{})}
	t.binds = keybinds(&opts)
	t.size = gridSize{80, 24, 8, 16}
	var err error
	t.term, err = vt.NewTerminal(80, 24, vt.Effects{
		WritePTY: func(p []byte) {
			if !t.replying {
				t.in.push(append([]byte(nil), p...))
				return
			}
			t.replying = false
			t.sendReply(p)
		},
		Bell:         func() { t.events.bells++ },
		TitleChanged: func() { t.events.title = true },
		RenderHold:   t.renderHold,
		Clipboard: func(text string, primary bool) bool {
			if primary || t.opts.NoClipboardWrite {
				return false
			}
			t.clipOut = append(t.clipOut, text)
			t.redraw()
			return true
		},
		ClipboardRead: t.clipboardRead,
		Size:          func() (int, int, int, int) { return t.size.cols, t.size.rows, t.size.cellW, t.size.cellH },
		Dark:          func() bool { return t.curTheme == nil || t.curTheme.dark() },
		Notify: func(title, body string) {
			t.events.notify = append(t.events.notify, [2]string{title, body})
		},
	})
	if err != nil {
		return nil, err
	}
	t.term.SetTerminfoName("xterm-256color")
	// Grapheme clusters take the cells of their width, as in Ghostty:
	// emoji with modifiers and joined emoji take two.
	t.term.SetModeDefault(vt.ModeGraphemes, true)
	switch {
	case opts.Scrollback < 0:
		t.term.SetScrollback(0)
	case opts.Scrollback > 0:
		t.term.SetScrollback(opts.Scrollback)
	default:
		t.term.SetScrollback(10 << 20)
	}
	t.term.SetDefaultCursor(vtCursor(opts.Cursor), !opts.NoBlink)
	if t.rs, err = vt.NewRenderState(); err != nil {
		t.term.Free()
		return nil, err
	}
	if opts.Conn != nil {
		t.src = opts.Conn
		t.run()
		return t, nil
	}
	path, args, err := command(opts.Command)
	if err != nil {
		t.rs.Free()
		t.term.Free()
		return nil, err
	}
	dir := opts.Dir
	if dir == "" {
		dir, _ = os.UserHomeDir()
	}
	t.launch = &pty.Options{Path: path, Args: args, Dir: dir, Env: environment(opts.Env)}
	if runtime.GOOS == "windows" {
		// ConPTY keeps a screen of its own, which a resize pulling rows
		// back from the scrollback would disagree with.
		t.term.SetResizePullBack(false)
	}
	// The program starts once a view knows the size of the screen, so
	// that it draws its first screen at that size; a terminal no view
	// shows starts after half a second, at 80×24.
	t.mu.Lock()
	t.timer = time.AfterFunc(500*time.Millisecond, func() {
		t.mu.Lock()
		o := t.launch
		t.launch = nil
		t.mu.Unlock()
		t.start(o)
	})
	t.mu.Unlock()
	return t, nil
}

// run reads the output and writes the input of the program, or Conn.
func (t *Terminal) run() {
	go t.read()
	go t.in.run(t.src)
}

// start starts the program of launch, which the caller took, in a
// pseudo-terminal of the screen's size.
func (t *Terminal) start(o *pty.Options) {
	if o == nil {
		return
	}
	t.mu.Lock()
	s := t.size
	t.mu.Unlock()
	o.Cols, o.Rows, o.Width, o.Height = s.cols, s.rows, s.cols*s.cellW, s.rows*s.cellH
	proc, err := pty.Start(*o)
	if err != nil {
		t.write([]byte("\x1b[31m" + err.Error() + "\x1b[0m"))
		t.code.Store(-1)
		t.finish()
		return
	}
	t.mu.Lock()
	if t.closed {
		t.mu.Unlock()
		proc.Close()
		t.finish()
		return
	}
	t.proc, t.src = proc, proc
	if t.size != s {
		// Resized while it started.
		proc.Resize(t.size.cols, t.size.rows, t.size.cols*t.size.cellW, t.size.rows*t.size.cellH)
	}
	t.mu.Unlock()
	go func() {
		code, _ := proc.Wait()
		t.code.Store(int64(code))
		// What the program wrote before exiting is still to be read; then
		// hang up whatever else has the terminal, as terminal apps do.
		select {
		case <-t.done:
		case <-time.After(100 * time.Millisecond):
			proc.Close()
		}
	}()
	t.run()
}

// command returns the program to run and its arguments, with args[0] its
// name: the user's shell as a login shell ("-zsh") when cmd is empty.
func command(cmd []string) (path string, args []string, err error) {
	if len(cmd) > 0 {
		path, err = exec.LookPath(cmd[0])
		if err != nil {
			return "", nil, err
		}
		return path, cmd, nil
	}
	if runtime.GOOS == "windows" {
		for _, name := range []string{"pwsh.exe", "powershell.exe", os.Getenv("COMSPEC"), "cmd.exe"} {
			if p, err := exec.LookPath(name); name != "" && err == nil {
				return p, []string{name}, nil
			}
		}
		return "", nil, errors.New("terminal: no shell found")
	}
	shell := os.Getenv("SHELL")
	if shell == "" {
		shell = "/bin/sh"
		if runtime.GOOS == "darwin" {
			shell = "/bin/zsh"
		}
	}
	return shell, []string{"-" + filepath.Base(shell)}, nil
}

// environment returns the environment of the program: this process's with
// the terminal's variables and extra.
func environment(extra []string) []string {
	env := os.Environ()
	set := func(kv string) {
		k, _, _ := strings.Cut(kv, "=")
		for i, e := range env {
			if strings.HasPrefix(e, k+"=") {
				env[i] = kv
				return
			}
		}
		env = append(env, kv)
	}
	set("TERM=xterm-256color")
	set("COLORTERM=truecolor")
	if runtime.GOOS != "windows" && os.Getenv("LANG") == "" && os.Getenv("LC_ALL") == "" && os.Getenv("LC_CTYPE") == "" {
		set("LANG=en_US.UTF-8")
	}
	for _, kv := range extra {
		set(kv)
	}
	return env
}

// read processes the program's output until it ends.
func (t *Terminal) read() {
	buf := make([]byte, 64<<10)
	for {
		n, err := t.src.Read(buf)
		if n > 0 {
			t.write(buf[:n])
		}
		if err != nil {
			break
		}
	}
	t.ended()
}

// write processes output, then runs the callbacks of what it did.
func (t *Terminal) write(p []byte) {
	t.mu.Lock()
	var ev events
	var title string
	if t.term != nil {
		t.term.Write(p)
		ev, t.events = t.events, events{}
		if ev.title {
			title = t.term.Title()
		}
	}
	t.mu.Unlock()
	t.fire(ev, title)
	t.redraw()
}

func (t *Terminal) fire(ev events, title string) {
	o := &t.opts
	if ev.title && o.OnTitle != nil {
		o.OnTitle(title)
	}
	for range ev.bells {
		if o.OnBell != nil {
			o.OnBell()
		}
	}
	for _, n := range ev.notify {
		if o.OnNotify != nil {
			o.OnNotify(n[0], n[1])
		}
	}
}

// ended marks the program ended, once its output did.
func (t *Terminal) ended() {
	t.closeSrc()
	if t.proc != nil {
		code, _ := t.proc.Wait()
		t.code.Store(int64(code))
	}
	t.mu.Lock()
	if t.term != nil && !t.closed {
		msg := "[Process exited]"
		if code := t.code.Load(); code != 0 {
			msg = fmt.Sprintf("[Process exited with code %d]", code)
		}
		if t.opts.Conn != nil {
			msg = "[Disconnected]"
		}
		if x, _ := t.term.Cursor(); x > 0 {
			msg = "\r\n" + msg
		}
		t.term.Write([]byte("\x1b[0;2m" + msg + "\x1b[0m"))
	}
	t.mu.Unlock()
	t.finish()
}

// finish closes Done and calls OnExit, once.
func (t *Terminal) finish() {
	t.doneOnce.Do(func() {
		close(t.done)
		t.in.close()
		if t.opts.OnExit != nil {
			t.opts.OnExit(int(t.code.Load()))
		}
		t.redraw()
	})
}

// Done is closed once the program exited (or Conn ended).
func (t *Terminal) Done() <-chan struct{} { return t.done }

// ExitCode returns the exit code of the program once it exited: 128 plus
// the signal's number for a program a signal killed, -1 for one that
// could not start.
func (t *Terminal) ExitCode() int { return int(t.code.Load()) }

// Title returns the title the program set (OSC 0, OSC 2).
func (t *Terminal) Title() string {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.term == nil {
		return ""
	}
	return t.term.Title()
}

// Dir returns the working directory the shell reported (OSC 7, which the
// shells of macOS report), or "".
func (t *Terminal) Dir() string {
	t.mu.Lock()
	pwd := ""
	if t.term != nil {
		pwd = t.term.Pwd()
	}
	t.mu.Unlock()
	return fileURLPath(pwd)
}

// SetFont changes the font, as Options.Font, and the size a reset zooms
// back to; the font takes its size, as zooming did not happen.
func (t *Terminal) SetFont(f Font) {
	f.Features = append([]string(nil), f.Features...)
	t.mu.Lock()
	t.opts.Font, t.fontSize = f, 0
	t.mu.Unlock()
	t.redraw()
}

// font returns the font, at the size zooms set; t.mu is held.
func (t *Terminal) font() Font {
	f := t.opts.Font
	if t.fontSize > 0 {
		f.Size = t.fontSize
	}
	return f
}

// zoom changes the font's size as z asks, a DIP a step, from 1 DIP, as
// Ghostty does.
func (t *Terminal) zoom(z Zoom) {
	t.mu.Lock()
	switch {
	case z.Reset:
		t.fontSize = 0
	case z.Steps != 0:
		f := t.font()
		size := f.key(1).size
		t.fontSize = max(size+float32(z.Steps), 1)
	}
	t.mu.Unlock()
	t.redraw()
}

// SetCopyOnSelect changes where a selection is copied as it ends, as
// Options.CopyOnSelect, as when the user changes a setting.
func (t *Terminal) SetCopyOnSelect(c CopyOnSelect) {
	t.mu.Lock()
	t.opts.CopyOnSelect = c
	t.mu.Unlock()
}

// SetTheme changes the colors, as Options.Theme and DarkTheme.
func (t *Terminal) SetTheme(theme, dark *Theme) {
	t.mu.Lock()
	t.opts.Theme, t.opts.DarkTheme = theme, dark
	t.mu.Unlock()
	t.redraw()
}

// Size returns the size of the screen in cells.
func (t *Terminal) Size() (cols, rows int) {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.size.cols, t.size.rows
}

// Text returns the text of the screen and its scrollback, with the lines a
// program's long output wrapped joined.
func (t *Terminal) Text() string {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.term == nil {
		return ""
	}
	return t.term.Text()
}

// Resize sets the size of the screen in cells, for a terminal no view
// shows, as one a server keeps for a session while no window does: a view
// sets the size of the terminal it shows.
func (t *Terminal) Resize(cols, rows int) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.resize(gridSize{max(cols, 1), max(rows, 1), t.size.cellW, t.size.cellH})
}

// Snapshot returns what the terminal shows as escape sequences: its
// scrollback and screen with their styles, the cursor and the modes
// programs set. Fed to a new terminal of the same size, it shows the same,
// as a terminal attaching again to a session a server keeps.
func (t *Terminal) Snapshot() []byte {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.term == nil {
		return nil
	}
	return t.term.VT()
}

// Send sends data to the program as if typed.
func (t *Terminal) Send(data []byte) {
	if len(data) > 0 {
		t.in.push(append([]byte(nil), data...))
	}
}

// Feed shows data as if the program wrote it: text and escape sequences.
func (t *Terminal) Feed(data []byte) { t.write(data) }

// Close closes the terminal: the program is hung up (SIGHUP), or Conn
// closed, and the terminal frees its memory. A view of it then shows
// nothing.
func (t *Terminal) Close() error {
	t.mu.Lock()
	if t.closed {
		t.mu.Unlock()
		return nil
	}
	t.closed = true
	t.offer = nil
	t.clipReads = nil
	src, proc, waiting := t.src, t.proc, t.launch != nil
	t.launch = nil
	if t.timer != nil {
		t.timer.Stop()
	}
	t.mu.Unlock()
	var err error
	switch {
	case src != nil:
		err = t.closeSrc()
	case waiting:
		// The program never started; one starting finishes itself.
		t.code.Store(-1)
		t.finish()
	}
	if proc != nil {
		// A program that ignores the hangup is killed after a moment.
		go func() {
			select {
			case <-proc.Exited():
			case <-time.After(2 * time.Second):
				proc.Kill()
			}
		}()
	}
	t.in.close()
	// The reader may be writing to the emulator, and a frame drawing it:
	// free it under the locks.
	t.mu.Lock()
	t.rmu.Lock()
	t.term.Free()
	t.term = nil
	t.rs.Free()
	t.rs = nil
	t.rmu.Unlock()
	t.mu.Unlock()
	t.redraw()
	return err
}

// closeSrc closes the program's pseudo-terminal, or Conn, once.
func (t *Terminal) closeSrc() error {
	t.srcOnce.Do(func() { t.srcErr = t.src.Close() })
	return t.srcErr
}

// resizeConn tells Conn the size of the screen, from a goroutine, the
// latest size only once it told the one before.
func (t *Terminal) resizeConn(c interface{ Resize(cols, rows int) error }, s gridSize) {
	t.connMu.Lock()
	defer t.connMu.Unlock()
	t.connSize = s
	if t.connBusy {
		return
	}
	t.connBusy = true
	go func() {
		sent := gridSize{}
		for {
			t.connMu.Lock()
			s := t.connSize
			if s == sent {
				t.connBusy = false
				t.connMu.Unlock()
				return
			}
			t.connMu.Unlock()
			c.Resize(s.cols, s.rows)
			sent = s
		}
	}()
}

// resize sets the size of the screen, from the view; t.mu is held.
func (t *Terminal) resize(s gridSize) {
	if t.term == nil {
		return
	}
	if o := t.launch; o != nil {
		t.launch = nil
		t.timer.Stop()
		defer func() { go t.start(o) }() // at this size
	}
	if s == t.size {
		return
	}
	t.size = s
	t.term.Resize(s.cols, s.rows, s.cellW, s.cellH)
	switch src := t.src.(type) {
	case *pty.PTY:
		src.Resize(s.cols, s.rows, s.cols*s.cellW, s.rows*s.cellH)
	case interface{ Resize(cols, rows int) error }:
		t.resizeConn(src, s)
	}
}

// renderHold starts or ends a program's synchronized update, during which
// frames show the screen as it was when it started; t.mu is held.
func (t *Terminal) renderHold(held bool) {
	t.rmu.Lock()
	if held && t.rs != nil {
		t.rs.Update(t.term)
		t.heldSince = time.Now()
	}
	t.held = held
	t.rmu.Unlock()
}

// redraw asks the view for a frame.
func (t *Terminal) redraw() {
	if f := t.draw.Load(); f != nil {
		(*f)()
	}
}

// fileURLPath returns the path of a file URL, as shells report their
// directory (file://host/path), or s when it is not one.
func fileURLPath(s string) string {
	rest, ok := strings.CutPrefix(s, "file://")
	if !ok {
		return s
	}
	if i := strings.IndexByte(rest, '/'); i >= 0 {
		rest = rest[i:]
	}
	return unescapePath(rest)
}

func unescapePath(s string) string {
	var b strings.Builder
	for i := 0; i < len(s); i++ {
		if s[i] == '%' && i+2 < len(s) {
			if v, ok := unhex(s[i+1]); ok {
				if w, ok := unhex(s[i+2]); ok {
					b.WriteByte(v<<4 | w)
					i += 2
					continue
				}
			}
		}
		b.WriteByte(s[i])
	}
	return b.String()
}

func unhex(c byte) (byte, bool) {
	switch {
	case '0' <= c && c <= '9':
		return c - '0', true
	case 'a' <= c && c <= 'f':
		return c - 'a' + 10, true
	case 'A' <= c && c <= 'F':
		return c - 'A' + 10, true
	}
	return 0, false
}

// inputQueue writes what is typed to the program in order, from a
// goroutine of its own: a program that does not read its input must not
// block the reader, which answers its queries.
type inputQueue struct {
	mu     sync.Mutex
	cond   *sync.Cond
	queue  []queued
	closed bool
}

// queued is input to write, and what to run once it is written.
type queued struct {
	p    []byte
	done func()
}

func (q *inputQueue) init() {
	if q.cond == nil {
		q.cond = sync.NewCond(&q.mu)
	}
}

func (q *inputQueue) push(p []byte) { q.pushDone(p, nil) }

// pushDone queues p, and done to run once it is written, on the queue's
// goroutine.
func (q *inputQueue) pushDone(p []byte, done func()) {
	q.mu.Lock()
	q.init()
	if !q.closed {
		q.queue = append(q.queue, queued{p, done})
		q.cond.Signal()
	}
	q.mu.Unlock()
}

func (q *inputQueue) close() {
	q.mu.Lock()
	q.init()
	q.closed = true
	q.queue = nil
	q.cond.Broadcast()
	q.mu.Unlock()
}

func (q *inputQueue) run(w io.Writer) {
	for {
		q.mu.Lock()
		q.init()
		for len(q.queue) == 0 && !q.closed {
			q.cond.Wait()
		}
		if q.closed {
			q.mu.Unlock()
			return
		}
		item := q.queue[0]
		q.queue = q.queue[1:]
		q.mu.Unlock()
		if len(item.p) == 0 {
			// Only something to run, in its place among the writes.
		} else if _, err := w.Write(item.p); err != nil {
			q.close()
			return
		}
		if item.done != nil {
			item.done()
		}
	}
}

func vtCursor(s CursorStyle) vt.CursorStyle {
	switch s {
	case CursorBar:
		return vt.CursorBar
	case CursorUnderline:
		return vt.CursorUnderline
	}
	return vt.CursorBlock
}
