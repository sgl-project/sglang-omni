package main

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

// con is the CLI's output, on standard error: a line per step of a
// command. On a terminal, the step in progress has a spinner, with what it
// is doing and for how long, and its line then says how it went. What the
// programs the CLI runs print, such as the dev server and the app, goes
// between these lines, each line labeled with its program.
var con = newConsole(os.Stderr)

type console struct {
	mu        sync.Mutex
	w         io.Writer
	f         *os.File // w, when it is a terminal: for its width
	live      bool     // a terminal: the spinner's line is redrawn in place
	color     bool
	truecolor bool
	sym       symbols

	tasks    []*task // in progress: the spinner shows the last one
	drawn    bool    // the spinner's line is on the screen
	frame    int
	spinning bool // the goroutine that animates the spinner runs

	// The banner while its animation runs, two lines above the cursor's
	// until another line moves it up.
	banner     func(lit int) string
	bannerLive bool
}

type symbols struct {
	ok, fail, warn, info, start, arrow, bar, ellipsis string
	spinner                                           []string
}

var (
	unicodeSymbols = symbols{"✓", "✗", "▲", "•", "○", "➜", "│", "…", strings.Split("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏", "")}
	// plainSymbols are for consoles whose fonts lack the others, as the
	// Windows console's may.
	plainSymbols = symbols{"√", "×", "!", "i", "o", ">", "|", "...", []string{"-", "\\", "|", "/"}}
)

func newConsole(f *os.File) *console {
	c := &console{w: f, sym: plainSymbols}
	if isTerminal(f) && os.Getenv("TERM") != "dumb" && enableVT(f) {
		c.live, c.f = true, f
		enableVT(os.Stdout) // for the colors of the programs' output
	}
	force := os.Getenv("FORCE_COLOR")
	c.color = os.Getenv("NO_COLOR") == "" && (c.live || force != "" && force != "0" && force != "false")
	if unicodeTerminal() {
		c.sym = unicodeSymbols
	}
	t := os.Getenv("COLORTERM")
	c.truecolor = t == "truecolor" || t == "24bit" || os.Getenv("WT_SESSION") != ""
	return c
}

// unicodeTerminal reports whether the terminal has the symbols of
// unicodeSymbols. The fonts of the Windows console, which other terminals
// on Windows replace, do not.
func unicodeTerminal() bool {
	if runtime.GOOS != "windows" {
		return os.Getenv("TERM") != "linux" // the Linux console
	}
	term := os.Getenv("TERM")
	return os.Getenv("WT_SESSION") != "" || os.Getenv("CI") != "" ||
		os.Getenv("TERM_PROGRAM") == "vscode" || os.Getenv("TERMINAL_EMULATOR") == "JetBrains-JediTerm" ||
		os.Getenv("ConEmuTask") == "{cmd::Cmder}" || term == "xterm-256color" || term == "alacritty"
}

func (c *console) style(sgr, s string) string {
	if !c.color || s == "" {
		return s
	}
	return "\x1b[" + sgr + "m" + s + "\x1b[0m"
}

func bold(s string) string   { return con.style("1", s) }
func dim(s string) string    { return con.style("2", s) }
func red(s string) string    { return con.style("31", s) }
func green(s string) string  { return con.style("32", s) }
func yellow(s string) string { return con.style("33", s) }
func cyan(s string) string   { return con.style("36", s) }

// A termColor is a color of the 24-bit palette and its nearest of the 256.
type termColor struct {
	r, g, b uint8
	xterm   int
}

func (c *console) fg(col termColor) string {
	if c.truecolor {
		return fmt.Sprintf("38;2;%d;%d;%d", col.r, col.g, col.b)
	}
	return fmt.Sprintf("38;5;%d", col.xterm)
}

// memberColors are the colors of MyGO!!!!!'s members, in the band's
// lineup: Tomori, Anon, Rāna, Soyo and Taki.
var memberColors = [5]termColor{
	{0x77, 0xbb, 0xdd, 110},
	{0xff, 0x88, 0x99, 210},
	{0x77, 0xdd, 0x77, 114},
	{0xff, 0xdd, 0x88, 222},
	{0x77, 0x77, 0xaa, 103},
}

// bannerFrame is the interval at which the banner's members join.
const bannerFrame = 110 * time.Millisecond

// startBanner opens the output of a command: MyGo, its exclamation marks
// those of MyGO!!!!!, one per member of the band, each in the member's
// color. On a terminal they light up one after the other, as the members
// joined, while the command goes on.
func (c *console) startBanner(command string) {
	render := func(lit int) string {
		s := "  " + bold("MyGo")
		for i, col := range memberColors {
			if i < lit {
				s += c.style("1;"+c.fg(col), "!")
			} else {
				s += dim("!")
			}
		}
		return s + " " + dim("v"+version) + "  " + command
	}
	if !c.live || !c.color {
		c.println("\n" + render(len(memberColors)) + "\n")
		return
	}
	c.mu.Lock()
	c.settle()
	c.clear()
	_, _ = io.WriteString(c.w, "\n"+render(0)+"\n\n")
	c.banner, c.bannerLive = render, true
	c.draw()
	c.mu.Unlock()
	go func() {
		for lit := 1; lit <= len(memberColors); lit++ {
			time.Sleep(bannerFrame)
			c.mu.Lock()
			if !c.bannerLive {
				c.mu.Unlock()
				return
			}
			c.paintBanner(lit)
			c.bannerLive = lit < len(memberColors)
			c.mu.Unlock()
		}
	}()
}

// paintBanner redraws the banner, two lines above the cursor's, which it
// leaves where it was. With c.mu held.
func (c *console) paintBanner(lit int) {
	_, _ = io.WriteString(c.w, "\x1b7\x1b[2A\r"+c.banner(lit)+"\x1b[K\x1b8")
}

// settle ends the banner's animation, before a line moves the banner up.
// With c.mu held.
func (c *console) settle() {
	if c.bannerLive {
		c.paintBanner(len(memberColors))
		c.bannerLive = false
	}
}

// println writes a line of the CLI's own.
func (c *console) println(s string) { c.write(s + "\n") }

func (c *console) write(s string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.settle()
	c.clear()
	_, _ = io.WriteString(c.w, s)
	c.draw()
}

// logf tells what a command does, or did.
func logf(format string, args ...any) { con.note(dim(con.sym.info), fmt.Sprintf(format, args...)) }

// warnf tells what a command did not do, or what may be wrong.
func warnf(format string, args ...any) { con.note(yellow(con.sym.warn), fmt.Sprintf(format, args...)) }

func (c *console) note(symbol, msg string) {
	first, rest, more := strings.Cut(msg, "\n")
	s := "  " + symbol + " " + first + "\n"
	if more {
		s += indent(rest, "    ")
	}
	c.write(s)
}

// indent puts prefix before each line of s, which ends with a newline.
func indent(s, prefix string) string {
	var b strings.Builder
	for _, l := range strings.Split(strings.TrimRight(s, "\n"), "\n") {
		b.WriteString(strings.TrimRight(prefix+l, " ") + "\n")
	}
	return b.String()
}

// details prints the error of a step that failed, under its line: for go
// build, what the compiler said.
func (c *console) details(err error) {
	msg := err.Error()
	var be *goBuildError
	if errors.As(err, &be) && strings.TrimSpace(be.out) != "" {
		msg = strings.TrimSpace(be.out)
	}
	c.write(indent(msg, "    "))
}

// fatal prints the error that ends a command. Steps still in progress are
// those it cut short.
func (c *console) fatal(err error) {
	c.mu.Lock()
	c.settle()
	c.clear()
	for i := len(c.tasks) - 1; i >= 0; i-- {
		_, _ = io.WriteString(c.w, c.tasks[i].failLine())
	}
	c.tasks = nil
	c.mu.Unlock()
	first, rest, more := strings.Cut(err.Error(), "\n")
	s := "\n  " + c.style("1;31", "error:") + " " + first + "\n"
	if more {
		s += indent(rest, "    ")
	}
	c.write(s)
}

// interrupted ends the output of a command that Ctrl+C stopped.
func (c *console) interrupted() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.settle()
	c.clear()
	c.tasks = nil
	_, _ = io.WriteString(c.w, "\n")
}

// clearScreen clears the terminal and its scrollback.
func (c *console) clearScreen() {
	if c.live {
		c.write("\x1b[H\x1b[2J\x1b[3J")
	}
}

// A task is a step of a command in progress.
type task struct {
	c      *console
	title  string
	detail string // what it is doing
	start  time.Time
}

// start begins a step, until done, fail or stop ends it. Steps started
// while another runs take the spinner until they end.
func (c *console) start(title string) *task {
	t := &task{c: c, title: title, start: time.Now()}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.clear()
	if !c.live {
		_, _ = io.WriteString(c.w, "  "+dim(c.sym.start)+" "+title+"\n")
	}
	c.tasks = append(c.tasks, t)
	c.draw()
	if c.live && !c.spinning {
		c.spinning = true
		go c.spin()
	}
	return t
}

// spin animates the spinner until no step is in progress.
func (c *console) spin() {
	tick := time.NewTicker(80 * time.Millisecond)
	defer tick.Stop()
	for range tick.C {
		c.mu.Lock()
		if len(c.tasks) == 0 {
			c.spinning = false
			c.mu.Unlock()
			return
		}
		c.frame++
		c.draw()
		c.mu.Unlock()
	}
}

// draw writes the spinner's line, cut to the terminal's width: a line that
// wraps could not be redrawn in place. With c.mu held.
func (c *console) draw() {
	if !c.live || len(c.tasks) == 0 {
		return
	}
	t := c.tasks[len(c.tasks)-1]
	width := termWidth(c.f)
	if width <= 0 {
		width = 80
	}
	room := width - 5 // the spinner, its margins and a spare column
	title := c.cut(t.title, room)
	room -= utf8.RuneCountInString(title)
	elapsed := ""
	if d := time.Since(t.start); d >= time.Second {
		elapsed = " " + formatElapsed(d)
	}
	if len(elapsed) > room {
		elapsed = ""
	}
	room -= len(elapsed)
	detail := ""
	if t.detail != "" && room > 8 {
		detail = c.cut("  "+t.detail, room)
	}
	frame := c.sym.spinner[c.frame%len(c.sym.spinner)]
	// Without autowrap, in case the width is wrong, as with wide
	// characters.
	_, _ = io.WriteString(c.w, "\x1b[?7l\r  "+cyan(frame)+" "+title+dim(elapsed+detail)+"\x1b[K\x1b[?7h")
	c.drawn = true
}

// clear erases the spinner's line, for other output. With c.mu held.
func (c *console) clear() {
	if c.drawn {
		_, _ = io.WriteString(c.w, "\r\x1b[K")
		c.drawn = false
	}
}

// cut shortens s to n characters.
func (c *console) cut(s string, n int) string {
	if utf8.RuneCountInString(s) <= n {
		return s
	}
	e := utf8.RuneCountInString(c.sym.ellipsis)
	if n <= e {
		return ""
	}
	return string([]rune(s)[:n-e]) + c.sym.ellipsis
}

// set shows what the step is doing.
func (t *task) set(detail string) {
	if t == nil {
		return
	}
	t.c.mu.Lock()
	t.detail = detail
	t.c.mu.Unlock()
}

// rename changes the step's title, as when it moves on to its next part.
func (t *task) rename(title string) {
	if t == nil {
		return
	}
	t.c.mu.Lock()
	t.title, t.detail = title, ""
	t.c.mu.Unlock()
}

func (t *task) elapsed() string { return formatDuration(time.Since(t.start)) }

// done ends the step with msg, and how long the step took.
func (t *task) done(msg string) {
	if t != nil {
		t.finish("  " + green(t.c.sym.ok) + " " + msg + "  " + dim(t.elapsed()) + "\n")
	}
}

// doneAs ends the step with msg alone, which tells how long it took.
func (t *task) doneAs(msg string) {
	if t != nil {
		t.finish("  " + green(t.c.sym.ok) + " " + msg + "\n")
	}
}

// fail ends the step as failed; its error comes after.
func (t *task) fail() {
	if t != nil {
		t.finish(t.failLine())
	}
}

func (t *task) failLine() string {
	return "  " + red(t.c.sym.fail) + " " + t.title + "  " + dim(t.elapsed()) + "\n"
}

// stop ends the step without a line, as when it was canceled.
func (t *task) stop() {
	if t != nil {
		t.finish("")
	}
}

// end ends the step with msg, or as failed with err, and returns err.
func (t *task) end(err error, msg string) error {
	if err != nil {
		t.fail()
	} else {
		t.done(msg)
	}
	return err
}

func (t *task) finish(line string) {
	c := t.c
	c.mu.Lock()
	defer c.mu.Unlock()
	i := slices.Index(c.tasks, t)
	if i < 0 {
		return // ended already
	}
	c.tasks = slices.Delete(c.tasks, i, i+1)
	if line != "" {
		c.settle()
	}
	c.clear()
	_, _ = io.WriteString(c.w, line)
	c.draw()
}

// formatDuration formats how long a step took.
func formatDuration(d time.Duration) string {
	switch {
	case d < time.Second:
		return fmt.Sprintf("%dms", d.Milliseconds())
	case d < time.Minute:
		return fmt.Sprintf("%.1fs", d.Seconds())
	}
	return formatElapsed(d)
}

// formatElapsed formats how long a step has been running.
func formatElapsed(d time.Duration) string {
	d = d.Truncate(time.Second)
	if d < time.Minute {
		return fmt.Sprintf("%ds", int(d.Seconds()))
	}
	return fmt.Sprintf("%dm %02ds", int(d.Minutes()), int(d.Seconds())%60)
}

// output returns a writer for what a program the CLI runs prints to w,
// which labels each line with name.
func (c *console) output(name string, w io.Writer) *labeled {
	l := &labeled{c: c, w: w}
	if f, ok := w.(*os.File); c.color && ok && (f == c.f || isTerminal(f)) {
		l.color = true
		l.prefix = "  " + c.style("2", name+" "+c.sym.bar)
	} else {
		l.prefix = "  " + name + " " + c.sym.bar
	}
	return l
}

// labeled writes the lines of a program's output with its label.
type labeled struct {
	c      *console
	w      io.Writer
	prefix string
	color  bool   // w shows colors
	buf    []byte // the line being written
	closed bool
}

func (l *labeled) Write(p []byte) (int, error) {
	l.c.mu.Lock()
	defer l.c.mu.Unlock()
	if l.closed {
		return len(p), nil
	}
	l.buf = append(l.buf, p...)
	var out bytes.Buffer
	for {
		i := bytes.IndexAny(l.buf, "\r\n")
		// A \r last may be the start of a \r\n.
		if i < 0 || l.buf[i] == '\r' && i == len(l.buf)-1 {
			break
		}
		n := i + 1
		if l.buf[i] == '\r' && l.buf[n] == '\n' {
			n++
		}
		l.line(&out, l.buf[:i])
		l.buf = l.buf[n:]
	}
	if len(l.buf) > 64<<10 {
		l.line(&out, l.buf)
		l.buf = nil
	}
	l.emit(out.Bytes())
	return len(p), nil
}

func (l *labeled) line(out *bytes.Buffer, s []byte) {
	out.WriteString(l.prefix)
	if len(s) > 0 {
		out.WriteByte(' ')
		out.Write(s)
		if l.color {
			out.WriteString("\x1b[0m") // colors end with their line
		}
	}
	out.WriteByte('\n')
}

// emit writes complete lines, with c.mu held.
func (l *labeled) emit(b []byte) {
	if len(b) == 0 {
		return
	}
	l.c.settle()
	l.c.clear()
	_, _ = l.w.Write(b)
	l.c.draw()
}

// flush writes the last line, which did not end.
func (l *labeled) flush() {
	l.c.mu.Lock()
	defer l.c.mu.Unlock()
	if !l.closed && len(l.buf) > 0 {
		var out bytes.Buffer
		l.line(&out, bytes.TrimRight(l.buf, "\r"))
		l.buf = nil
		l.emit(out.Bytes())
	}
}

// close drops what is written after, such as what script runners print
// when they are stopped.
func (l *labeled) close() {
	l.c.mu.Lock()
	l.closed = true
	l.c.mu.Unlock()
}

// pipe returns a file for a program to write its output to, which reaches
// w, and a channel closed once everything written got there. Unlike a
// writer given to exec.Cmd, Wait does not wait for what holds it: the
// processes an app starts, such as its web view's, may hold it after the
// app exited. The caller closes the file once the program started.
func pipe(w *labeled) (*os.File, <-chan struct{}, error) {
	r, pw, err := os.Pipe()
	if err != nil {
		return nil, nil, err
	}
	done := make(chan struct{})
	go func() {
		_, _ = io.Copy(w, r)
		r.Close()
		w.flush()
		close(done)
	}()
	return pw, done, nil
}

// changeSummary tells which of the files at paths changed, relative to
// root.
func changeSummary(root string, paths []string) string {
	names := make([]string, 0, 3)
	for _, p := range paths[:min(len(paths), 3)] {
		names = append(names, relPathTo(root, p))
	}
	switch n := len(paths); {
	case n > 3:
		return strings.Join(names, ", ") + fmt.Sprintf(" and %d more changed", n-3)
	case n > 1:
		return strings.Join(names[:n-1], ", ") + " and " + names[n-1] + " changed"
	case n == 1:
		return names[0] + " changed"
	}
	return "files changed"
}

// relPathTo returns path relative to root, with slashes, for messages.
func relPathTo(root, path string) string {
	if r, err := filepath.Rel(root, path); err == nil {
		return filepath.ToSlash(r)
	}
	return path
}

func isTerminal(f *os.File) bool {
	info, err := f.Stat()
	return err == nil && info.Mode()&os.ModeCharDevice != 0
}
