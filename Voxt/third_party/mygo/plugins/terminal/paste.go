package terminal

import (
	"errors"
	"slices"
	"strings"
	"time"

	"github.com/egoist/mygo/plugins/terminal/internal/vt"
	"github.com/egoist/mygo/transfer"
)

// notTakenAfter is how long after a paste's offer is written OnPasteStatus
// tells that the program did not read it: long enough for a program far
// away, over SSH, to read it.
var notTakenAfter = 5 * time.Second

// Paste pastes text, as PasteData does: bracketed when the program asks
// (mode 2004), so that it does not run as commands as it would typed, or
// offered to a program that reads the clipboard itself (mode 5522), which
// reads it as text/plain.
func (t *Terminal) Paste(text string) {
	if text != "" {
		t.PasteData(transfer.TextData(text))
	}
}

// PasteData pastes data, as the paste keys paste the clipboard. A program
// that reads the clipboard itself (mode 5522, the Kitty clipboard
// protocol, as programs that take images do) is offered d's formats as
// MIME types (a file list as text/uri-list) and reads those it wants,
// even over SSH; any other gets d's text (transfer.Text), bracketed when
// it asks (mode 2004). With no text but other formats, as only an image,
// a program that asks for bracketed pastes gets an empty one, as VS
// Code's terminal sends, and any other nothing.
//
// The terminal keeps a snapshot of d until the program reads it or
// another paste comes, and answers only that read: other terminals showing
// the same program, as other windows on a session, stay silent. A
// representation made with transfer.Lazy is read only when the program
// asks for it, on a goroutine of the terminal while it processes the
// program's output: its provider must not call the terminal nor wait on
// the main thread.
func (t *Terminal) PasteData(d transfer.Data) {
	t.pasteData(d, nil)
}

// pasteOffer is what a paste offered a program that reads the clipboard
// itself.
type pasteOffer struct {
	data  transfer.Data
	mimes []string
}

// offerMIME returns the MIME type a paste offers format as, or "" for a
// format the protocol does not carry: one of MyGo's own, or no MIME type.
func offerMIME(f transfer.Format) string {
	if f == transfer.FileList {
		return string(transfer.URIList)
	}
	m := string(f)
	if !strings.Contains(m, "/") || strings.ContainsAny(m, " \t") || strings.HasPrefix(m, "application/x-mygo-") {
		return ""
	}
	return m
}

func newOffer(d transfer.Data) *pasteOffer {
	o := &pasteOffer{data: d}
	for _, f := range d.Formats() {
		if m := offerMIME(f); m != "" && !slices.Contains(o.mimes, m) {
			o.mimes = append(o.mimes, m)
		}
	}
	return o
}

// read reads what the offer has of a MIME type.
func (o *pasteOffer) read(mime string) []byte {
	if !slices.Contains(o.mimes, mime) {
		return nil
	}
	return readFormat(o.data, mime)
}

// readFormat reads what data holds of a MIME type, as the protocols
// carry it: a file list answers text/uri-list.
func readFormat(d transfer.Data, mime string) []byte {
	b, err := d.Read(transfer.Format(mime))
	if errors.Is(err, transfer.ErrFormat) && mime == string(transfer.URIList) {
		b, err = d.Read(transfer.FileList)
	}
	if err != nil {
		return nil
	}
	return b
}

// pasteData pastes d; held are the formats the clipboard held when the
// view read only some of them, which tell whether a paste of no text had
// something else.
func (t *Terminal) pasteData(d transfer.Data, held []transfer.Format) {
	o := newOffer(d.Snapshot())
	if held == nil {
		held = o.data.Formats()
	}
	t.mu.Lock()
	if t.term == nil {
		t.mu.Unlock()
		return
	}
	var data []byte
	var status PasteStatus = -1
	if len(o.mimes) > 0 && t.term.PasteEvent(o.mimes) {
		// The event went to the input queue; the offer waits for the
		// program's read.
		t.offer = o
		status = PasteOffered
	} else {
		text, _ := o.data.Read(transfer.Text)
		bracketed := t.term.Mode(vt.ModeBracketedPaste)
		switch {
		case len(text) > 0:
			data = vt.EncodePaste(string(text), bracketed)
		case !slices.ContainsFunc(held, func(f transfer.Format) bool { return f != transfer.Text }):
		case bracketed:
			// An empty bracketed paste, as VS Code's terminal sends, which
			// programs that take images, as Claude Code, read as a paste
			// of what the clipboard holds and read it themselves.
			data = []byte("\x1b[200~\x1b[201~")
		default:
			status = PasteNoText
		}
	}
	t.term.ScrollToBottom()
	t.mu.Unlock()
	t.Send(data)
	t.redraw()
	if status < 0 {
		return
	}
	// Once what the paste sent is written, from the queue's goroutine as
	// the other statuses.
	t.in.pushDone(nil, func() {
		t.pasteStatus(PasteProgress{Status: status})
		if status != PasteOffered {
			return
		}
		time.AfterFunc(notTakenAfter, func() {
			t.mu.Lock()
			waiting := t.offer == o
			t.mu.Unlock()
			if waiting {
				t.pasteStatus(PasteProgress{Status: PasteNotTaken})
			}
		})
	})
}

func (t *Terminal) pasteStatus(p PasteProgress) {
	if on := t.opts.OnPasteStatus; on != nil {
		on(p)
	}
}

// clipboardRequest is a program's read of the clipboard that waits for
// the main thread: t.mu guards it (the reader queues it, a frame takes it
// and answers it, and Allow and Deny set answer from any goroutine).
type clipboardRequest struct {
	t        *Terminal
	deferred *vt.DeferredClipboardRead
	mimes    []string
	list     bool
	primary  bool
	ask      bool // the app is asked before it is answered
	text     string
	// answer is 0 until Allow (1) or Deny (2), the first of which wins;
	// asked tells that Options.OnClipboardRequest was called with it.
	answer int8
	asked  bool
}

// clipboardRead answers a program's read of the clipboard. What a paste
// offered the program is answered as it was; a read with a password that
// this terminal cannot grant is left to the terminal the paste was made
// in, which stays silent otherwise; and other reads follow
// Options.ClipboardRead: refused, or answered with the clipboard read on
// the main thread (ClipboardDeferred), which may first ask the app.
// t.mu is held.
func (t *Terminal) clipboardRead(req vt.ClipboardRead) (vt.ClipboardReply, vt.ClipboardAnswer) {
	o := t.offer
	if req.Granted && o != nil {
		t.offer = nil
		reply := vt.ClipboardReply{Contents: map[string][]byte{}}
		for _, mime := range req.Mimes {
			if b := o.read(mime); len(b) > 0 {
				reply.Contents[mime] = b
			}
		}
		if req.List {
			reply.Available = slices.Clone(o.mimes)
		}
		// The reply, encoded, comes to WritePTY as this returns.
		t.replying = len(reply.Contents) > 0
		return reply, vt.ClipboardAnswered
	}
	if req.CanRemember {
		// The read carries a paste event's password, which this terminal
		// does not know: it is another window's paste follow-up, and the
		// terminal that pasted answers it. Staying silent keeps this one
		// from answering twice.
		return vt.ClipboardReply{}, vt.ClipboardRefused
	}
	if t.opts.ClipboardRead == ClipboardDeny {
		// A policy denial is the answer — the library writes an empty
		// OSC 52 reply, or EPERM for OSC 5522 — so it goes through the
		// replied path, not ClipboardRefused, which keeps the terminal
		// silent for another window to answer.
		return vt.ClipboardReply{}, vt.ClipboardAnswered
	}
	r := &clipboardRequest{
		t:        t,
		deferred: &vt.DeferredClipboardRead{},
		mimes:    slices.Clone(req.Mimes),
		list:     req.List,
		primary:  req.Primary,
		ask:      t.opts.ClipboardRead == ClipboardAsk,
	}
	if r.ask && req.List && len(req.Mimes) == 0 {
		// A read of the types available is served without asking, as
		// Kitty does.
		r.ask = false
	}
	t.clipReads = append(t.clipReads, r)
	t.redraw()
	return vt.ClipboardReply{Deferred: r.deferred}, vt.ClipboardDeferred
}

// sendReply queues a paste's reply in pieces, each telling how far it went
// once written, which a slow link makes worth watching; t.mu is held.
func (t *Terminal) sendReply(p []byte) {
	total, sent := int64(len(p)), int64(0)
	// Told from the queue's goroutine, as the pieces are: here the
	// terminal is locked, which OnPasteStatus may need.
	t.in.pushDone(nil, func() {
		t.pasteStatus(PasteProgress{Status: PasteSending, Total: total})
	})
	for len(p) > 0 {
		n := min(len(p), 64<<10)
		piece := append([]byte(nil), p[:n]...)
		p = p[n:]
		sent += int64(n)
		status, done := PasteSending, sent
		if len(p) == 0 {
			status = PasteSent
		}
		t.in.pushDone(piece, func() {
			t.pasteStatus(PasteProgress{Status: status, Sent: done, Total: total})
		})
	}
}

// readsClipboard reports whether the program reads the clipboard itself
// (mode 5522), as a paste then offers it all the clipboard holds.
func (t *Terminal) readsClipboard() bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.term != nil && t.term.Mode(vt.ModePasteEvents)
}
