package vt

import (
	"bytes"
	"encoding/base64"
)

// ClipboardAnswer is how the ClipboardRead effect answers a program's
// read of the clipboard.
type ClipboardAnswer int

const (
	// ClipboardRefused has the library refuse the read itself: an empty
	// OSC 52 reply, an EPERM status for the Kitty protocol's OSC 5522,
	// which the binding drops, as the read of a terminal that is not the
	// one to answer. It is the zero value.
	ClipboardRefused ClipboardAnswer = iota
	// ClipboardAnswered answers the read with ClipboardReply.Contents
	// now, which the library encodes as it returns.
	ClipboardAnswered
	// ClipboardDeferred holds the library's refusal back and answers the
	// read later with Terminal.AnswerClipboard, which needs a
	// ClipboardReply carrying a DeferredClipboardRead.
	ClipboardDeferred
)

// DeferredClipboardRead is a read the embedder answers after its effect
// returned (ClipboardDeferred). The binding fills it as the library
// writes the refusal it holds back: kitty tells that the read came by
// the Kitty clipboard protocol (OSC 5522) rather than OSC 52; for OSC
// 52, prefix is the reply's beginning ("\x1b]52;" with the selector and
// its ";") and terminator how the request ended; refusal is what the
// library wrote to refuse the read (the empty OSC 52 reply, or the EPERM
// packet), which a denial writes again; mimes, list and primary are what
// the program asked for, and id is the id of a Kitty protocol read,
// echoed in its answer. Pass it to Terminal.AnswerClipboard.
type DeferredClipboardRead struct {
	kitty                       bool
	prefix, terminator, refusal []byte
	mimes                       []string
	list, primary               bool
	id                          string
}

// readChunkSize is how many raw bytes of one MIME type a Kitty protocol
// DATA packet carries.
const readChunkSize = 4096

// AnswerClipboard answers a read that the ClipboardRead effect deferred
// (ClipboardDeferred): allowed serves contents by MIME type, and
// available for the types listing (OSC 5522's "."), while denied writes
// the refusal the library wrote for the read and the binding held back.
// It is safe from any goroutine, and contents and available are not
// retained after it returns.
func (t *Terminal) AnswerClipboard(d *DeferredClipboardRead, allowed bool, contents map[string][]byte, available []string) {
	if d == nil || t.effects.WritePTY == nil {
		return
	}
	if !allowed {
		t.effects.WritePTY(d.refusal)
		return
	}
	if d.kitty {
		t.effects.WritePTY(d.kittyAnswer(contents, available))
		return
	}
	out := make([]byte, 0, len(d.prefix)+len(d.terminator)+16)
	out = append(out, d.prefix...)
	out = base64.StdEncoding.AppendEncode(out, contents["text/plain"])
	out = append(out, d.terminator...)
	t.effects.WritePTY(out)
}

// kittyAnswer encodes the read's answer as the library encodes one
// (ReadSuccess.encode in Ghostty's response.zig): an OK packet, the
// targets listing when asked for, the requested representations in
// request order as DATA packets, and DONE. The loc metadata is only on
// the OK packet, and no packet carries a password, as for a reply to a
// read.
func (d *DeferredClipboardRead) kittyAnswer(contents map[string][]byte, available []string) []byte {
	out := make([]byte, 0, 64)
	packet := func(out []byte, status string, loc bool) []byte {
		out = append(out, "\x1b]5522;type=read:status="...)
		out = append(out, status...)
		if loc && d.primary {
			out = append(out, ":loc=primary"...)
		}
		if d.id != "" {
			out = append(out, ":id="...)
			out = append(out, d.id...)
		}
		return out
	}
	out = packet(out, "OK", true)
	out = append(out, d.terminator...)
	if d.list {
		out = packet(out, "DATA", false)
		out = append(out, ":mime="...)
		out = base64.StdEncoding.AppendEncode(out, []byte("."))
		if len(available) > 0 {
			// The types joined by spaces with a trailing newline, whole
			// types only, as the library writes them.
			var raw []byte
			for i, mime := range available {
				sep := 0
				if i > 0 {
					sep = 1
				}
				if len(raw)+sep+len(mime)+1 > readChunkSize {
					break
				}
				if i > 0 {
					raw = append(raw, ' ')
				}
				raw = append(raw, mime...)
			}
			raw = append(raw, '\n')
			out = append(out, ';')
			out = base64.StdEncoding.AppendEncode(out, raw)
		}
		out = append(out, d.terminator...)
	}
	for _, mime := range d.mimes {
		data := contents[mime]
		for i := 0; i < len(data); i += readChunkSize {
			n := min(len(data)-i, readChunkSize)
			out = packet(out, "DATA", false)
			out = append(out, ":mime="...)
			out = base64.StdEncoding.AppendEncode(out, []byte(mime))
			out = append(out, ';')
			out = base64.StdEncoding.AppendEncode(out, data[i:i+n])
			out = append(out, d.terminator...)
		}
	}
	out = packet(out, "DONE", false)
	out = append(out, d.terminator...)
	return out
}

// setRefusal parses the bytes the library wrote to refuse a deferred read
// and keeps them for a denial.
func (d *DeferredClipboardRead) setRefusal(p []byte) bool {
	switch {
	case bytes.HasPrefix(p, []byte("\x1b]52;")):
		// p is "\x1b]52;<selector>;<terminator>": the payload is empty.
		first := bytes.IndexByte(p, ';')
		if first < 0 {
			return false
		}
		second := bytes.IndexByte(p[first+1:], ';')
		if second < 0 {
			return false
		}
		second += first + 1
		d.kitty = false
		d.prefix = append([]byte(nil), p[:second+1]...)
		d.terminator = append([]byte(nil), p[second+1:]...)
		d.refusal = append([]byte(nil), p...)
		return true
	case bytes.HasPrefix(p, []byte("\x1b]5522;type=read:status=")):
		d.kitty = true
		d.refusal = append([]byte(nil), p...)
		switch {
		case bytes.HasSuffix(p, []byte("\x1b\\")):
			d.terminator = append([]byte(nil), p[len(p)-2:]...)
		case len(p) > 0 && p[len(p)-1] == '\x07':
			d.terminator = append([]byte(nil), p[len(p)-1:]...)
		default:
			d.terminator = append([]byte(nil), "\x1b\\"...)
		}
		d.id = ""
		if i := bytes.Index(p, []byte(":id=")); i >= 0 {
			rest := p[i+len(":id="):]
			if j := bytes.IndexAny(rest, ":\x1b\x07"); j >= 0 {
				rest = rest[:j]
			}
			d.id = string(rest)
		}
		if bytes.Contains(p, []byte(":loc=primary")) {
			d.primary = true
		}
		return true
	}
	return false
}
