// Package transfer describes data exchanged through the clipboard and native
// drag and drop. Values contain explicit representations, never Go object
// references. The immutable model is shared by both integrations.
package transfer

import (
	"bytes"
	"errors"
	"fmt"
	"net/url"
	"path/filepath"
	"slices"
	"strings"
	"sync"
)

// Format names a representation. Use a MIME type for custom serialized data.
type Format string

const (
	Text    Format = "text/plain"
	HTML    Format = "text/html"
	URIList Format = "text/uri-list"
	// FileList is a URI list known to contain local files. Backends expose
	// it as the system's file-list format as well as text/uri-list.
	FileList Format = "application/x-mygo-file-list"
	PNG      Format = "image/png"
	// Markdown is Markdown's text (RFC 7763), net.daringfireball.markdown
	// on macOS.
	Markdown Format = "text/markdown"
)

// ErrFormat means the requested item or representation is not available.
var ErrFormat = errors.New("transfer: representation unavailable")

// ErrProviderPanic means a lazy provider panicked. The panic is contained at
// the data boundary, rather than escaping through a native callback.
var ErrProviderPanic = errors.New("transfer: provider panicked")

type value struct {
	once     sync.Once
	provider func() ([]byte, error)
	bytes    []byte
	err      error
}

// Representation is one format of an item, made with Bytes or Lazy.
type Representation struct {
	format Format
	value  *value
}

// Bytes makes a representation, copying b. Empty data is valid.
func Bytes(format Format, b []byte) Representation {
	validateFormat(format)
	return Representation{format: format, value: &value{bytes: bytes.Clone(b)}}
}

// Lazy advertises a format without producing its bytes. The provider runs
// at most once per snapshot, when a destination requests this format;
// failures are cached too. Native integrations call it on the main thread.
// It must return promptly and must not wait for another main-thread action.
// Prepare expensive data on a goroutine before offering it.
func Lazy(format Format, provider func() ([]byte, error)) Representation {
	validateFormat(format)
	if provider == nil {
		panic("transfer: nil provider")
	}
	return Representation{format: format, value: &value{provider: provider}}
}

func validateFormat(f Format) {
	if f == "" || strings.ContainsAny(string(f), "\x00\r\n") {
		panic("transfer: invalid format")
	}
}

// Item is a value with alternative representations, in preference order.
// For example, one item can offer HTML and plain text of the same content.
type Item struct{ reps []Representation }

// NewItem makes an item. Duplicate formats and zero representations panic.
func NewItem(reps ...Representation) Item {
	seen := map[Format]bool{}
	for _, r := range reps {
		if r.value == nil || seen[r.format] {
			panic("transfer: invalid or duplicate representation")
		}
		seen[r.format] = true
	}
	return Item{reps: slices.Clone(reps)}
}

// Formats lists the item's advertised formats without invoking providers.
func (i Item) Formats() []Format {
	f := make([]Format, len(i.reps))
	for n, r := range i.reps {
		f[n] = r.format
	}
	return f
}

// Read returns a copy of the requested bytes. It is safe from any goroutine.
func (i Item) Read(format Format) ([]byte, error) {
	for _, r := range i.reps {
		if r.format != format {
			continue
		}
		v := r.value
		v.once.Do(func() {
			defer func() {
				if p := recover(); p != nil {
					v.bytes, v.err = nil, fmt.Errorf("%w: %v", ErrProviderPanic, p)
				}
			}()
			if v.provider != nil {
				v.bytes, v.err = v.provider()
				if v.err != nil {
					v.bytes = nil
				}
				v.bytes = bytes.Clone(v.bytes)
			}
		})
		return bytes.Clone(v.bytes), v.err
	}
	return nil, ErrFormat
}

// Data is an immutable collection of items. Its zero value is empty.
type Data struct{ items []Item }

// New makes data from items, preserving their order.
func New(items ...Item) Data { return Data{items: slices.Clone(items)} }

// Items returns the items. Modifying the returned slice does not change Data.
func (d Data) Items() []Item { return slices.Clone(d.items) }

// Formats lists all formats, in item order, without invoking providers.
func (d Data) Formats() []Format {
	var formats []Format
	for _, i := range d.items {
		for _, f := range i.Formats() {
			if !slices.Contains(formats, f) {
				formats = append(formats, f)
			}
		}
	}
	return formats
}

// Preferred chooses the first preferred format advertised by the data,
// without invoking any provider. The caller supplies its own preference
// order (for example, custom JSON before HTML before plain text).
func (d Data) Preferred(formats ...Format) (Format, bool) {
	return PreferredFormat(d.Formats(), formats...)
}

// PreferredFormat negotiates formats from a clipboard or drag offer without
// requesting data. Receiver preference order wins over source item order.
func PreferredFormat(offered []Format, preferred ...Format) (Format, bool) {
	for _, f := range preferred {
		if slices.Contains(offered, f) {
			return f, true
		}
	}
	return "", false
}

// Snapshot starts a new provider cache, as each clipboard write or drag does.
func (d Data) Snapshot() Data {
	items := make([]Item, len(d.items))
	for n, i := range d.items {
		for _, r := range i.reps {
			v := &value{provider: r.value.provider}
			if v.provider == nil {
				v.bytes = bytes.Clone(r.value.bytes)
			}
			items[n].reps = append(items[n].reps, Representation{r.format, v})
		}
	}
	return New(items...)
}

// Read reads a format for platforms with a single selection/data object.
// Text items are joined with newlines and URI lists with CRLF. Other
// formats use the first item offering them; macOS preserves separate items.
func (d Data) Read(format Format) ([]byte, error) {
	var parts [][]byte
	for _, i := range d.items {
		if !slices.Contains(i.Formats(), format) {
			continue
		}
		b, err := i.Read(format)
		if err != nil {
			return nil, err
		}
		parts = append(parts, b)
		if format != Text && format != URIList && format != FileList {
			break
		}
	}
	if len(parts) == 0 {
		return nil, ErrFormat
	}
	if format == URIList || format == FileList {
		return bytes.Join(parts, []byte("\r\n")), nil
	}
	return bytes.Join(parts, []byte("\n")), nil
}

// Materialize copies matching representations into data with no providers.
// A failed representation rejects the transfer; an empty match is ErrFormat.
func (d Data) Materialize(formats []Format) (Data, error) {
	var items []Item
	for _, i := range d.items {
		var reps []Representation
		for _, f := range i.Formats() {
			if !slices.Contains(formats, f) {
				continue
			}
			b, err := i.Read(f)
			if err != nil {
				return Data{}, err
			}
			reps = append(reps, Bytes(f, b))
		}
		if len(reps) > 0 {
			items = append(items, NewItem(reps...))
		}
	}
	if len(items) == 0 {
		return Data{}, ErrFormat
	}
	return New(items...), nil
}

// TextData offers a plain text item.
func TextData(text string) Data { return New(NewItem(Bytes(Text, []byte(text)))) }

// URLData offers a URL as a URI list and as plain text.
func URLData(raw string) (Data, error) {
	u, err := url.Parse(raw)
	if err != nil || u.Scheme == "" || strings.ContainsAny(raw, "\r\n\x00") {
		return Data{}, fmt.Errorf("transfer: invalid URL %q", raw)
	}
	return New(NewItem(Bytes(URIList, []byte(raw)), Bytes(Text, []byte(raw)))), nil
}

// FileData offers existing file paths as file URLs, one item per path.
// Paths must be absolute. It neither reads nor deletes the files.
func FileData(paths ...string) (Data, error) {
	items := make([]Item, 0, len(paths))
	for _, path := range paths {
		if !filepath.IsAbs(path) || strings.ContainsRune(path, 0) {
			return Data{}, fmt.Errorf("transfer: file path must be absolute: %q", path)
		}
		p := filepath.ToSlash(path)
		if !strings.HasPrefix(p, "/") {
			p = "/" + p
		}
		u := (&url.URL{Scheme: "file", Path: p}).String()
		items = append(items, NewItem(Bytes(FileList, []byte(u)), Bytes(URIList, []byte(u))))
	}
	return New(items...), nil
}

// URLs returns the URLs of all URI-list items, ignoring comments and blanks.
func (d Data) URLs() ([]string, error) {
	b, err := d.Read(URIList)
	if err != nil {
		return nil, err
	}
	var out []string
	for _, line := range strings.Split(strings.ReplaceAll(string(b), "\r\n", "\n"), "\n") {
		line = strings.TrimSpace(line)
		if line != "" && !strings.HasPrefix(line, "#") {
			out = append(out, line)
		}
	}
	return out, nil
}

// Files returns local file paths only. Remote file authorities and other
// URL schemes are ignored, so a network URL never becomes a filesystem path.
func (d Data) Files() ([]string, error) {
	var urls []string
	var err error
	if b, e := d.Read(FileList); e == nil {
		urls = strings.Split(strings.ReplaceAll(string(b), "\r\n", "\n"), "\n")
	} else if errors.Is(e, ErrFormat) {
		urls, err = d.URLs()
	} else {
		err = e
	}
	if err != nil {
		return nil, err
	}
	var paths []string
	for _, raw := range urls {
		u, err := url.Parse(raw)
		if err != nil || u.Scheme != "file" || (u.Host != "" && u.Host != "localhost") || strings.ContainsRune(u.Path, 0) {
			continue
		}
		p := u.Path
		if len(p) > 3 && p[0] == '/' && p[2] == ':' {
			p = p[1:]
		}
		p = filepath.FromSlash(p)
		if filepath.IsAbs(p) {
			paths = append(paths, p)
		}
	}
	return paths, nil
}
