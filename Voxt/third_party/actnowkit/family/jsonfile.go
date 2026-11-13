package family

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// MaxFileSize bounds what is read of another app's file (spec §13).
const MaxFileSize = 1 << 20

// Errors of Load.
var (
	// ErrNewerSchema: the file was written by a newer major version of its
	// format. Load still fills in what it understands; the file must not
	// be written back (spec §5.1). Apps show
	// TextNewerFormat on the related settings page.
	ErrNewerSchema = errors.New("written by a newer ActNow app: read only")
	// ErrWrongKind: the file's schema names another kind of file.
	ErrWrongKind = errors.New("not the expected kind of ActNow file")
)

// CorruptError is a file that is not a JSON object of a known schema
// shape (or is too big). Treat it as absent, but never overwrite it in
// place: Quarantine it first (spec §5.1).
type CorruptError struct {
	Path string
	Err  error
}

func (e *CorruptError) Error() string { return "family: corrupt " + e.Path + ": " + e.Err.Error() }
func (e *CorruptError) Unwrap() error { return e.Err }

// Schema is a parsed "actnow.<kind>/<major>".
type Schema struct {
	Kind  string
	Major int
}

func (s Schema) String() string { return SchemaOf(s.Kind, s.Major) }

// SchemaOf formats a schema: SchemaOf("settings", 1) = "actnow.settings/1".
func SchemaOf(kind string, major int) string { return "actnow." + kind + "/" + strconv.Itoa(major) }

// ParseSchema parses "actnow.<kind>/<major>".
func ParseSchema(s string) (Schema, error) {
	rest, ok := strings.CutPrefix(s, "actnow.")
	if !ok {
		return Schema{}, fmt.Errorf("schema %q: not actnow.<kind>/<major>", s)
	}
	i := strings.LastIndexByte(rest, '/')
	if i <= 0 {
		return Schema{}, fmt.Errorf("schema %q: not actnow.<kind>/<major>", s)
	}
	major, err := strconv.Atoi(rest[i+1:])
	if err != nil || major < 1 {
		return Schema{}, fmt.Errorf("schema %q: bad major version", s)
	}
	return Schema{Kind: rest[:i], Major: major}, nil
}

// Load reads the family file at path into v (a pointer to a struct with a
// "schema" member), checking the schema against kind and major, the
// newest major the caller writes. It returns:
//
//   - nil: v holds the file;
//   - an error matching fs.ErrNotExist: there is no file;
//   - *CorruptError: not JSON, not an object, no schema, or over MaxFileSize;
//   - ErrWrongKind: the schema names another kind;
//   - ErrNewerSchema: a newer major; v holds what this version understands,
//     and the caller must not write the file back.
func Load(path, kind string, major int, v any) error {
	data, err := ReadLimited(path)
	if err != nil {
		return err
	}
	return Decode(path, data, kind, major, v)
}

// ReadLimited reads a file of at most MaxFileSize bytes; a bigger one is a
// *CorruptError.
func ReadLimited(path string) ([]byte, error) {
	f, err := openForLimitedRead(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, MaxFileSize+1))
	if err != nil {
		return nil, err
	}
	if len(data) > MaxFileSize {
		return nil, &CorruptError{Path: path, Err: errors.New("larger than 1 MB")}
	}
	return data, nil
}

// Decode is Load for bytes already read (path names them in errors).
func Decode(path string, data []byte, kind string, major int, v any) error {
	var head struct {
		Schema string `json:"schema"`
	}
	if err := json.Unmarshal(data, &head); err != nil {
		return &CorruptError{Path: path, Err: err}
	}
	s, err := ParseSchema(head.Schema)
	if err != nil {
		return &CorruptError{Path: path, Err: err}
	}
	if s.Kind != kind {
		return fmt.Errorf("%s: %s: %w", path, head.Schema, ErrWrongKind)
	}
	if err := json.Unmarshal(data, v); err != nil {
		if s.Major > major {
			// A newer format may change a member's type: keep what decoded.
			return fmt.Errorf("%s: %s: %w", path, head.Schema, ErrNewerSchema)
		}
		return &CorruptError{Path: path, Err: err}
	}
	if s.Major > major {
		return fmt.Errorf("%s: %s: %w", path, head.Schema, ErrNewerSchema)
	}
	return nil
}

// IsNotExist reports a Load error meaning "no file".
func IsNotExist(err error) bool { return errors.Is(err, fs.ErrNotExist) }

// Quarantine moves a corrupt file aside as <name>.corrupt-<time>, so that
// a new one can be written without destroying it (spec §5.1).
func Quarantine(path string) error {
	dst := path + ".corrupt-" + time.Now().UTC().Format("20060102T150405Z")
	for i := 1; ; i++ {
		if _, err := os.Lstat(dst); os.IsNotExist(err) {
			break
		}
		dst = fmt.Sprintf("%s.corrupt-%s-%d", path, time.Now().UTC().Format("20060102T150405Z"), i)
	}
	if err := os.Rename(path, dst); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

// Extra holds the members of a JSON object that this version does not
// know, so that a read-modify-write keeps them (spec §5.1, §3.7). Types of
// this module carry one in a field `json:"-"` and use UnmarshalKnown and
// MarshalKnown in their JSON methods.
type Extra map[string]json.RawMessage

// UnmarshalKnown decodes data into v (a pointer to a struct type without
// JSON methods, usually a local "plain" copy of the real type) and puts
// the members v has no field for into extra.
func UnmarshalKnown(data []byte, v any, extra *Extra) error {
	if err := json.Unmarshal(data, v); err != nil {
		return err
	}
	var all map[string]json.RawMessage
	if err := json.Unmarshal(data, &all); err != nil {
		return err
	}
	known := knownKeys(reflect.TypeOf(v).Elem())
	*extra = nil
	for k, raw := range all {
		if known[strings.ToLower(k)] {
			continue
		}
		if *extra == nil {
			*extra = Extra{}
		}
		(*extra)[k] = raw
	}
	return nil
}

// MarshalKnown encodes v (a struct without JSON methods) and appends the
// members of extra, sorted by name, after its own.
func MarshalKnown(v any, extra Extra) ([]byte, error) {
	b, err := json.Marshal(v)
	if err != nil || len(extra) == 0 {
		return b, err
	}
	b = bytes.TrimSpace(b)
	if len(b) < 2 || b[0] != '{' || b[len(b)-1] != '}' {
		return nil, errors.New("family: MarshalKnown of a non-object")
	}
	known := knownKeys(reflect.TypeOf(v))
	keys := make([]string, 0, len(extra))
	for k := range extra {
		if !known[strings.ToLower(k)] {
			keys = append(keys, k)
		}
	}
	sort.Strings(keys)
	var buf bytes.Buffer
	buf.Write(b[:len(b)-1])
	empty := len(bytes.TrimSpace(b[1:len(b)-1])) == 0
	for _, k := range keys {
		raw := extra[k]
		if !json.Valid(raw) {
			continue
		}
		if !empty {
			buf.WriteByte(',')
		}
		empty = false
		kb, _ := json.Marshal(k)
		buf.Write(kb)
		buf.WriteByte(':')
		buf.Write(raw)
	}
	buf.WriteByte('}')
	return buf.Bytes(), nil
}

var knownCache sync.Map // reflect.Type → map[string]bool

// knownKeys are the JSON member names of a struct type, lower case
// (encoding/json matches names case-insensitively).
func knownKeys(t reflect.Type) map[string]bool {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if m, ok := knownCache.Load(t); ok {
		return m.(map[string]bool)
	}
	m := map[string]bool{}
	if t.Kind() == reflect.Struct {
		for i := range t.NumField() {
			f := t.Field(i)
			if !f.IsExported() {
				continue
			}
			name := f.Name
			if tag, ok := f.Tag.Lookup("json"); ok {
				n, _, _ := strings.Cut(tag, ",")
				if n == "-" && !strings.HasPrefix(tag, "-,") {
					continue
				}
				if n != "" {
					name = n
				}
			}
			m[strings.ToLower(name)] = true
		}
	}
	knownCache.Store(t, m)
	return m
}

// MarshalIndent encodes v as family files are written: two-space indent,
// trailing newline.
func MarshalIndent(v any) ([]byte, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	var buf bytes.Buffer
	if err := json.Indent(&buf, b, "", "  "); err != nil {
		return nil, err
	}
	buf.WriteByte('\n')
	return buf.Bytes(), nil
}

// WriteJSON writes v to path atomically (spec §5.2), 0600, creating the
// directory (0700) when needed.
func WriteJSON(path string, v any) error {
	data, err := MarshalIndent(v)
	if err != nil {
		return err
	}
	if len(data) > MaxFileSize {
		return fmt.Errorf("family: %s would be larger than 1 MB", path)
	}
	return WriteFileAtomic(path, data, 0o600)
}
