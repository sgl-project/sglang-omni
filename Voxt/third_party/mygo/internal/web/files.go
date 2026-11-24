//go:build js

package web

import (
	"path"
	"slices"
	"strconv"
	"strings"
	"syscall/js"
	"time"
)

// Files dropped on a window have no paths in browsers: the backend reads
// them as they drop and keeps them at paths of its own, as
// /mygo-drop/3/photo.png, which FileDrop gives, and which os.Open,
// os.ReadFile, os.Stat and os.ReadDir read, through the fs object that Go's
// syscall package calls in browsers (wasm_exec.js's, whose functions fail
// there): the backend answers for its paths and passes the others on.

// dropRoot is the directory of the files dropped, one directory for each
// drop.
const dropRoot = "/mygo-drop"

// keptDrops is how many drops the backend keeps the files of: the oldest
// go, with their memory.
const keptDrops = 16

// dropped are the files dropped, by path.
type dropped struct {
	files map[string]*droppedFile
	drops []string // the directories of the drops kept, oldest first
	n     int
	fds   map[int]*openFile
	fd    int
}

type droppedFile struct {
	data []byte
	time time.Time
}

type openFile struct {
	path string
	f    *droppedFile // nil for a directory
	at   int
}

var drops = dropped{files: map[string]*droppedFile{}, fds: map[int]*openFile{}, fd: 1 << 20}

// add keeps the files of a drop, by name, and returns their paths.
func (d *dropped) add(names []string, data [][]byte) []string {
	d.n++
	dir := dropRoot + "/" + strconv.Itoa(d.n)
	d.drops = append(d.drops, dir)
	if len(d.drops) > keptDrops {
		old := d.drops[0]
		d.drops = d.drops[1:]
		for p := range d.files {
			if strings.HasPrefix(p, old+"/") {
				delete(d.files, p)
			}
		}
	}
	now := time.Now()
	paths := make([]string, 0, len(names))
	for i, name := range names {
		// A name is a file's, never a path: browsers give no folders.
		name = strings.ReplaceAll(name, "/", "_")
		p := dir + "/" + name
		for k := 2; d.files[p] != nil; k++ {
			p = dir + "/" + strconv.Itoa(k) + "-" + name
		}
		d.files[p] = &droppedFile{data: data[i], time: now}
		paths = append(paths, p)
	}
	return paths
}

// isDir reports whether p is the root of the drops or a drop's directory.
func (d *dropped) isDir(p string) bool {
	return p == dropRoot || slices.Contains(d.drops, p)
}

// list returns the names in directory p.
func (d *dropped) list(p string) []string {
	var names []string
	if p == dropRoot {
		for _, dir := range d.drops {
			names = append(names, path.Base(dir))
		}
		return names
	}
	for f := range d.files {
		if path.Dir(f) == p {
			names = append(names, path.Base(f))
		}
	}
	slices.Sort(names)
	return names
}

// installFiles has the fs object answer for the paths of the files
// dropped.
func installFiles() {
	fs := js.Global().Get("fs")
	if fs.IsUndefined() {
		return
	}
	ours := func(p js.Value) (string, bool) {
		if p.Type() != js.TypeString {
			return "", false
		}
		s := path.Clean(p.String())
		return s, s == dropRoot || strings.HasPrefix(s, dropRoot+"/")
	}
	// The flags of opening to write, as the fs object numbers them (-1 in
	// browsers, where it writes nothing).
	writes := 0
	for _, flag := range []string{"O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND"} {
		if v := fs.Get("constants").Get(flag); v.Type() == js.TypeNumber {
			writes |= v.Int()
		}
	}
	wrap := func(name string, fn func(args []js.Value, cb js.Value) bool) {
		orig := fs.Get(name)
		fs.Set(name, js.FuncOf(func(this js.Value, args []js.Value) any {
			if len(args) > 0 && fn(args[:len(args)-1], args[len(args)-1]) {
				return nil
			}
			return orig.Call("apply", this, js.ValueOf(toAny(args)))
		}))
	}
	stat := func(args []js.Value, cb js.Value) bool {
		p, ok := ours(args[0])
		if !ok {
			return false
		}
		if drops.isDir(p) {
			cb.Invoke(js.Null(), statOf(nil, time.Now()))
		} else if f := drops.files[p]; f != nil {
			cb.Invoke(js.Null(), statOf(f, f.time))
		} else {
			cb.Invoke(fsError("ENOENT"))
		}
		return true
	}
	wrap("stat", stat)
	wrap("lstat", stat)
	wrap("open", func(args []js.Value, cb js.Value) bool {
		p, ok := ours(args[0])
		if !ok {
			return false
		}
		f := drops.files[p]
		switch {
		case f == nil && !drops.isDir(p):
			cb.Invoke(fsError("ENOENT"))
		case len(args) > 1 && args[1].Type() == js.TypeNumber && args[1].Int()&writes != 0:
			cb.Invoke(fsError("EROFS"))
		default:
			drops.fd++
			drops.fds[drops.fd] = &openFile{path: p, f: f}
			cb.Invoke(js.Null(), drops.fd)
		}
		return true
	})
	fd := func(v js.Value) *openFile {
		if v.Type() != js.TypeNumber {
			return nil
		}
		return drops.fds[v.Int()]
	}
	wrap("fstat", func(args []js.Value, cb js.Value) bool {
		o := fd(args[0])
		if o == nil {
			return false
		}
		if o.f == nil {
			cb.Invoke(js.Null(), statOf(nil, time.Now()))
		} else {
			cb.Invoke(js.Null(), statOf(o.f, o.f.time))
		}
		return true
	})
	wrap("read", func(args []js.Value, cb js.Value) bool {
		o := fd(args[0])
		if o == nil {
			return false
		}
		if o.f == nil {
			cb.Invoke(fsError("EISDIR"))
			return true
		}
		buf, offset, length := args[1], args[2].Int(), args[3].Int()
		at := o.at
		if len(args) > 4 && args[4].Type() == js.TypeNumber {
			at = args[4].Int() // pread
		} else {
			defer func() { o.at = at }()
		}
		n := min(length, max(len(o.f.data)-at, 0))
		if n > 0 {
			js.CopyBytesToJS(buf.Call("subarray", offset, offset+n), o.f.data[at:at+n])
			at += n
		}
		cb.Invoke(js.Null(), n)
		return true
	})
	wrap("close", func(args []js.Value, cb js.Value) bool {
		if fd(args[0]) == nil {
			return false
		}
		delete(drops.fds, args[0].Int())
		cb.Invoke(js.Null())
		return true
	})
	wrap("readdir", func(args []js.Value, cb js.Value) bool {
		p, ok := ours(args[0])
		if !ok {
			return false
		}
		if !drops.isDir(p) {
			cb.Invoke(fsError("ENOTDIR"))
			return true
		}
		cb.Invoke(js.Null(), js.ValueOf(toAny(drops.list(p))))
		return true
	})
}

// statOf returns the stat of a file dropped, or of a directory for nil, as
// Node's fs gives it.
func statOf(f *droppedFile, t time.Time) js.Value {
	mode, size := 0o40555, 0 // a directory, read only
	if f != nil {
		mode, size = 0o100444, len(f.data)
	}
	ms := float64(t.UnixMilli())
	st := js.ValueOf(map[string]any{
		"dev": 0, "ino": 0, "mode": mode, "nlink": 1, "uid": 0, "gid": 0, "rdev": 0,
		"size": size, "blksize": 4096, "blocks": (size + 511) / 512,
		"atimeMs": ms, "mtimeMs": ms, "ctimeMs": ms,
	})
	st.Set("isDirectory", isDirectory[f == nil])
	return st
}

// isDirectory are the isDirectory methods of stats: of files and of
// directories.
var isDirectory = map[bool]js.Func{
	false: js.FuncOf(func(js.Value, []js.Value) any { return false }),
	true:  js.FuncOf(func(js.Value, []js.Value) any { return true }),
}

// fsError returns an error of Node's fs, which Go's syscall package reads
// the code of.
func fsError(code string) js.Value {
	e := js.Global().Get("Error").New(code)
	e.Set("code", code)
	return e
}

func toAny[T any](s []T) []any {
	out := make([]any, len(s))
	for i, v := range s {
		out[i] = v
	}
	return out
}
