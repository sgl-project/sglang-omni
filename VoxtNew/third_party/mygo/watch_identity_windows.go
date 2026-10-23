package mygo

import (
	"os"
	"reflect"
	"sync"
	"sync/atomic"
)

// fileStatFields are the indices of the fields of os.fileStat that
// os.SameFile compares (os/types_windows.go), found once; ok is false
// when they aren't there as expected.
var fileStatFields struct {
	// typ is the os package's fileStat, once seen.
	typ                     atomic.Pointer[reflect.Type]
	once                    sync.Once
	ok                      bool
	path, vol, idxhi, idxlo int
}

// rememberFileStatType stores t, in a function of its own so that only a
// call that stores it moves a copy of t to the heap.
//
//go:noinline
func rememberFileStatType(t reflect.Type) { fileStatFields.typ.Store(&t) }

// fileWatchIdentity returns the identity os.SameFile compares, the volume
// and file index of the os package's fileStat, once it holds them: its
// path is empty when they were read from a handle or by StatIdentity.
// Another FileInfo, or one whose ID is still to be read, has none.
func fileWatchIdentity(info os.FileInfo) (fileWatchID, bool) {
	v := reflect.ValueOf(info)
	if v.Kind() != reflect.Pointer || v.IsNil() {
		return fileWatchID{}, false
	}
	v = v.Elem()
	t := v.Type()
	f := &fileStatFields
	if known := f.typ.Load(); known == nil || *known != t {
		if t.Kind() != reflect.Struct || t.Name() != "fileStat" || t.PkgPath() != "os" {
			return fileWatchID{}, false
		}
		rememberFileStatType(t)
	}
	f.once.Do(func() {
		index := func(name string, kind reflect.Kind) int {
			field, ok := t.FieldByName(name)
			if !ok || field.Type.Kind() != kind || len(field.Index) != 1 {
				return -1
			}
			return field.Index[0]
		}
		f.path, f.vol, f.idxhi, f.idxlo = index("path", reflect.String), index("vol", reflect.Uint32), index("idxhi", reflect.Uint32), index("idxlo", reflect.Uint32)
		f.ok = f.path >= 0 && f.vol >= 0 && f.idxhi >= 0 && f.idxlo >= 0
	})
	if !f.ok || v.Field(f.path).Len() != 0 {
		return fileWatchID{}, false
	}
	return fileWatchID{v.Field(f.vol).Uint(), v.Field(f.idxhi).Uint()<<32 | v.Field(f.idxlo).Uint()}, true
}
