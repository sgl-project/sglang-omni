package sqlite

import (
	"fmt"
	"unsafe"
)

var native struct {
	open, close, errmsg, extendedCode, extendedResults, version, threadsafe                             uintptr
	prepare, prepareControl, step, finalize, parameterCount                                             uintptr
	bindNull, bindInt64, bindDouble, bindText, bindBlob                                                 uintptr
	columnCount, columnName, columnType, columnInt64, columnDouble, columnText, columnBlob, columnBytes uintptr
	changes, totalChanges, lastInsertID, autocommit                                                     uintptr
	operationNew, operationCancel, operationStart, operationEnd, free, abiVersion                       uintptr
}

func loadNative(path string) error {
	h, err := openLibrary(path)
	if err != nil {
		return fmt.Errorf("sqlite: loading %s: %w", path, err)
	}
	ok := false
	defer func() {
		if !ok {
			closeLibrary(h)
		}
	}()
	symbols := []struct {
		name string
		addr *uintptr
	}{
		{"mygo_sqlite_abi_version", &native.abiVersion},
		{"sqlite3_open_v2", &native.open}, {"sqlite3_close_v2", &native.close},
		{"sqlite3_errmsg", &native.errmsg}, {"sqlite3_extended_errcode", &native.extendedCode},
		{"sqlite3_extended_result_codes", &native.extendedResults},
		{"sqlite3_libversion", &native.version}, {"sqlite3_threadsafe", &native.threadsafe},
		{"mygo_sqlite_prepare", &native.prepare}, {"sqlite3_prepare_v2", &native.prepareControl},
		{"sqlite3_step", &native.step}, {"sqlite3_finalize", &native.finalize},
		{"sqlite3_bind_parameter_count", &native.parameterCount},
		{"sqlite3_bind_null", &native.bindNull}, {"sqlite3_bind_int64", &native.bindInt64},
		{"mygo_sqlite_bind_double_bits", &native.bindDouble},
		{"sqlite3_bind_text", &native.bindText}, {"sqlite3_bind_blob", &native.bindBlob},
		{"sqlite3_column_count", &native.columnCount}, {"sqlite3_column_name", &native.columnName},
		{"sqlite3_column_type", &native.columnType}, {"sqlite3_column_int64", &native.columnInt64},
		{"mygo_sqlite_column_double_bits", &native.columnDouble},
		{"sqlite3_column_text", &native.columnText}, {"sqlite3_column_blob", &native.columnBlob},
		{"sqlite3_column_bytes", &native.columnBytes}, {"sqlite3_changes64", &native.changes},
		{"sqlite3_total_changes64", &native.totalChanges}, {"sqlite3_last_insert_rowid", &native.lastInsertID},
		{"sqlite3_get_autocommit", &native.autocommit},
		{"mygo_sqlite_operation_new", &native.operationNew}, {"mygo_sqlite_operation_cancel", &native.operationCancel},
		{"mygo_sqlite_operation_start", &native.operationStart}, {"mygo_sqlite_operation_end", &native.operationEnd},
		{"sqlite3_free", &native.free},
	}
	for _, s := range symbols {
		*s.addr, err = librarySymbol(h, s.name)
		if err != nil {
			return fmt.Errorf("sqlite: %s is not the plugin's SQLite library: %w", path, err)
		}
	}
	if call(native.abiVersion) != 1 || call(native.threadsafe) == 0 {
		return fmt.Errorf("sqlite: %s has an incompatible ABI or is not threadsafe", path)
	}
	// Keep the library loaded for the process: connections use its symbols.
	ok = true
	return nil
}

// mem converts an address returned by C without hiding a Go pointer.
func mem(p uintptr) unsafe.Pointer { return *(*unsafe.Pointer)(unsafe.Pointer(&p)) }

func cString(p uintptr) string {
	if p == 0 {
		return ""
	}
	n := 0
	for *(*byte)(unsafe.Add(mem(p), n)) != 0 {
		n++
	}
	return string(unsafe.Slice((*byte)(mem(p)), n))
}
