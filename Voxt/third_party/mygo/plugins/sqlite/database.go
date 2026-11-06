package sqlite

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"reflect"
	"runtime"
	"strings"
	"time"
	"unicode/utf8"
	"unsafe"
)

// ErrClosed is returned when an operation uses a closed database.
var ErrClosed = errors.New("sqlite: database is closed")

// Error is an SQLite error. Code is the primary result code; ExtendedCode
// distinguishes such errors as unique, foreign-key and not-null constraints.
type Error struct {
	Code         int
	ExtendedCode int
	Message      string
}

func (e *Error) Error() string { return fmt.Sprintf("sqlite: %s (code %d)", e.Message, e.ExtendedCode) }

// OpenOptions configure a connection. Foreign keys are enabled by default.
type OpenOptions struct {
	// ReadOnly opens an existing database without allowing writes.
	ReadOnly bool
	// BusyTimeout is how long an operation waits for another connection's
	// lock: five seconds when zero, no wait when negative. Cancellation
	// stops the wait early.
	BusyTimeout time.Duration
}

// Result describes a statement. Changes counts directly affected rows;
// LastInsertID is SQLite's last inserted rowid on this connection.
type Result struct {
	Changes      int64 `json:"changes"`
	LastInsertID int64 `json:"lastInsertId"`
}

// Rows holds columns and their values in order, preserving duplicate column
// names. Values are nil, int64, float64, string or []byte. BLOBs are copied
// before the next step, so they remain valid after the statement closes.
type Rows struct {
	Columns []string `json:"columns"`
	Values  [][]any  `json:"rows"`
}

// Statement is one parameterized statement in a transaction.
type Statement struct {
	SQL  string
	Args []any
}

// DB is a connection to the plugin's SQLite library. It is safe from any
// goroutine; operations serialize on the connection and may be canceled
// while waiting for it. Use Transaction for atomic groups of statements.
type DB struct {
	gate    chan struct{}
	life    context.Context
	cancel  context.CancelFunc
	handle  uintptr // guarded by gate
	token   uintptr // the current C operation, guarded by gate
	timeout int
}

// Open opens path (or ":memory:") with the SQLite C library compiled by
// Zig, without cgo. It creates a missing file unless ReadOnly. The parent
// directory must already exist. Close the returned connection when done.
func Open(ctx context.Context, path string, opts OpenOptions) (*DB, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if path == "" || strings.IndexByte(path, 0) >= 0 || !utf8.ValidString(path) {
		return nil, errors.New("sqlite: database path must be nonempty UTF-8 without NUL")
	}
	if err := Load(); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	flags := uintptr(0x00000002 | 0x00000004 | 0x00010000) // READWRITE | CREATE | FULLMUTEX
	if opts.ReadOnly {
		flags = 0x00000001 | 0x00010000
	}
	var h uintptr
	name := append([]byte(path), 0)
	rc := call(native.open, uintptr(unsafe.Pointer(&name[0])), uintptr(unsafe.Pointer(&h)), flags, 0)
	runtime.KeepAlive(name)
	if int32(rc) != 0 {
		err := sqliteError(h, rc)
		if h != 0 {
			call(native.close, h)
		}
		return nil, err
	}
	call(native.extendedResults, h, 1)
	timeout := opts.BusyTimeout
	if timeout == 0 {
		timeout = 5 * time.Second
	}
	ms := timeout.Milliseconds()
	if timeout > 0 && ms == 0 {
		ms = 1
	}
	ms = max(0, min(ms, math.MaxInt32))
	life, cancel := context.WithCancel(context.Background())
	db := &DB{handle: h, gate: make(chan struct{}, 1), life: life, cancel: cancel, timeout: int(ms)}
	db.gate <- struct{}{}
	if err := ctx.Err(); err != nil {
		db.Close()
		return nil, err
	}
	return db, nil
}

// Close cancels in-flight operations, waits for their statements to be
// finalized, and closes the connection. Repeated calls are harmless.
func (d *DB) Close() error {
	d.cancel()
	<-d.gate
	defer func() { d.gate <- struct{}{} }()
	if d.handle == 0 {
		return nil
	}
	rc := call(native.close, d.handle)
	if int32(rc) != 0 {
		return sqliteError(d.handle, rc)
	}
	d.handle = 0
	return nil
}

// Execute runs exactly one statement with positional parameters. Supported
// values are nil, bool, integers, finite floats, strings, []byte and
// json.Number. SQL text may not contain NUL. Transaction control, ATTACH
// and VACUUM INTO are rejected; use Transaction for groups of statements.
func (d *DB) Execute(ctx context.Context, sql string, args ...any) (result Result, err error) {
	err = d.operation(ctx, func() error {
		var e error
		result, _, e = d.run(sql, args, false)
		return e
	})
	return
}

// Query runs exactly one statement and copies its rows. It also supports
// statements with RETURNING. For large results, select a bounded page.
func (d *DB) Query(ctx context.Context, sql string, args ...any) (rows Rows, err error) {
	err = d.operation(ctx, func() error {
		var e error
		_, rows, e = d.run(sql, args, true)
		return e
	})
	return
}

// Transaction runs statements under BEGIN IMMEDIATE, without interleaving
// other operations on this connection. Any error, including cancellation,
// rolls back the batch. Results are returned only after COMMIT succeeds.
func (d *DB) Transaction(ctx context.Context, statements []Statement) (results []Result, err error) {
	err = d.operation(ctx, func() (err error) {
		if len(statements) == 0 {
			results = []Result{}
			return nil
		}
		if err = d.control("BEGIN IMMEDIATE"); err != nil {
			return err
		}
		committed := false
		defer func() {
			if !committed {
				// Cleanup must run even when the operation's token is canceled.
				call(native.operationEnd, d.handle)
				if rollback := d.control("ROLLBACK"); rollback != nil {
					// SQLITE_INTERRUPT may already have rolled the transaction back.
					if call(native.autocommit, d.handle) == 0 {
						err = errors.Join(err, rollback)
					}
				}
			}
		}()
		results = make([]Result, 0, len(statements))
		for i, s := range statements {
			if err = ctx.Err(); err != nil {
				return err
			}
			var result Result
			result, _, err = d.run(s.SQL, s.Args, false)
			if err != nil {
				return fmt.Errorf("sqlite: statement %d: %w", i+1, err)
			}
			results = append(results, result)
		}
		if err = ctx.Err(); err != nil {
			return err
		}
		if d.life.Err() != nil {
			return ErrClosed
		}
		if err = d.control("COMMIT"); err != nil {
			return err
		}
		committed = true
		return nil
	})
	if err != nil {
		results = nil
	}
	return
}

func (d *DB) operation(ctx context.Context, fn func() error) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-d.life.Done():
		return ErrClosed
	case <-d.gate:
	}
	defer func() { d.gate <- struct{}{} }()
	if err := ctx.Err(); err != nil {
		return err
	}
	if d.handle == 0 || d.life.Err() != nil {
		return ErrClosed
	}
	op := call(native.operationNew, uintptr(d.timeout))
	if op == 0 {
		return &Error{7, 7, "out of memory"}
	}
	d.token = op
	call(native.operationStart, d.handle, op)
	watch := func(c context.Context) (func() bool, chan struct{}) {
		done := make(chan struct{})
		stop := context.AfterFunc(c, func() { call(native.operationCancel, op); close(done) })
		return stop, done
	}
	stopCaller, callerDone := watch(ctx)
	stopClose, closeDone := watch(d.life)
	defer func() {
		// A callback that started must finish before its C token is freed.
		if !stopCaller() {
			<-callerDone
		}
		if !stopClose() {
			<-closeDone
		}
		call(native.operationEnd, d.handle)
		call(native.free, op)
		d.token = 0
	}()
	err := fn()
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if d.life.Err() != nil {
			return ErrClosed
		}
	}
	return err
}

func sqliteError(db, rc uintptr) error {
	code := int(int32(rc))
	message := "could not open database"
	if db != 0 {
		code = int(int32(call(native.extendedCode, db)))
		message = cString(call(native.errmsg, db))
	}
	return &Error{Code: code & 255, ExtendedCode: code, Message: message}
}

func (d *DB) prepare(sql string, control bool) (uintptr, error) {
	if len(sql) > math.MaxInt32-1 || strings.IndexByte(sql, 0) >= 0 || !utf8.ValidString(sql) {
		return 0, errors.New("sqlite: SQL must be UTF-8 without NUL and fit in a C int")
	}
	buf := append([]byte(sql), 0)
	defer runtime.KeepAlive(buf)
	fn := native.prepare
	if control {
		fn = native.prepareControl
	}
	var stmt, tail uintptr
	base := uintptr(unsafe.Pointer(&buf[0]))
	rc := call(fn, d.handle, uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)), uintptr(unsafe.Pointer(&stmt)), uintptr(unsafe.Pointer(&tail)), d.token)
	if int32(rc) != 0 {
		err := sqliteError(d.handle, rc)
		if stmt != 0 {
			call(native.finalize, stmt)
		}
		return 0, err
	}
	if stmt == 0 {
		return 0, errors.New("sqlite: SQL contains no statement")
	}
	// Compile the tail before executing anything: extra SQL, including invalid
	// SQL after a valid first statement, never runs a partial operation.
	for offset := int(tail - base); offset < len(sql); offset = int(tail - base) {
		var extra uintptr
		rc = call(fn, d.handle, uintptr(unsafe.Pointer(&buf[offset])), uintptr(len(buf)-offset), uintptr(unsafe.Pointer(&extra)), uintptr(unsafe.Pointer(&tail)), d.token)
		if int32(rc) != 0 || extra != 0 {
			err := errors.New("sqlite: expected exactly one statement")
			if int32(rc) != 0 {
				err = sqliteError(d.handle, rc)
			}
			if extra != 0 {
				call(native.finalize, extra)
			}
			call(native.finalize, stmt)
			return 0, err
		}
		if int(tail-base) <= offset {
			break
		}
	}
	return stmt, nil
}

func (d *DB) run(sql string, args []any, query bool) (Result, Rows, error) {
	stmt, err := d.prepare(sql, false)
	if err != nil {
		return Result{}, Rows{}, err
	}
	defer call(native.finalize, stmt)
	if count := int(call(native.parameterCount, stmt)); count != len(args) {
		return Result{}, Rows{}, fmt.Errorf("sqlite: statement needs %d parameters, got %d", count, len(args))
	}
	for i, arg := range args {
		if err := d.bind(stmt, i+1, arg); err != nil {
			return Result{}, Rows{}, fmt.Errorf("sqlite: parameter %d: %w", i+1, err)
		}
	}
	rows := Rows{Columns: []string{}, Values: [][]any{}}
	if query {
		for i, n := 0, int(call(native.columnCount, stmt)); i < n; i++ {
			rows.Columns = append(rows.Columns, cString(call(native.columnName, stmt, uintptr(i))))
		}
	}
	before := call(native.totalChanges, d.handle)
	for {
		rc := call(native.step, stmt)
		switch int32(rc) {
		case 100: // SQLITE_ROW
			if query {
				row := make([]any, len(rows.Columns))
				for i := range row {
					row[i] = column(stmt, i)
				}
				rows.Values = append(rows.Values, row)
			}
		case 101: // SQLITE_DONE
			result := Result{LastInsertID: int64(call(native.lastInsertID, d.handle))}
			if call(native.totalChanges, d.handle) != before {
				result.Changes = int64(call(native.changes, d.handle))
			}
			return result, rows, nil
		default:
			return Result{}, Rows{}, sqliteError(d.handle, rc)
		}
	}
}

func (d *DB) control(sql string) error {
	stmt, err := d.prepare(sql, true)
	if err != nil {
		return err
	}
	defer call(native.finalize, stmt)
	if rc := call(native.step, stmt); int32(rc) != 101 {
		return sqliteError(d.handle, rc)
	}
	return nil
}

func (d *DB) bind(stmt uintptr, index int, value any) error {
	var rc uintptr
	if value == nil {
		rc = call(native.bindNull, stmt, uintptr(index))
	} else if b, ok := value.([]byte); ok {
		if len(b) > math.MaxInt32 {
			return errors.New("BLOB is too large")
		}
		// Non-NULL pointer for an empty BLOB distinguishes it from SQL NULL.
		data := b
		if len(data) == 0 {
			data = []byte{0}
		}
		rc = call(native.bindBlob, stmt, uintptr(index), uintptr(unsafe.Pointer(&data[0])), uintptr(len(b)), ^uintptr(0))
		runtime.KeepAlive(data)
	} else if n, ok := value.(json.Number); ok {
		if v, err := n.Int64(); err == nil {
			return d.bind(stmt, index, v)
		}
		if !strings.ContainsAny(string(n), ".eE") {
			return errors.New("integer exceeds SQLite's signed 64-bit range")
		}
		v, err := n.Float64()
		if err != nil {
			return err
		}
		return d.bind(stmt, index, v)
	} else {
		v := reflect.ValueOf(value)
		switch v.Kind() {
		case reflect.Bool:
			n := uintptr(0)
			if v.Bool() {
				n = 1
			}
			rc = call(native.bindInt64, stmt, uintptr(index), n)
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			rc = call(native.bindInt64, stmt, uintptr(index), uintptr(v.Int()))
		case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			if v.Uint() > math.MaxInt64 {
				return errors.New("integer exceeds SQLite's signed 64-bit range")
			}
			rc = call(native.bindInt64, stmt, uintptr(index), uintptr(v.Uint()))
		case reflect.Float32, reflect.Float64:
			f := v.Float()
			if math.IsNaN(f) || math.IsInf(f, 0) {
				return errors.New("number must be finite")
			}
			rc = call(native.bindDouble, stmt, uintptr(index), uintptr(math.Float64bits(f)))
		case reflect.String:
			s := v.String()
			if len(s) > math.MaxInt32 || !utf8.ValidString(s) {
				return errors.New("text must be UTF-8 and fit in a C int")
			}
			data := append([]byte(s), 0)
			rc = call(native.bindText, stmt, uintptr(index), uintptr(unsafe.Pointer(&data[0])), uintptr(len(s)), ^uintptr(0))
			runtime.KeepAlive(data)
		default:
			return fmt.Errorf("unsupported value %T", value)
		}
	}
	if int32(rc) != 0 {
		return sqliteError(d.handle, rc)
	}
	return nil
}

func column(stmt uintptr, index int) any {
	i := uintptr(index)
	switch call(native.columnType, stmt, i) {
	case 1:
		return int64(call(native.columnInt64, stmt, i))
	case 2:
		return math.Float64frombits(uint64(call(native.columnDouble, stmt, i)))
	case 3:
		p := call(native.columnText, stmt, i)
		n := int(call(native.columnBytes, stmt, i))
		if n == 0 {
			return ""
		}
		return string(unsafe.Slice((*byte)(mem(p)), n))
	case 4:
		p := call(native.columnBlob, stmt, i)
		n := int(call(native.columnBytes, stmt, i))
		if n == 0 {
			return []byte{}
		}
		return append([]byte{}, unsafe.Slice((*byte)(mem(p)), n)...)
	default:
		return nil
	}
}
