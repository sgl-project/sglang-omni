//go:build !js

package mygo

import (
	"context"
	jsonv1 "encoding/json"
	"encoding/json/jsontext"
	"encoding/json/v2"
	"errors"
	"fmt"
	"log"
	"reflect"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"sync"
	"unsafe"

	"github.com/egoist/mygo/internal/tsgen"
)

// jsonOptions are used for every value crossing the IPC boundary.
var jsonOptions = json.JoinOptions(
	jsonv1.FormatDurationAsNano(true),
	jsontext.EscapeForJS(true),
)

type service struct {
	name     string
	typ      reflect.Type
	methods  []*method
	internal bool
}

type method struct {
	service  string
	name     string
	fn       reflect.Value
	ctx      bool
	params   []reflect.Type
	variadic bool
	// chans marks the parameters that are channels (*Channel[T]), if any
	// is (streams).
	chans   []bool
	streams bool
	result  reflect.Type
	hasErr  bool
	pc      uintptr
}

type eventInfo struct {
	name string
	typ  reflect.Type
	pc   uintptr
	file string
	line int
}

var ipc struct {
	sync.RWMutex
	services []*service
	methods  map[string]*method
	events   []*eventInfo
	// writeClient is WriteTypeScript once the app bound a service or
	// declared an event, which App.Run calls under MYGO_GENERATE: an app
	// that binds nothing writes no client, and links no generator.
	writeClient func(path string) error
}

var (
	contextType = reflect.TypeFor[context.Context]()
	errorType   = reflect.TypeFor[error]()
)

// Bind exposes the exported methods of each service to the frontend. The
// generated TypeScript client (see `mygo generate`) turns
//
//	type Greeter struct{}
//
//	// Greet returns a greeting.
//	func (Greeter) Greet(name string) (string, error)
//
// into a typed `Greeter.greet(name: string): Promise<string>` function.
//
// Arguments and results are encoded as JSON. A method may take a
// context.Context as its first parameter: it is canceled when the calling
// page navigates away or its window closes, and CallerWindow returns the
// calling window from it. A method may return nothing, a value, an error, or
// a value and an error; a non-nil error rejects the promise on the
// JavaScript side. Calls run on their own goroutine.
//
// The service is named after its type; Bind panics if the name is taken or a
// method uses a type that cannot be encoded as JSON (channels, functions).
func Bind(services ...any) {
	linkClient()
	for _, s := range services {
		if err := bind("", s, false); err != nil {
			panic(err)
		}
	}
}

// BindAs is Bind with an explicit service name, for services whose type
// names collide.
func BindAs(name string, service any) {
	linkClient()
	if err := bind(name, service, false); err != nil {
		panic(err)
	}
}

// linkClient makes App.Run write the TypeScript client under
// MYGO_GENERATE, for Bind and BindAs: only apps that bind something link
// the generator (go/parser and all), which plugins' services, bound by
// MyGo, and apps of native UI have no use for.
func linkClient() {
	ipc.Lock()
	ipc.writeClient = WriteTypeScript
	ipc.Unlock()
}

func bind(name string, svc any, internal bool) error {
	v := reflect.ValueOf(svc)
	if !v.IsValid() {
		return errors.New("mygo: cannot bind nil")
	}
	t := v.Type()
	base := t
	if base.Kind() == reflect.Pointer {
		base = base.Elem()
	}
	if name == "" {
		name = base.Name()
	}
	if name == "" {
		return fmt.Errorf("mygo: cannot bind %s: type has no name, use BindAs", t)
	}
	if !internal && !tsgen.IsIdentifier(name) {
		return fmt.Errorf("mygo: service name %q is not a valid identifier", name)
	}
	s := &service{name: name, typ: t, internal: internal}
	for i := range t.NumMethod() {
		m := t.Method(i)
		mi, err := newMethod(name, m.Name, v.Method(m.Index))
		if err != nil {
			return err
		}
		mi.pc = methodPC(t, m.Name)
		s.methods = append(s.methods, mi)
	}
	if len(s.methods) == 0 {
		return fmt.Errorf("mygo: %s has no exported methods to bind", t)
	}
	return register(s)
}

// bindFuncs binds a service of MyGo's own whose methods are funcs, by
// name. Unlike bind, it does not list methods with reflect: a program
// that does reflect.Type.Method keeps every exported method of its types,
// as the linker cannot tell which it calls, so only apps that call Bind
// should pay for it.
func bindFuncs(name string, funcs ...namedFunc) error {
	s := &service{name: name, internal: true}
	for _, f := range funcs {
		fn := reflect.ValueOf(f.fn)
		mi, err := newMethod(name, f.name, fn)
		if err != nil {
			return err
		}
		mi.pc = fn.Pointer()
		s.methods = append(s.methods, mi)
	}
	return register(s)
}

// namedFunc is a method of a service that bindFuncs binds.
type namedFunc struct {
	name string
	fn   any
}

// register adds the bound service s.
func register(s *service) error {
	ipc.Lock()
	defer ipc.Unlock()
	for _, x := range ipc.services {
		if x.name == s.name {
			return fmt.Errorf("mygo: a service named %s is already bound", s.name)
		}
	}
	if ipc.methods == nil {
		ipc.methods = map[string]*method{}
	}
	ipc.services = append(ipc.services, s)
	for _, m := range s.methods {
		ipc.methods[s.name+"."+m.name] = m
	}
	return nil
}

// newMethod makes the method name of service, which calls fn, a func.
func newMethod(service, name string, fn reflect.Value) (*method, error) {
	ft := fn.Type()
	mi := &method{service: service, name: name, fn: fn, variadic: ft.IsVariadic()}
	where := service + "." + name
	start := 0
	if ft.NumIn() > 0 && ft.In(0) == contextType {
		mi.ctx = true
		start = 1
	}
	for i := start; i < ft.NumIn(); i++ {
		p := ft.In(i)
		isChan := p.Implements(channelParamType)
		t := p
		if isChan {
			t = reflect.Zero(p).Interface().(channelParam).valueType()
		}
		if err := validateType(t); err != nil {
			return nil, fmt.Errorf("mygo: %s: parameter %d: %w", where, i-start+1, err)
		}
		mi.params = append(mi.params, p)
		mi.chans = append(mi.chans, isChan)
		mi.streams = mi.streams || isChan
	}
	switch ft.NumOut() {
	case 0:
	case 1:
		if ft.Out(0) == errorType {
			mi.hasErr = true
		} else {
			mi.result = ft.Out(0)
		}
	case 2:
		if ft.Out(1) != errorType {
			return nil, fmt.Errorf("mygo: %s: the second result must be an error", where)
		}
		mi.result, mi.hasErr = ft.Out(0), true
	default:
		return nil, fmt.Errorf("mygo: %s: methods may return at most a value and an error", where)
	}
	if mi.result != nil {
		if err := validateType(mi.result); err != nil {
			return nil, fmt.Errorf("mygo: %s: result: %w", where, err)
		}
	}
	return mi, nil
}

// validateType reports an error if values of type t cannot cross the IPC
// boundary.
func validateType(t reflect.Type) error {
	if hasChannel(t, map[reflect.Type]bool{}) {
		return fmt.Errorf("type %s holds a Channel, which can only be a parameter", t)
	}
	return tsgen.Validate(t)
}

// methodPC returns the entry of the method's own code rather than the
// wrapper Go generates for value receivers called through a pointer.
func methodPC(t reflect.Type, name string) uintptr {
	if t.Kind() == reflect.Pointer {
		if m, ok := t.Elem().MethodByName(name); ok {
			return m.Func.Pointer()
		}
	}
	if m, ok := t.MethodByName(name); ok {
		return m.Func.Pointer()
	}
	return 0
}

func lookupMethod(name string) *method {
	ipc.RLock()
	defer ipc.RUnlock()
	return ipc.methods[name]
}

// call calls the method for the page of w whose context is page and whose
// token is token.
func (m *method) call(w *Window, page context.Context, token string, args []rawValue) (result any, err error) {
	ctx, cancel := page, context.CancelFunc(nil)
	var chans []*channel
	if m.streams {
		// The page closing a channel cancels the call. The channels close
		// when it returns, before its result is sent.
		ctx, cancel = context.WithCancel(page)
		defer func() {
			for _, c := range chans {
				c.close(closedByGo)
			}
			cancel()
		}()
	}
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("panic: %v", r)
			log.Printf("mygo: panic in %s.%s: %v\n%s", m.service, m.name, r, debug.Stack())
		}
	}()
	in := make([]reflect.Value, 0, len(m.params)+1)
	if m.ctx {
		in = append(in, reflect.ValueOf(ctx))
	}
	last := len(m.params) - 1
	for i, t := range m.params {
		if m.chans[i] {
			// The page passes the id of its channel.
			var id int64
			if i < len(args) {
				_ = json.Unmarshal(args[i], &id)
			}
			if id <= 0 {
				return nil, fmt.Errorf("invalid argument %d: not a Channel", i+1)
			}
			c := w.newChannel(page, ctx, cancel, id, token)
			chans = append(chans, c)
			v := reflect.New(t.Elem())
			v.Interface().(channelParam).init(c)
			in = append(in, v)
			continue
		}
		if m.variadic && i == last {
			for j := i; j < len(args); j++ {
				v, err := decodeArg(args[j], t.Elem(), j)
				if err != nil {
					return nil, err
				}
				in = append(in, v)
			}
			break
		}
		if i >= len(args) {
			in = append(in, reflect.Zero(t))
			continue
		}
		v, err := decodeArg(args[i], t, i)
		if err != nil {
			return nil, err
		}
		in = append(in, v)
	}
	out := m.fn.Call(in)
	if m.hasErr {
		if e := out[len(out)-1]; !e.IsNil() {
			return nil, e.Interface().(error)
		}
	}
	if m.result != nil {
		return out[0].Interface(), nil
	}
	return nil, nil
}

func decodeArg(raw rawValue, t reflect.Type, i int) (reflect.Value, error) {
	v := reflect.New(t)
	if err := json.Unmarshal(raw, v.Interface(), jsonOptions); err != nil {
		return reflect.Value{}, fmt.Errorf("invalid argument %d: %w", i+1, err)
	}
	return v.Elem(), nil
}

// rawValue is a JSON value that shares the bytes it was decoded from,
// which must outlive it unchanged: arguments, which can be large, go from
// the message to their parameters without being copied.
type rawValue []byte

// UnmarshalJSONFrom keeps the value in place. json.Unmarshal decodes from
// the slice it is given, which it never modifies.
func (v *rawValue) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	raw, err := dec.ReadValue()
	*v = rawValue(raw)
	return err
}

// stringBytes returns the bytes of s without copying them. They must not
// be modified.
func stringBytes(s string) []byte { return unsafe.Slice(unsafe.StringData(s), len(s)) }

// handleCall decodes and runs a call from the page on the current
// goroutine, then queues the reply. Calls from untrusted pages are
// rejected.
func handleCall(w *Window, ctx context.Context, raw string, trusted bool) {
	var m struct {
		ID int64      `json:"id"`
		K  string     `json:"k"`
		M  string     `json:"m"`
		A  []rawValue `json:"a"` // shares raw
	}
	if err := json.Unmarshal(stringBytes(raw), &m); err != nil {
		log.Printf("mygo: malformed call from window %d: %v", w.id, err)
		return
	}
	var result any
	var err error
	if !trusted {
		err = fmt.Errorf("this page is not allowed to call Go methods (see PageOptions.TrustedOrigins)")
	} else if mi := lookupMethod(m.M); mi == nil {
		err = notBound(m.M)
	} else {
		result, err = mi.call(w, ctx, m.K, m.A)
	}
	w.enqueue(encodeReply(m.ID, m.K, result, err), false)
}

var closeBrace = []byte("}")

func encodeReply(id int64, token string, result any, err error) message {
	b := make([]byte, 0, 96)
	b = append(b, `{"t":"reply","id":`...)
	b = strconv.AppendInt(b, id, 10)
	b = append(b, `,"k":`...)
	b, _ = jsontext.AppendQuote(b, token)
	if err == nil {
		v, merr := json.Marshal(result, jsonOptions)
		if merr == nil {
			return message{append(b, `,"ok":true,"v":`...), v, closeBrace}
		}
		err = fmt.Errorf("cannot encode result: %w", merr)
	}
	b = append(b, `,"ok":false,"e":`...)
	b, _ = jsontext.AppendQuote(b, err.Error())
	return message{head: append(b, '}')}
}

type callerKey struct{}

// CallerWindow returns the window whose page called a bound method, given
// the context.Context passed to that method: for the page of a web view
// (WebView), the window it is in. It returns nil for other contexts.
func CallerWindow(ctx context.Context) *Window {
	w, _ := ctx.Value(callerKey{}).(*Window)
	return w
}

// pageKey holds the calling page in the context of a call.
type pageKey struct{}

// CallerPage returns the page that called a bound method, given the
// context.Context passed to that method: the page of the calling window,
// or of a web view in it (WebView), whose window CallerWindow returns. It
// returns nil for other contexts.
func CallerPage(ctx context.Context) *Page {
	p, _ := ctx.Value(pageKey{}).(*Page)
	return p
}

// Event is a typed event the Go side sends to pages. Declare events at
// package level so `mygo generate` includes them in the TypeScript client:
//
//	var Progress = mygo.NewEvent[ProgressInfo]("progress")
//
//	Progress.Emit(win, ProgressInfo{Done: 1, Total: 3})
//
// and subscribe in the frontend with `events.progress.on(p => ...)`.
type Event[T any] struct {
	name string
}

// NewEvent declares an event with payload type T. Names must be unique;
// those starting with "mygo:" are reserved.
func NewEvent[T any](name string) *Event[T] {
	if strings.HasPrefix(name, "mygo:") {
		panic(fmt.Sprintf("mygo: event %q: names starting with \"mygo:\" are reserved", name))
	}
	t := reflect.TypeFor[T]()
	if err := validateType(t); err != nil {
		panic(fmt.Sprintf("mygo: event %q: %v", name, err))
	}
	pc, file, line, _ := runtime.Caller(1)
	ipc.Lock()
	defer ipc.Unlock()
	for _, e := range ipc.events {
		if e.name == name {
			panic(fmt.Sprintf("mygo: event %q is already declared", name))
		}
	}
	ipc.events = append(ipc.events, &eventInfo{name: name, typ: t, pc: pc, file: file, line: line})
	ipc.writeClient = WriteTypeScript
	return &Event[T]{name: name}
}

// Name returns the event name.
func (e *Event[T]) Name() string { return e.name }

// Emit sends the event to the page shown in w, or to the pages of the web
// views of a window showing native UI (WebView). Events sent before the
// page's DOM is ready are delivered once it is.
func (e *Event[T]) Emit(w *Window, payload T) error {
	msg, err := encodeEvent(e.name, payload)
	if err != nil {
		return err
	}
	if w.IsDestroyed() {
		return errDestroyed
	}
	if w.content != nil {
		windows.Lock()
		views := append([]*Window(nil), w.webViews...)
		windows.Unlock()
		for _, v := range views {
			v.enqueue(msg, true)
		}
		return nil
	}
	w.enqueue(msg, true)
	return nil
}

// EmitPage sends the event to a page: of a window, or of a web view
// (WebView.Page).
func (e *Event[T]) EmitPage(p *Page, payload T) error {
	msg, err := encodeEvent(e.name, payload)
	if err != nil {
		return err
	}
	if p == nil || p.w.IsDestroyed() {
		return errDestroyed
	}
	p.w.enqueue(msg, true)
	return nil
}

// Broadcast sends the event to every window, and the web views in them.
func (e *Event[T]) Broadcast(payload T) error {
	msg, err := encodeEvent(e.name, payload)
	if err != nil {
		return err
	}
	for _, w := range pageWindows() {
		w.enqueue(msg, true)
	}
	return nil
}

func encodeEvent(name string, payload any) (message, error) {
	p, err := json.Marshal(payload, jsonOptions)
	if err != nil {
		return message{}, fmt.Errorf("mygo: cannot encode event %q: %w", name, err)
	}
	b := make([]byte, 0, len(name)+32)
	b = append(b, `{"t":"event","n":`...)
	b, _ = jsontext.AppendQuote(b, name)
	b = append(b, `,"p":`...)
	return message{b, p, closeBrace}, nil
}

// The service mygo:window backs window.mygo.window in the page.
func init() {
	err := bindFuncs("mygo:window",
		namedFunc{"Minimize", func(ctx context.Context) { CallerWindow(ctx).Minimize() }},
		namedFunc{"Maximize", func(ctx context.Context) { CallerWindow(ctx).Maximize() }},
		namedFunc{"Unmaximize", func(ctx context.Context) { CallerWindow(ctx).Unmaximize() }},
		namedFunc{"ToggleMaximize", func(ctx context.Context) { CallerWindow(ctx).ToggleMaximize() }},
		namedFunc{"IsMaximized", func(ctx context.Context) bool { return CallerWindow(ctx).IsMaximized() }},
		namedFunc{"ToggleFullScreen", func(ctx context.Context) { CallerWindow(ctx).ToggleFullScreen() }},
		namedFunc{"Close", func(ctx context.Context) { CallerWindow(ctx).Close() }},
		namedFunc{"SetTitle", func(ctx context.Context, title string) { CallerWindow(ctx).SetTitle(title) }},
	)
	if err != nil {
		panic(err)
	}
}
