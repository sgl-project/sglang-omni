//go:build js

package gl

import (
	"errors"
	"syscall/js"
	"unsafe"
)

// Context is the WebGL 2 context of a canvas, which renderers draw with
// while it is current (MakeCurrent), as a GL context on Linux. WebGL names
// its objects by JavaScript objects, GL by numbers: the package numbers
// them, and the functions GL's renderer calls (glCreateShader, …) take and
// return those numbers.
type Context struct {
	v  js.Value
	id uintptr
	// dual tells that the context has WEBGL_blend_func_extended, enabled.
	dual bool
	// The pixel store's unpacking, which uploads read Go's pixels by.
	unpackAlign, unpackRow int32
	// scratch is a Uint8Array that Go's bytes are copied into for WebGL
	// to read, grown as needed.
	scratch js.Value
	size    int
}

var (
	// cur is the current context.
	cur *Context
	// objects are the WebGL objects by their numbers, which their
	// property objectKey holds too; locations the uniform locations.
	objects   = map[uint32]js.Value{}
	locations = []js.Value{}
	lastName  uint32
	lastID    uintptr
	// errLoad is load's error on Linux; the browser has nothing to load.
	errLoad error
)

// objectKey is the property of a WebGL object holding its number.
const objectKey = "__mygo"

// NewContext makes the WebGL 2 context of canvas, which draws premultiplied
// colors without depth, stencil or antialiasing, as frames need: the
// renderer cuts shapes' edges itself. It fails where the browser has no
// WebGL 2, or will not make a context for the canvas.
func NewContext(canvas js.Value) (*Context, error) {
	attrs := map[string]any{
		"alpha": true, "premultipliedAlpha": true, "antialias": false, "depth": false, "stencil": false,
		"preserveDrawingBuffer": false, "powerPreference": "default",
	}
	v := canvas.Call("getContext", "webgl2", attrs)
	if v.IsNull() || v.IsUndefined() {
		return nil, errors.New("gl: the browser has no WebGL 2")
	}
	c := &Context{v: v, unpackAlign: 4}
	c.Renew()
	return c, nil
}

// Renew numbers the context anew, as after the browser restored it once it
// was lost: renderers then make their objects again, the old ones gone
// with the context.
func (c *Context) Renew() {
	lastID++
	c.id = lastID
	c.dual = !c.v.Call("getExtension", "WEBGL_blend_func_extended").IsNull()
	c.unpackAlign, c.unpackRow = 4, 0
}

// ID returns the context's number, which changes with Renew.
func (c *Context) ID() uintptr { return c.id }

// Lost reports whether the browser took the context away.
func (c *Context) Lost() bool { return c.v.Call("isContextLost").Bool() }

// MakeCurrent makes c the context renderers draw with.
func (c *Context) MakeCurrent() { cur = c }

func load() error {
	if cur == nil {
		return errors.New("gl: no WebGL context is current")
	}
	return nil
}

func currentContext() uintptr {
	if cur == nil {
		return 0
	}
	return cur.id
}

func contextVersion() (version int32, es bool) { return 30, true }

func hasExtension(name string) bool {
	return name == "GL_EXT_blend_func_extended" && cur.dual
}

func versionString() string { return cur.v.Call("getParameter", glVersion).String() }

// rendererString names the context's renderer: as the browser names it
// (Firefox, the GPU's, sanitized), or, where that is WebGL's own name
// (Chrome and Safari), as the GPU's driver does, should the browser tell
// (WEBGL_debug_renderer_info, which Firefox deprecates).
func rendererString() string {
	const unmaskedRenderer = 0x9246
	name := cur.v.Call("getParameter", glRenderer).String()
	if name == "WebKit WebGL" && !cur.v.Call("getExtension", "WEBGL_debug_renderer_info").IsNull() {
		if v := cur.v.Call("getParameter", unmaskedRenderer); v.Type() == js.TypeString {
			return v.String()
		}
	}
	return name
}

func call(name string, args ...any) js.Value { return cur.v.Call(name, args...) }

// name numbers a new WebGL object.
func name(v js.Value) uint32 {
	if v.IsNull() {
		return 0
	}
	lastName++
	objects[lastName] = v
	v.Set(objectKey, lastName)
	return lastName
}

// object returns the WebGL object of a number, null for 0.
func object(n uint32) js.Value {
	if n == 0 {
		return js.Null()
	}
	if v, ok := objects[n]; ok {
		return v
	}
	return js.Null()
}

// number returns the number of a WebGL object, 0 for null.
func number(v js.Value) uint32 {
	if v.IsNull() || v.IsUndefined() {
		return 0
	}
	return uint32(v.Get(objectKey).Int())
}

func deleteObjects(fn string, n int32, names *uint32) {
	for _, k := range unsafe.Slice(names, n) {
		if v, ok := objects[k]; ok {
			call(fn, v)
			delete(objects, k)
		}
	}
}

func genObjects(fn string, n int32, out *uint32) {
	names := unsafe.Slice(out, n)
	for i := range names {
		names[i] = name(call(fn))
	}
}

// jsBytes copies n bytes at p into the scratch array, for WebGL to read.
func jsBytes(p unsafe.Pointer, n int) js.Value {
	c := cur
	if c.size < n {
		c.size = max(n, 2*c.size, 1<<16)
		c.scratch = js.Global().Get("Uint8Array").New(c.size)
	}
	js.CopyBytesToJS(c.scratch, unsafe.Slice((*byte)(p), n))
	return c.scratch
}

// uploadSize returns how many bytes an upload of w×h pixels reads, with
// the pixel store's unpacking.
func uploadSize(w, h int32, format uint32) int {
	bpp := int32(4)
	if format == glRed {
		bpp = 1
	}
	row := w
	if cur.unpackRow > 0 {
		row = cur.unpackRow
	}
	a := cur.unpackAlign
	stride := (row*bpp + a - 1) / a * a
	return int((h-1)*stride + w*bpp)
}

func setShaderSource(s uint32, src string) { call("shaderSource", object(s), src) }

func shaderLog(s uint32) string  { return call("getShaderInfoLog", object(s)).String() }
func programLog(p uint32) string { return call("getProgramInfoLog", object(p)).String() }

func glGetError() uint32 { return uint32(call("getError").Int()) }

func glGetIntegerv(param uint32, data *int32) {
	v := call("getParameter", param)
	switch param {
	case glFramebufferBinding:
		*data = int32(number(v))
	default:
		*data = int32(v.Int())
	}
}

func glCreateShader(typ uint32) uint32 { return name(call("createShader", typ)) }
func glCompileShader(s uint32)         { call("compileShader", object(s)) }
func glDeleteShader(s uint32)          { deleteObjects("deleteShader", 1, &s) }

func glGetShaderiv(s, param uint32, out *int32) {
	*out = boolInt(call("getShaderParameter", object(s), param).Truthy())
}

func glCreateProgram() uint32            { return name(call("createProgram")) }
func glAttachShader(program, s uint32)   { call("attachShader", object(program), object(s)) }
func glLinkProgram(program uint32)       { call("linkProgram", object(program)) }
func glUseProgram(program uint32)        { call("useProgram", object(program)) }
func glDeleteProgram(program uint32)     { deleteObjects("deleteProgram", 1, &program) }
func glBindVertexArray(array uint32)     { call("bindVertexArray", object(array)) }
func glBindBuffer(target, buffer uint32) { call("bindBuffer", target, object(buffer)) }

func glGetProgramiv(program, param uint32, out *int32) {
	*out = boolInt(call("getProgramParameter", object(program), param).Truthy())
}

func glGetUniformLocation(program uint32, uniform string) int32 {
	v := call("getUniformLocation", object(program), uniform)
	if v.IsNull() {
		return -1
	}
	locations = append(locations, v)
	return int32(len(locations) - 1)
}

func glGetUniformBlockIndex(program uint32, block string) uint32 {
	return uint32(call("getUniformBlockIndex", object(program), block).Int())
}

func glUniformBlockBinding(program, index, binding uint32) {
	call("uniformBlockBinding", object(program), index, binding)
}

func glBindBufferBase(target, index, buffer uint32) {
	call("bindBufferBase", target, index, object(buffer))
}

func location(l int32) js.Value {
	if l < 0 || int(l) >= len(locations) {
		return js.Null()
	}
	return locations[l]
}

func glUniform1i(l, v int32)            { call("uniform1i", location(l), v) }
func glUniform2f(l int32, x, y float32) { call("uniform2f", location(l), x, y) }
func glUniform1f(l int32, v float32)    { call("uniform1f", location(l), v) }
func glUniform2i(l, x, y int32)         { call("uniform2i", location(l), x, y) }
func glUniform4i(l, x, y, z, w int32)   { call("uniform4i", location(l), x, y, z, w) }

func glCopyTexSubImage2D(target uint32, level, xoff, yoff, x, y, w, h int32) {
	call("copyTexSubImage2D", target, level, xoff, yoff, x, y, w, h)
}

func glGenVertexArrays(n int32, out *uint32)       { genObjects("createVertexArray", n, out) }
func glDeleteVertexArrays(n int32, arrays *uint32) { deleteObjects("deleteVertexArray", n, arrays) }
func glGenBuffers(n int32, out *uint32)            { genObjects("createBuffer", n, out) }
func glDeleteBuffers(n int32, buffers *uint32)     { deleteObjects("deleteBuffer", n, buffers) }
func glGenTextures(n int32, out *uint32)           { genObjects("createTexture", n, out) }
func glDeleteTextures(n int32, textures *uint32)   { deleteObjects("deleteTexture", n, textures) }
func glGenFramebuffers(n int32, out *uint32)       { genObjects("createFramebuffer", n, out) }
func glDeleteFramebuffers(n int32, fbs *uint32)    { deleteObjects("deleteFramebuffer", n, fbs) }

func glBufferData(target uint32, size int, data unsafe.Pointer, usage uint32) {
	if data == nil {
		call("bufferData", target, size, usage)
		return
	}
	call("bufferData", target, jsBytes(data, size), usage, 0, size)
}

func glEnableVertexAttrib(index uint32) { call("enableVertexAttribArray", index) }

func glVertexAttribPointer(index uint32, size int32, typ uint32, normalized bool, stride int32, offset uintptr) {
	call("vertexAttribPointer", index, size, typ, normalized, stride, offset)
}

func glVertexAttribDivisor(index, divisor uint32) { call("vertexAttribDivisor", index, divisor) }
func glBindTexture(target, texture uint32)        { call("bindTexture", target, object(texture)) }
func glActiveTexture(unit uint32)                 { call("activeTexture", unit) }

func glTexImage2D(target uint32, level, internalFormat, w, h, border int32, format, typ uint32, pix unsafe.Pointer) {
	if pix == nil {
		call("texImage2D", target, level, internalFormat, w, h, border, format, typ, js.Null())
		return
	}
	call("texImage2D", target, level, internalFormat, w, h, border, format, typ, jsPixels(pix, w, h, format, typ), 0)
}

func glTexSubImage2D(target uint32, level, x, y, w, h int32, format, typ uint32, pix unsafe.Pointer) {
	call("texSubImage2D", target, level, x, y, w, h, format, typ, jsPixels(pix, w, h, format, typ), 0)
}

// jsPixels returns the pixels of an upload as WebGL takes them: bytes, or
// for floats (glFloat) a Float32Array over them.
func jsPixels(pix unsafe.Pointer, w, h int32, format, typ uint32) js.Value {
	n := uploadSize(w, h, format)
	if typ != glFloat {
		return jsBytes(pix, n)
	}
	b := jsBytes(pix, 4*n)
	return js.Global().Get("Float32Array").New(b.Get("buffer"), 0, n)
}

func glTexParameteri(target, param uint32, v int32) { call("texParameteri", target, param, v) }

func glPixelStorei(param uint32, v int32) {
	switch param {
	case glUnpackAlignment:
		cur.unpackAlign = v
	case glUnpackRowLength:
		cur.unpackRow = v
	}
	call("pixelStorei", param, v)
}

func glViewport(x, y, w, h int32)         { call("viewport", x, y, w, h) }
func glScissor(x, y, w, h int32)          { call("scissor", x, y, w, h) }
func glEnable(capability uint32)          { call("enable", capability) }
func glDisable(capability uint32)         { call("disable", capability) }
func glBlendFunc(src, dst uint32)         { call("blendFunc", src, dst) }
func glClearColor(r, g, b, a float32)     { call("clearColor", r, g, b, a) }
func glClear(mask uint32)                 { call("clear", mask) }
func glBindFramebuffer(target, fb uint32) { call("bindFramebuffer", target, object(fb)) }
func glCheckFramebufferState(target uint32) uint32 {
	return uint32(call("checkFramebufferStatus", target).Int())
}

func glBlendFuncSeparate(srcRGB, dstRGB, srcAlpha, dstAlpha uint32) {
	call("blendFuncSeparate", srcRGB, dstRGB, srcAlpha, dstAlpha)
}

func glDrawArraysInstanced(mode uint32, first, count, instances int32) {
	call("drawArraysInstanced", mode, first, count, instances)
}

func glReadPixels(x, y, w, h int32, format, typ uint32, pix unsafe.Pointer) {
	bpp := int32(4)
	if format == glRed {
		bpp = 1
	}
	n := int(w * h * bpp)
	arr := js.Global().Get("Uint8Array").New(n)
	call("readPixels", x, y, w, h, format, typ, arr)
	js.CopyBytesToGo(unsafe.Slice((*byte)(pix), n), arr)
}

func glFramebufferTexture2D(target, attachment, texTarget, texture uint32, level int32) {
	call("framebufferTexture2D", target, attachment, texTarget, object(texture), level)
}

func glBlitFramebuffer(sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1 int32, mask, filter uint32) {
	call("blitFramebuffer", sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1, mask, filter)
}

func boolInt(b bool) int32 {
	if b {
		return 1
	}
	return 0
}
