// Package interp runs shaders on the CPU, for MyGo's CPU renderer: it
// compiles a program of package shaderir into Go closures, which compute
// what the GPU renderers' code computes, within the rounding of their
// floats.
//
// Values are lanes of 32 bits in a register file, floats as their bits,
// ints as their two's complement and bools as 0 and 1; vectors, matrices
// (by columns) and arrays are runs of lanes. Shaders call no function
// recursively, so each function has registers of its own, and running
// one allocates nothing.
package interp

import (
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/egoist/mygo/internal/shaderir"
)

// Image is an image a shader reads: its size in pixels, and its pixels,
// premultiplied, from 0 to 1.
type Image interface {
	Size() (w, h int)
	At(x, y int) [4]float32
}

// Inputs are what a program reads: its uniform variables, as the words of
// their lanes, in the order of the program's, and its images.
type Inputs struct {
	Uniforms []uint32
	Images   []Image
}

// Program is a shader compiled for the CPU. It is safe for concurrent
// use.
type Program struct {
	entry    *function
	position int   // the registers of the entry point's position
	varyings []int // and of its varyings
	words    int   // the uniforms' words
	regs     int
	pool     sync.Pool
}

// state is a run's: its registers, inputs, and whether it discarded the
// pixel.
type state struct {
	r       []uint32
	in      *Inputs
	discard bool
}

// UniformWords returns how many words each of p's uniform variables
// takes in Inputs.Uniforms.
func UniformWords(p *shaderir.Program) []int {
	words := make([]int, len(p.Uniforms))
	for i := range p.Uniforms {
		words[i] = lanes(&p.Uniforms[i])
	}
	return words
}

// New compiles p, a program with a fragment entry point.
func New(p *shaderir.Program) (prog *Program, err error) {
	lib, entry, funcs := p.FragmentLibrary()
	if entry == nil {
		return nil, errors.New("interp: the program has no fragment entry point")
	}
	defer func() {
		if r := recover(); r != nil {
			prog, err = nil, fmt.Errorf("interp: %v", r)
		}
	}()
	c := &compiler{p: lib, funcs: map[int]*function{}}
	for _, w := range UniformWords(lib) {
		c.uniforms = append(c.uniforms, c.words)
		c.words += w
	}
	// Registers for every function first, so that bodies can call any.
	for _, f := range funcs {
		c.funcs[f.Index] = c.declare(f)
	}
	for _, f := range funcs {
		c.define(c.funcs[f.Index])
	}
	e := c.funcs[entry.Index]
	prog = &Program{entry: e, position: e.params[0].off, words: c.words, regs: c.size}
	for i := 1; i < len(entry.InParams); i++ {
		prog.varyings = append(prog.varyings, e.params[i].off)
	}
	prog.pool.New = func() any { return &state{r: make([]uint32, prog.regs)} }
	return prog, nil
}

// Color returns the color the entry point returns for the pixel at
// position, with the varyings' lanes in order, and false where it
// discards the pixel.
func (p *Program) Color(in *Inputs, position [4]float32, varyings ...float32) ([4]float32, bool) {
	s := p.pool.Get().(*state)
	defer p.pool.Put(s)
	if len(in.Uniforms) < p.words {
		return [4]float32{}, false
	}
	s.in, s.discard = in, false
	for i, v := range position {
		s.r[p.position+i] = math.Float32bits(v)
	}
	// The varyings' lanes follow one another in the arguments.
	lane := 0
	for vi, off := range p.varyings {
		n := p.entry.params[vi+1].n
		for i := range n {
			if lane < len(varyings) {
				s.r[off+i] = math.Float32bits(varyings[lane])
			}
			lane++
		}
	}
	p.entry.run(s)
	if s.discard {
		return [4]float32{}, false
	}
	var out [4]float32
	for i := range out {
		out[i] = math.Float32frombits(s.r[p.entry.ret.off+i])
	}
	return out, true
}

// lanes returns how many lanes a value of type t takes.
func lanes(t *shaderir.Type) int {
	switch t.Main {
	case shaderir.Bool, shaderir.Int, shaderir.Uint, shaderir.Float, shaderir.Texture:
		return 1
	case shaderir.Vec2, shaderir.IVec2, shaderir.UVec2:
		return 2
	case shaderir.Vec3, shaderir.IVec3, shaderir.UVec3:
		return 3
	case shaderir.Vec4, shaderir.IVec4, shaderir.UVec4, shaderir.Mat2:
		return 4
	case shaderir.Mat3:
		return 9
	case shaderir.Mat4:
		return 16
	case shaderir.Array:
		return t.Length * lanes(&t.Sub[0])
	case shaderir.Struct:
		n := 0
		for i := range t.Sub {
			n += lanes(&t.Sub[i])
		}
		return n
	}
	return 0
}

// slot is where a value lies in the registers.
type slot struct {
	off, n int
	t      shaderir.Type
}

// function is a function compiled: the registers of its parameters,
// inputs then outputs, and of its result, and its body. Its other locals
// have registers in the blocks declaring them (compiler.scopes): an index
// names a local of the innermost block around that declares it, as the
// backends declare them, in blocks of C.
type function struct {
	f      *shaderir.Func
	params []slot
	ret    slot
	body   []stmt
}

// control tells what a statement did: went on, or broke out of a loop,
// went on to its next iteration, returned or discarded the pixel.
type control uint8

const (
	next control = iota
	breakLoop
	continueLoop
	returned
)

type stmt func(s *state) control

func (f *function) run(s *state) {
	for _, st := range f.body {
		if st(s) != next || s.discard {
			return
		}
	}
}

type compiler struct {
	p        *shaderir.Program
	funcs    map[int]*function
	uniforms []int // each uniform's first word
	words    int
	size     int
	// scopes are the locals of the blocks around what is being compiled,
	// the innermost last.
	scopes []map[int]slot
}

// local returns the registers of local index of fn where it is being
// compiled.
func (c *compiler) local(fn *function, index int) slot {
	for i := len(c.scopes) - 1; i >= 0; i-- {
		if l, ok := c.scopes[i][index]; ok {
			return l
		}
	}
	if index < len(fn.params) {
		return fn.params[index]
	}
	panic(fmt.Sprintf("no local %d", index))
}

func (c *compiler) alloc(t shaderir.Type) slot {
	n := lanes(&t)
	s := slot{off: c.size, n: n, t: t}
	c.size += n
	return s
}

// declare gives f's parameters and result registers of their own.
func (c *compiler) declare(f *shaderir.Func) *function {
	fn := &function{f: f}
	for _, t := range append(append([]shaderir.Type(nil), f.InParams...), f.OutParams...) {
		fn.params = append(fn.params, c.alloc(t))
	}
	if f.Return.Main != shaderir.None {
		fn.ret = c.alloc(f.Return)
	}
	return fn
}

func (c *compiler) define(fn *function) {
	fn.body = c.block(fn, fn.f.Block)
}

// env is a function's types.
type env struct {
	c  *compiler
	fn *function
}

func (e env) LocalType(index int) shaderir.Type   { return e.c.local(e.fn, index).t }
func (e env) UniformType(index int) shaderir.Type { return e.c.p.Uniforms[index] }
func (e env) FuncReturn(index int) shaderir.Type  { return e.c.funcs[index].f.Return }
