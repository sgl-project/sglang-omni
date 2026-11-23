package interp

import (
	"fmt"
	"go/constant"
	"math"

	"github.com/egoist/mygo/internal/shaderir"
)

// maxIterations bounds a loop, as a GPU's watchdog would: a shader whose
// loop does not end draws what it has.
const maxIterations = 1 << 20

// operand is an expression compiled: once run has run (nil for a value
// already in place, as a local's), its value lies in the registers from
// off, of type t.
type operand struct {
	run func(s *state)
	off int
	t   shaderir.Type
}

func (o operand) n() int { return lanes(&o.t) }

// eval runs the operands' computations.
func eval(s *state, os ...operand) {
	for _, o := range os {
		if o.run != nil {
			o.run(s)
		}
	}
}

func (c *compiler) block(fn *function, b *shaderir.Block) []stmt {
	if b == nil {
		return nil
	}
	// The block's locals, a loop's counter among them, which has the type
	// of its loop, start at zero, as the backends declare them.
	scope := map[int]slot{}
	var zero []slot
	for i, t := range b.LocalVars {
		index := b.LocalVarIndexOffset + i
		if t.Main == shaderir.None {
			for _, st := range b.Stmts {
				if st.Type == shaderir.For && st.ForVarIndex == index {
					t = st.ForVarType
				}
			}
			scope[index] = c.alloc(t)
			continue
		}
		scope[index] = c.alloc(t)
		zero = append(zero, scope[index])
	}
	c.scopes = append(c.scopes, scope)
	defer func() { c.scopes = c.scopes[:len(c.scopes)-1] }()

	var out []stmt
	if len(zero) > 0 {
		out = append(out, func(s *state) control {
			for _, z := range zero {
				clear(s.r[z.off : z.off+z.n])
			}
			return next
		})
	}
	for i := range b.Stmts {
		if st := c.stmt(fn, &b.Stmts[i]); st != nil {
			out = append(out, st)
		}
	}
	return out
}

func runBlock(s *state, body []stmt) control {
	for _, st := range body {
		if ctl := st(s); ctl != next {
			return ctl
		}
		if s.discard {
			return returned
		}
	}
	return next
}

func (c *compiler) stmt(fn *function, st *shaderir.Stmt) stmt {
	switch st.Type {
	case shaderir.ExprStmt:
		o := c.expr(fn, &st.Exprs[0])
		if o.run == nil {
			return nil
		}
		return func(s *state) control { o.run(s); return next }
	case shaderir.BlockStmt:
		body := c.block(fn, st.Blocks[0])
		return func(s *state) control { return runBlock(s, body) }
	case shaderir.Assign:
		return c.assign(fn, &st.Exprs[0], &st.Exprs[1])
	case shaderir.Init:
		l := c.local(fn, st.InitIndex)
		return func(s *state) control { clear(s.r[l.off : l.off+l.n]); return next }
	case shaderir.If:
		cond := c.expr(fn, &st.Exprs[0])
		then := c.block(fn, st.Blocks[0])
		var els []stmt
		if len(st.Blocks) > 1 {
			els = c.block(fn, st.Blocks[1])
		}
		return func(s *state) control {
			eval(s, cond)
			if s.r[cond.off] != 0 {
				return runBlock(s, then)
			}
			return runBlock(s, els)
		}
	case shaderir.For:
		return c.loop(fn, st)
	case shaderir.Continue:
		return func(*state) control { return continueLoop }
	case shaderir.Break:
		return func(*state) control { return breakLoop }
	case shaderir.Return:
		if len(st.Exprs) == 0 {
			return func(*state) control { return returned }
		}
		o := c.expr(fn, &st.Exprs[0])
		ret := fn.ret
		return func(s *state) control {
			eval(s, o)
			copy(s.r[ret.off:ret.off+ret.n], s.r[o.off:o.off+ret.n])
			return returned
		}
	case shaderir.Discard:
		return func(s *state) control { s.discard = true; return returned }
	}
	panic(fmt.Sprintf("unexpected statement %d", st.Type))
}

// loop compiles a for loop: its counter goes from the start by the step
// while it compares with the end, evaluated before each iteration.
func (c *compiler) loop(fn *function, st *shaderir.Stmt) stmt {
	counter := c.local(fn, st.ForVarIndex)
	var start, end operand
	if len(st.Exprs) == 2 {
		start, end = c.expr(fn, &st.Exprs[0]), c.expr(fn, &st.Exprs[1])
	} else {
		start = c.constant(st.ForInit, st.ForVarType)
		end = c.constant(st.ForEnd, st.ForVarType)
	}
	delta := c.constant(st.ForDelta, st.ForVarType)
	kind := st.ForVarType.Scalar()
	add := arith(shaderir.Add, kind)
	cmp := compare(st.ForOp, kind)
	body := c.block(fn, st.Blocks[0])
	return func(s *state) control {
		eval(s, start, delta)
		s.r[counter.off] = s.r[start.off]
		for i := 0; i < maxIterations; i++ {
			eval(s, end)
			if cmp(s.r[counter.off], s.r[end.off]) == 0 {
				break
			}
			switch runBlock(s, body) {
			case breakLoop:
				return next
			case returned:
				return returned
			}
			s.r[counter.off] = add(s.r[counter.off], s.r[delta.off])
		}
		return next
	}
}

// assign compiles lhs = rhs: lhs a local, an element of one or a swizzle
// of one, at any depth.
func (c *compiler) assign(fn *function, lhs, rhs *shaderir.Expr) stmt {
	src := c.expr(fn, rhs)
	n := src.n()
	if lhs.Type == shaderir.LocalVariable {
		dst := c.local(fn, lhs.Index)
		return func(s *state) control {
			eval(s, src)
			copy(s.r[dst.off:dst.off+n], s.r[src.off:src.off+n])
			return next
		}
	}
	// The value is copied first, as the target may overlap it.
	tmp := c.alloc(src.t)
	write := c.lvalue(fn, lhs)
	return func(s *state) control {
		eval(s, src)
		copy(s.r[tmp.off:tmp.off+n], s.r[src.off:src.off+n])
		write(s, tmp.off)
		return next
	}
}

// lvalue returns what writes the value at src to e's lanes.
func (c *compiler) lvalue(fn *function, e *shaderir.Expr) func(s *state, src int) {
	switch e.Type {
	case shaderir.LocalVariable:
		l := c.local(fn, e.Index)
		return func(s *state, src int) { copy(s.r[l.off:l.off+l.n], s.r[src:src+l.n]) }
	case shaderir.Index:
		base := &e.Exprs[0]
		bt := shaderir.TypeOf(base, env{c, fn})
		et := shaderir.TypeOf(e, env{c, fn})
		size, count := lanes(&et), elements(&bt)
		idx := c.expr(fn, &e.Exprs[1])
		unsigned := shaderir.TypeOf(&e.Exprs[1], env{c, fn}).Scalar() == shaderir.Uint
		// Write the element into a copy of the base, then the base back.
		tmp := c.alloc(bt)
		read := c.expr(fn, base)
		write := c.lvalue(fn, base)
		bn := lanes(&bt)
		return func(s *state, src int) {
			eval(s, read, idx)
			copy(s.r[tmp.off:tmp.off+bn], s.r[read.off:read.off+bn])
			i := clampIndex(indexAt(s.r[idx.off], unsigned), count)
			copy(s.r[tmp.off+i*size:tmp.off+(i+1)*size], s.r[src:src+size])
			write(s, tmp.off)
		}
	case shaderir.FieldSelector:
		base := &e.Exprs[0]
		bt := shaderir.TypeOf(base, env{c, fn})
		m := e.Exprs[1]
		tmp := c.alloc(bt)
		read := c.expr(fn, base)
		write := c.lvalue(fn, base)
		bn := lanes(&bt)
		var targets []int
		if m.Type == shaderir.SwizzlingExpr {
			targets = swizzle(m.Swizzling)
		} else {
			off := memberOffset(&bt, m.Index)
			for i := range lanes(&bt.Sub[m.Index]) {
				targets = append(targets, off+i)
			}
		}
		return func(s *state, src int) {
			eval(s, read)
			copy(s.r[tmp.off:tmp.off+bn], s.r[read.off:read.off+bn])
			for i, t := range targets {
				s.r[tmp.off+t] = s.r[src+i]
			}
			write(s, tmp.off)
		}
	}
	panic(fmt.Sprintf("cannot assign to expression %d", e.Type))
}

// elements returns how many elements indexing t reaches.
func elements(t *shaderir.Type) int {
	switch {
	case t.Main == shaderir.Array:
		return t.Length
	case t.IsMatrix():
		return t.MatrixSize()
	}
	return t.VectorElementCount()
}

// clampIndex keeps an index within count, where a GPU's would be
// undefined.
func clampIndex(i int64, count int) int {
	return int(min(max(i, 0), int64(count-1)))
}

// indexAt returns the index in r, a uint's bits when unsigned, else an
// int's.
func indexAt(r uint32, unsigned bool) int64 {
	if unsigned {
		return int64(r)
	}
	return int64(int32(r))
}

func swizzle(s string) []int {
	out := make([]int, len(s))
	for i, r := range s {
		switch r {
		case 'x':
			out[i] = 0
		case 'y':
			out[i] = 1
		case 'z':
			out[i] = 2
		case 'w':
			out[i] = 3
		}
	}
	return out
}

func memberOffset(t *shaderir.Type, index int) int {
	off := 0
	for i := 0; i < index; i++ {
		off += lanes(&t.Sub[i])
	}
	return off
}

// constant returns an operand of the constant v as type t's kind.
func (c *compiler) constant(v constant.Value, t shaderir.Type) operand {
	st := shaderir.Type{Main: t.Scalar()}
	o := c.alloc(st)
	bits := constBits(v, st.Main)
	return operand{off: o.off, t: st, run: func(s *state) { s.r[o.off] = bits }}
}

func constBits(v constant.Value, kind shaderir.BasicType) uint32 {
	switch kind {
	case shaderir.Float:
		f, _ := constant.Float64Val(constant.ToFloat(v))
		return math.Float32bits(float32(f))
	case shaderir.Bool:
		if constant.BoolVal(v) {
			return 1
		}
		return 0
	}
	if x, ok := constant.Int64Val(constant.ToInt(v)); ok {
		return uint32(x)
	}
	x, _ := constant.Uint64Val(constant.ToInt(v))
	return uint32(x)
}

func (c *compiler) expr(fn *function, e *shaderir.Expr) operand {
	en := env{c, fn}
	switch e.Type {
	case shaderir.NumberExpr:
		return c.constant(e.Const, shaderir.TypeOf(e, en))
	case shaderir.LocalVariable:
		l := c.local(fn, e.Index)
		return operand{off: l.off, t: l.t}
	case shaderir.UniformVariable:
		t := c.p.Uniforms[e.Index]
		o := c.alloc(t)
		w, n := c.uniforms[e.Index], o.n
		return operand{off: o.off, t: t, run: func(s *state) {
			copy(s.r[o.off:o.off+n], s.in.Uniforms[w:w+n])
		}}
	case shaderir.TextureVariable:
		o := c.alloc(shaderir.Type{Main: shaderir.Texture})
		i := uint32(e.Index)
		return operand{off: o.off, t: o.t, run: func(s *state) { s.r[o.off] = i }}
	case shaderir.Unary:
		return c.unary(fn, e)
	case shaderir.Binary:
		return c.binary(fn, e)
	case shaderir.Selection:
		cond, a, b := c.expr(fn, &e.Exprs[0]), c.expr(fn, &e.Exprs[1]), c.expr(fn, &e.Exprs[2])
		o := c.alloc(a.t)
		n := o.n
		return operand{off: o.off, t: a.t, run: func(s *state) {
			eval(s, cond, a, b)
			src := b.off
			if s.r[cond.off] != 0 {
				src = a.off
			}
			copy(s.r[o.off:o.off+n], s.r[src:src+n])
		}}
	case shaderir.Call:
		if e.Exprs[0].Type == shaderir.FunctionExpr {
			return c.call(fn, e)
		}
		return c.builtin(fn, e)
	case shaderir.FieldSelector:
		base := c.expr(fn, &e.Exprs[0])
		t := shaderir.TypeOf(e, en)
		m := e.Exprs[1]
		var from []int
		if m.Type == shaderir.SwizzlingExpr {
			from = swizzle(m.Swizzling)
		} else {
			off := memberOffset(&base.t, m.Index)
			for i := range lanes(&t) {
				from = append(from, off+i)
			}
		}
		// Lanes in order are read in place.
		inOrder := true
		for i := range from {
			inOrder = inOrder && from[i] == from[0]+i
		}
		if inOrder {
			return operand{run: base.run, off: base.off + from[0], t: t}
		}
		o := c.alloc(t)
		return operand{off: o.off, t: t, run: func(s *state) {
			eval(s, base)
			for i, f := range from {
				s.r[o.off+i] = s.r[base.off+f]
			}
		}}
	case shaderir.Index:
		return c.index(fn, e)
	}
	panic(fmt.Sprintf("unexpected expression %d", e.Type))
}

// index compiles an element of an array, a vector or a matrix: in place
// for a constant index, copied for another.
func (c *compiler) index(fn *function, e *shaderir.Expr) operand {
	en := env{c, fn}
	t := shaderir.TypeOf(e, en)
	bt := shaderir.TypeOf(&e.Exprs[0], en)
	size, count := lanes(&t), elements(&bt)
	ie := &e.Exprs[1]
	unsigned := shaderir.TypeOf(ie, en).Scalar() == shaderir.Uint
	// An element of a uniform is read from the uniforms, not the whole.
	if e.Exprs[0].Type == shaderir.UniformVariable {
		w := c.uniforms[e.Exprs[0].Index]
		idx := c.expr(fn, ie)
		o := c.alloc(t)
		return operand{off: o.off, t: t, run: func(s *state) {
			eval(s, idx)
			i := clampIndex(indexAt(s.r[idx.off], unsigned), count)
			copy(s.r[o.off:o.off+size], s.in.Uniforms[w+i*size:w+(i+1)*size])
		}}
	}
	base := c.expr(fn, &e.Exprs[0])
	if ie.Type == shaderir.NumberExpr {
		i64, _ := constant.Int64Val(constant.ToInt(ie.Const))
		i := clampIndex(i64, count)
		return operand{run: base.run, off: base.off + i*size, t: t}
	}
	idx := c.expr(fn, ie)
	o := c.alloc(t)
	return operand{off: o.off, t: t, run: func(s *state) {
		eval(s, base, idx)
		i := clampIndex(indexAt(s.r[idx.off], unsigned), count)
		copy(s.r[o.off:o.off+size], s.r[base.off+i*size:base.off+(i+1)*size])
	}}
}

// call compiles a call of a function of the program: the arguments go to
// its parameters, and its output parameters to the caller's locals after.
func (c *compiler) call(fn *function, e *shaderir.Expr) operand {
	callee := c.funcs[e.Exprs[0].Index]
	if callee == nil {
		panic("call of a function not compiled")
	}
	in := len(callee.f.InParams)
	var args []operand
	for i := 1; i <= in; i++ {
		args = append(args, c.expr(fn, &e.Exprs[i]))
	}
	// The arguments are computed before any goes to the parameters, as one
	// may call the same function.
	tmps := make([]slot, len(args))
	for i, a := range args {
		tmps[i] = c.alloc(a.t)
	}
	var outs []slot
	for i := in + 1; i < len(e.Exprs); i++ {
		outs = append(outs, c.local(fn, e.Exprs[i].Index))
	}
	var o slot
	if callee.f.Return.Main != shaderir.None {
		o = c.alloc(callee.f.Return)
	}
	return operand{off: o.off, t: callee.f.Return, run: func(s *state) {
		for i, a := range args {
			eval(s, a)
			copy(s.r[tmps[i].off:tmps[i].off+tmps[i].n], s.r[a.off:a.off+tmps[i].n])
		}
		for i, t := range tmps {
			p := callee.params[i]
			copy(s.r[p.off:p.off+p.n], s.r[t.off:t.off+t.n])
		}
		callee.run(s)
		for i, out := range outs {
			p := callee.params[in+i]
			copy(s.r[out.off:out.off+out.n], s.r[p.off:p.off+p.n])
		}
		if o.n > 0 {
			copy(s.r[o.off:o.off+o.n], s.r[callee.ret.off:callee.ret.off+o.n])
		}
	}}
}
