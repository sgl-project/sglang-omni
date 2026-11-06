package interp

import (
	"fmt"
	"math"

	"github.com/egoist/mygo/internal/shaderir"
)

func f32(b uint32) float32  { return math.Float32frombits(b) }
func bits(f float32) uint32 { return math.Float32bits(f) }

func boolBits(b bool) uint32 {
	if b {
		return 1
	}
	return 0
}

// arith returns op on two lanes of kind.
func arith(op shaderir.Op, kind shaderir.BasicType) func(a, b uint32) uint32 {
	switch kind {
	case shaderir.Float:
		switch op {
		case shaderir.Add:
			return func(a, b uint32) uint32 { return bits(f32(a) + f32(b)) }
		case shaderir.Sub:
			return func(a, b uint32) uint32 { return bits(f32(a) - f32(b)) }
		case shaderir.ComponentWiseMul, shaderir.MatrixMul:
			return func(a, b uint32) uint32 { return bits(f32(a) * f32(b)) }
		case shaderir.Div:
			return func(a, b uint32) uint32 { return bits(f32(a) / f32(b)) }
		}
	case shaderir.Int:
		switch op {
		case shaderir.Add:
			return func(a, b uint32) uint32 { return uint32(int32(a) + int32(b)) }
		case shaderir.Sub:
			return func(a, b uint32) uint32 { return uint32(int32(a) - int32(b)) }
		case shaderir.ComponentWiseMul, shaderir.MatrixMul:
			return func(a, b uint32) uint32 { return uint32(int32(a) * int32(b)) }
		case shaderir.Div:
			// Dividing by zero is undefined on GPUs; here it gives zero.
			return func(a, b uint32) uint32 {
				if b == 0 || int32(a) == math.MinInt32 && int32(b) == -1 {
					return 0
				}
				return uint32(int32(a) / int32(b))
			}
		case shaderir.ModOp:
			return func(a, b uint32) uint32 {
				if b == 0 || int32(b) == -1 {
					return 0
				}
				return uint32(int32(a) % int32(b))
			}
		}
	case shaderir.Uint:
		switch op {
		case shaderir.Add:
			return func(a, b uint32) uint32 { return a + b }
		case shaderir.Sub:
			return func(a, b uint32) uint32 { return a - b }
		case shaderir.ComponentWiseMul, shaderir.MatrixMul:
			return func(a, b uint32) uint32 { return a * b }
		case shaderir.Div:
			return func(a, b uint32) uint32 {
				if b == 0 {
					return 0
				}
				return a / b
			}
		case shaderir.ModOp:
			return func(a, b uint32) uint32 {
				if b == 0 {
					return 0
				}
				return a % b
			}
		}
	}
	switch op {
	case shaderir.And:
		return func(a, b uint32) uint32 { return a & b }
	case shaderir.AndNot:
		return func(a, b uint32) uint32 { return a &^ b }
	case shaderir.Or:
		return func(a, b uint32) uint32 { return a | b }
	case shaderir.Xor:
		return func(a, b uint32) uint32 { return a ^ b }
	case shaderir.LeftShift:
		return func(a, b uint32) uint32 { return a << (b & 31) }
	case shaderir.RightShift:
		if kind == shaderir.Int {
			return func(a, b uint32) uint32 { return uint32(int32(a) >> (b & 31)) }
		}
		return func(a, b uint32) uint32 { return a >> (b & 31) }
	case shaderir.AndAnd:
		return func(a, b uint32) uint32 { return a & b }
	case shaderir.OrOr:
		return func(a, b uint32) uint32 { return a | b }
	}
	panic(fmt.Sprintf("unexpected operator %d on %d", op, kind))
}

// compare returns a comparison of two lanes of kind, 1 for true.
func compare(op shaderir.Op, kind shaderir.BasicType) func(a, b uint32) uint32 {
	var less func(a, b uint32) bool
	eq := func(a, b uint32) bool { return a == b }
	switch kind {
	case shaderir.Float:
		less = func(a, b uint32) bool { return f32(a) < f32(b) }
		eq = func(a, b uint32) bool { return f32(a) == f32(b) }
	case shaderir.Int:
		less = func(a, b uint32) bool { return int32(a) < int32(b) }
	default:
		less = func(a, b uint32) bool { return a < b }
	}
	switch op {
	case shaderir.LessThanOp:
		return func(a, b uint32) uint32 { return boolBits(less(a, b)) }
	case shaderir.LessThanEqualOp:
		return func(a, b uint32) uint32 { return boolBits(less(a, b) || eq(a, b)) }
	case shaderir.GreaterThanOp:
		return func(a, b uint32) uint32 { return boolBits(less(b, a)) }
	case shaderir.GreaterThanEqualOp:
		return func(a, b uint32) uint32 { return boolBits(less(b, a) || eq(a, b)) }
	case shaderir.EqualOp, shaderir.VectorEqualOp:
		return func(a, b uint32) uint32 { return boolBits(eq(a, b)) }
	case shaderir.NotEqualOp, shaderir.VectorNotEqualOp:
		return func(a, b uint32) uint32 { return boolBits(!eq(a, b)) }
	}
	panic(fmt.Sprintf("unexpected comparison %d", op))
}

func (c *compiler) unary(fn *function, e *shaderir.Expr) operand {
	x := c.expr(fn, &e.Exprs[0])
	if e.Op == shaderir.Add {
		return x
	}
	o := c.alloc(x.t)
	n, kind := o.n, x.t.Scalar()
	var f func(uint32) uint32
	switch e.Op {
	case shaderir.Sub:
		switch kind {
		case shaderir.Float:
			f = func(a uint32) uint32 { return a ^ 0x80000000 }
		default:
			f = func(a uint32) uint32 { return -a }
		}
	case shaderir.NotOp:
		f = func(a uint32) uint32 { return a ^ 1 }
	case shaderir.ComplementOp:
		f = func(a uint32) uint32 { return ^a }
	default:
		panic(fmt.Sprintf("unexpected unary operator %d", e.Op))
	}
	return operand{off: o.off, t: x.t, run: func(s *state) {
		eval(s, x)
		for i := range n {
			s.r[o.off+i] = f(s.r[x.off+i])
		}
	}}
}

func (c *compiler) binary(fn *function, e *shaderir.Expr) operand {
	l, r := c.expr(fn, &e.Exprs[0]), c.expr(fn, &e.Exprs[1])
	t := shaderir.TypeOf(e, env{c, fn})
	o := c.alloc(t)
	ln, rn := l.n(), r.n()

	// A matrix times a vector or a matrix; times a scalar, it goes lane by
	// lane.
	if e.Op == shaderir.MatrixMul && (l.t.IsMatrix() || r.t.IsMatrix()) && ln > 1 && rn > 1 {
		return c.matrixMul(l, r, o)
	}

	switch e.Op {
	case shaderir.LessThanOp, shaderir.LessThanEqualOp, shaderir.GreaterThanOp, shaderir.GreaterThanEqualOp:
		cmp := compare(e.Op, l.t.Scalar())
		return operand{off: o.off, t: t, run: func(s *state) {
			eval(s, l, r)
			s.r[o.off] = cmp(s.r[l.off], s.r[r.off])
		}}
	case shaderir.EqualOp, shaderir.VectorEqualOp, shaderir.NotEqualOp, shaderir.VectorNotEqualOp:
		eq := compare(shaderir.EqualOp, l.t.Scalar())
		want := uint32(1)
		if e.Op == shaderir.NotEqualOp || e.Op == shaderir.VectorNotEqualOp {
			want = 0
		}
		n := ln
		return operand{off: o.off, t: t, run: func(s *state) {
			eval(s, l, r)
			all := uint32(1)
			for i := range n {
				all &= eq(s.r[l.off+i], s.r[r.off+i])
			}
			s.r[o.off] = boolBits(all == want)
		}}
	}

	// Lane by lane, a scalar going with every lane of the other operand.
	op := arith(e.Op, l.t.Scalar())
	n := o.n
	ls, rs := 1, 1
	if ln == 1 {
		ls = 0
	}
	if rn == 1 {
		rs = 0
	}
	return operand{off: o.off, t: t, run: func(s *state) {
		eval(s, l, r)
		for i := range n {
			s.r[o.off+i] = op(s.r[l.off+i*ls], s.r[r.off+i*rs])
		}
	}}
}

// matrixMul compiles the product of a matrix and a vector, a vector and a
// matrix, or two matrices, whose columns follow one another.
func (c *compiler) matrixMul(l, r operand, o slot) operand {
	switch {
	case l.t.IsMatrix() && r.t.IsMatrix():
		n := l.t.MatrixSize()
		return operand{off: o.off, t: o.t, run: func(s *state) {
			eval(s, l, r)
			for col := range n {
				for row := range n {
					var sum float32
					for k := range n {
						sum += f32(s.r[l.off+k*n+row]) * f32(s.r[r.off+col*n+k])
					}
					s.r[o.off+col*n+row] = bits(sum)
				}
			}
		}}
	case l.t.IsMatrix():
		n := l.t.MatrixSize()
		return operand{off: o.off, t: o.t, run: func(s *state) {
			eval(s, l, r)
			for row := range n {
				var sum float32
				for k := range n {
					sum += f32(s.r[l.off+k*n+row]) * f32(s.r[r.off+k])
				}
				s.r[o.off+row] = bits(sum)
			}
		}}
	default:
		n := r.t.MatrixSize()
		return operand{off: o.off, t: o.t, run: func(s *state) {
			eval(s, l, r)
			for col := range n {
				var sum float32
				for k := range n {
					sum += f32(s.r[l.off+k]) * f32(s.r[r.off+col*n+k])
				}
				s.r[o.off+col] = bits(sum)
			}
		}}
	}
}
