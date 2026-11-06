package interp

import (
	"fmt"
	"math"

	"github.com/egoist/mygo/internal/shaderir"
)

// convert returns a lane of kind from converts to kind to, as the
// shading languages' constructors convert: a float to an integer drops
// its fraction, an int and a uint keep their bits, and a bool is 0 or 1.
func convert(from, to shaderir.BasicType) func(uint32) uint32 {
	if from == to {
		return func(a uint32) uint32 { return a }
	}
	switch to {
	case shaderir.Float:
		switch from {
		case shaderir.Int:
			return func(a uint32) uint32 { return bits(float32(int32(a))) }
		case shaderir.Uint:
			return func(a uint32) uint32 { return bits(float32(a)) }
		case shaderir.Bool:
			return func(a uint32) uint32 { return bits(float32(a)) }
		}
	case shaderir.Int, shaderir.Uint:
		switch from {
		case shaderir.Float:
			return func(a uint32) uint32 {
				f := float64(f32(a))
				if math.IsNaN(f) {
					return 0
				}
				if to == shaderir.Uint && f >= 0 {
					return uint32(int64(min(f, math.MaxUint32)))
				}
				return uint32(int32(max(min(f, math.MaxInt32), math.MinInt32)))
			}
		default:
			return func(a uint32) uint32 { return a }
		}
	case shaderir.Bool:
		if from == shaderir.Float {
			return func(a uint32) uint32 { return boolBits(f32(a) != 0) }
		}
		return func(a uint32) uint32 { return boolBits(a != 0) }
	}
	panic(fmt.Sprintf("cannot convert %d to %d", from, to))
}

// float1 and float2 are functions of float lanes.
type (
	float1 func(x float64) float64
	float2 func(x, y float64) float64
)

var unaryFloat = map[shaderir.BuiltinFunc]float1{
	shaderir.Sin:  math.Sin,
	shaderir.Cos:  math.Cos,
	shaderir.Tan:  math.Tan,
	shaderir.Asin: math.Asin,
	shaderir.Acos: math.Acos,
	shaderir.Atan: math.Atan,
	shaderir.Exp:  math.Exp,
	shaderir.Log:  math.Log,
	shaderir.Exp2: math.Exp2,
	shaderir.Log2: math.Log2,
	shaderir.Sqrt: math.Sqrt,
	shaderir.Inversesqrt: func(x float64) float64 {
		return 1 / math.Sqrt(x)
	},
	shaderir.Floor:   math.Floor,
	shaderir.Ceil:    math.Ceil,
	shaderir.Fract:   func(x float64) float64 { return x - math.Floor(x) },
	shaderir.Radians: func(x float64) float64 { return x * math.Pi / 180 },
	shaderir.Degrees: func(x float64) float64 { return x * 180 / math.Pi },
	// The CPU shades a pixel at a time, without its neighbors: the
	// derivatives are zero.
	shaderir.Dfdx:   func(float64) float64 { return 0 },
	shaderir.Dfdy:   func(float64) float64 { return 0 },
	shaderir.Fwidth: func(float64) float64 { return 0 },
}

var binaryFloat = map[shaderir.BuiltinFunc]float2{
	shaderir.Atan2: math.Atan2,
	shaderir.Pow:   math.Pow,
	shaderir.Mod:   func(x, y float64) float64 { return x - y*math.Floor(x/y) },
	shaderir.Step: func(edge, x float64) float64 {
		if x < edge {
			return 0
		}
		return 1
	},
}

// minMax returns min or max of two lanes of kind.
func minMax(f shaderir.BuiltinFunc, kind shaderir.BasicType) func(a, b uint32) uint32 {
	less := compare(shaderir.LessThanOp, kind)
	if kind == shaderir.Float {
		// As GPUs' (IEEE 754's minNum and maxNum), a NaN gives way to the
		// other operand.
		return func(a, b uint32) uint32 {
			x, y := f32(a), f32(b)
			switch {
			case x != x:
				return b
			case y != y:
				return a
			case f == shaderir.Min && y < x, f == shaderir.Max && y > x:
				return b
			}
			return a
		}
	}
	if f == shaderir.Min {
		return func(a, b uint32) uint32 {
			if less(b, a) != 0 {
				return b
			}
			return a
		}
	}
	return func(a, b uint32) uint32 {
		if less(a, b) != 0 {
			return b
		}
		return a
	}
}

// builtin compiles a call of a built-in function.
func (c *compiler) builtin(fn *function, e *shaderir.Expr) operand {
	f := e.Exprs[0].BuiltinFunc
	t := shaderir.TypeOf(e, env{c, fn})
	var args []operand
	for i := 1; i < len(e.Exprs); i++ {
		args = append(args, c.expr(fn, &e.Exprs[i]))
	}

	switch f {
	case shaderir.BoolF, shaderir.IntF, shaderir.UintF, shaderir.FloatF,
		shaderir.Vec2F, shaderir.Vec3F, shaderir.Vec4F,
		shaderir.IVec2F, shaderir.IVec3F, shaderir.IVec4F,
		shaderir.UVec2F, shaderir.UVec3F, shaderir.UVec4F,
		shaderir.Mat2F, shaderir.Mat3F, shaderir.Mat4F:
		return c.construct(t, args)
	case shaderir.FrontFacing:
		return c.constBits(t, 1)
	case shaderir.TexelAt:
		return c.texelAt(t, args[0], args[1])
	case shaderir.Min, shaderir.Max:
		return c.fold(t, args, minMax(f, t.Scalar()))
	case shaderir.Abs, shaderir.Sign:
		return c.absSign(f, t, args[0])
	case shaderir.Clamp:
		mn, mx := minMax(shaderir.Max, t.Scalar()), minMax(shaderir.Min, t.Scalar())
		return c.lanes3(t, args, func(x, lo, hi uint32) uint32 { return mx(mn(x, lo), hi) })
	case shaderir.Mix:
		return c.lanes3(t, args, func(a, b, w uint32) uint32 {
			x, y, k := f32(a), f32(b), f32(w)
			return bits(x*(1-k) + y*k)
		})
	case shaderir.Smoothstep:
		return c.lanes3(t, args, func(e0, e1, x uint32) uint32 {
			// Clamped as GPUs clamp: a NaN, as from equal edges, gives 0.
			k := (f32(x) - f32(e0)) / (f32(e1) - f32(e0))
			if !(k > 0) {
				k = 0
			} else if k > 1 {
				k = 1
			}
			return bits(k * k * (3 - 2*k))
		})
	case shaderir.Length:
		return c.reduce(t, args, func(s *state, v []operand) float64 { return math.Sqrt(dot(s, v[0], v[0])) })
	case shaderir.Distance:
		return c.reduce(t, args, func(s *state, v []operand) float64 {
			var sum float64
			for i := range v[0].n() {
				d := float64(f32(s.r[v[0].off+i])) - float64(f32(s.r[v[1].off+i]))
				sum += d * d
			}
			return math.Sqrt(sum)
		})
	case shaderir.Dot:
		return c.reduce(t, args, func(s *state, v []operand) float64 { return dot(s, v[0], v[1]) })
	case shaderir.Cross:
		a, b := args[0], args[1]
		o := c.alloc(t)
		return operand{off: o.off, t: t, run: func(s *state) {
			eval(s, a, b)
			x := func(o operand, i int) float32 { return f32(s.r[o.off+i]) }
			s.r[o.off] = bits(x(a, 1)*x(b, 2) - x(a, 2)*x(b, 1))
			s.r[o.off+1] = bits(x(a, 2)*x(b, 0) - x(a, 0)*x(b, 2))
			s.r[o.off+2] = bits(x(a, 0)*x(b, 1) - x(a, 1)*x(b, 0))
		}}
	case shaderir.Normalize:
		a := args[0]
		o := c.alloc(t)
		n := o.n
		return operand{off: o.off, t: t, run: func(s *state) {
			eval(s, a)
			k := 1 / math.Sqrt(dot(s, a, a))
			for i := range n {
				s.r[o.off+i] = bits(float32(float64(f32(s.r[a.off+i])) * k))
			}
		}}
	case shaderir.Faceforward:
		nv, iv, ref := args[0], args[1], args[2]
		o := c.alloc(t)
		n := o.n
		return operand{off: o.off, t: t, run: func(s *state) {
			eval(s, nv, iv, ref)
			sign := uint32(0)
			if dot(s, ref, iv) >= 0 {
				sign = 0x80000000
			}
			for i := range n {
				s.r[o.off+i] = s.r[nv.off+i] ^ sign
			}
		}}
	case shaderir.Reflect:
		iv, nv := args[0], args[1]
		o := c.alloc(t)
		n := o.n
		return operand{off: o.off, t: t, run: func(s *state) {
			eval(s, iv, nv)
			d := dot(s, nv, iv)
			for i := range n {
				s.r[o.off+i] = bits(float32(float64(f32(s.r[iv.off+i])) - 2*d*float64(f32(s.r[nv.off+i]))))
			}
		}}
	case shaderir.Refract:
		iv, nv, eta := args[0], args[1], args[2]
		o := c.alloc(t)
		n := o.n
		return operand{off: o.off, t: t, run: func(s *state) {
			eval(s, iv, nv, eta)
			d := dot(s, nv, iv)
			k := float64(f32(s.r[eta.off]))
			q := 1 - k*k*(1-d*d)
			for i := range n {
				if q < 0 {
					s.r[o.off+i] = 0
					continue
				}
				v := k*float64(f32(s.r[iv.off+i])) - (k*d+math.Sqrt(q))*float64(f32(s.r[nv.off+i]))
				s.r[o.off+i] = bits(float32(v))
			}
		}}
	case shaderir.Transpose:
		a := args[0]
		o := c.alloc(t)
		n := t.MatrixSize()
		return operand{off: o.off, t: t, run: func(s *state) {
			eval(s, a)
			for col := range n {
				for row := range n {
					s.r[o.off+row*n+col] = s.r[a.off+col*n+row]
				}
			}
		}}
	}
	if g, ok := unaryFloat[f]; ok {
		a := args[0]
		o := c.alloc(t)
		n := o.n
		return operand{off: o.off, t: t, run: func(s *state) {
			eval(s, a)
			for i := range n {
				s.r[o.off+i] = bits(float32(g(float64(f32(s.r[a.off+i])))))
			}
		}}
	}
	if g, ok := binaryFloat[f]; ok {
		return c.lanes2(t, args, func(a, b uint32) uint32 {
			return bits(float32(g(float64(f32(a)), float64(f32(b)))))
		})
	}
	panic(fmt.Sprintf("built-in function %s is not supported", f))
}

func dot(s *state, a, b operand) float64 {
	var sum float64
	for i := range a.n() {
		sum += float64(f32(s.r[a.off+i])) * float64(f32(s.r[b.off+i]))
	}
	return sum
}

func (c *compiler) constBits(t shaderir.Type, v uint32) operand {
	o := c.alloc(t)
	return operand{off: o.off, t: t, run: func(s *state) { s.r[o.off] = v }}
}

// construct compiles a constructor: the arguments' lanes in order, each
// converted to the type's kind, a scalar alone filling a vector, and
// making the diagonal of a matrix.
func (c *compiler) construct(t shaderir.Type, args []operand) operand {
	o := c.alloc(t)
	n, kind := o.n, t.Scalar()
	if len(args) == 1 && args[0].n() == 1 {
		a := args[0]
		conv := convert(a.t.Scalar(), kind)
		if t.IsMatrix() {
			size := t.MatrixSize()
			return operand{off: o.off, t: t, run: func(s *state) {
				eval(s, a)
				clear(s.r[o.off : o.off+n])
				v := conv(s.r[a.off])
				for i := range size {
					s.r[o.off+i*size+i] = v
				}
			}}
		}
		return operand{off: o.off, t: t, run: func(s *state) {
			eval(s, a)
			v := conv(s.r[a.off])
			for i := range n {
				s.r[o.off+i] = v
			}
		}}
	}
	convs := make([]func(uint32) uint32, len(args))
	for i, a := range args {
		convs[i] = convert(a.t.Scalar(), kind)
	}
	return operand{off: o.off, t: t, run: func(s *state) {
		k := 0
		for i, a := range args {
			eval(s, a)
			for j := 0; j < a.n() && k < n; j++ {
				s.r[o.off+k] = convs[i](s.r[a.off+j])
				k++
			}
		}
	}}
}

// texelAt compiles the pixel of an image at a position, transparent
// outside it.
func (c *compiler) texelAt(t shaderir.Type, tex, pos operand) operand {
	o := c.alloc(t)
	return operand{off: o.off, t: t, run: func(s *state) {
		eval(s, tex, pos)
		clear(s.r[o.off : o.off+4])
		i := int(s.r[tex.off])
		if i >= len(s.in.Images) || s.in.Images[i] == nil {
			return
		}
		img := s.in.Images[i]
		w, h := img.Size()
		x, y := math.Floor(float64(f32(s.r[pos.off]))), math.Floor(float64(f32(s.r[pos.off+1])))
		if !(x >= 0 && y >= 0 && x < float64(w) && y < float64(h)) {
			return
		}
		px := img.At(int(x), int(y))
		for k := range px {
			s.r[o.off+k] = bits(px[k])
		}
	}}
}

// fold compiles min or max of any number of arguments, a scalar going
// with every lane of a vector.
func (c *compiler) fold(t shaderir.Type, args []operand, g func(a, b uint32) uint32) operand {
	o := c.alloc(t)
	n := o.n
	return operand{off: o.off, t: t, run: func(s *state) {
		eval(s, args...)
		for i := range n {
			v := s.r[args[0].off+i*stride(args[0])]
			for _, a := range args[1:] {
				v = g(v, s.r[a.off+i*stride(a)])
			}
			s.r[o.off+i] = v
		}
	}}
}

func stride(o operand) int {
	if o.n() == 1 {
		return 0
	}
	return 1
}

// lanes2 and lanes3 compile functions of two and three lanes, lane by
// lane, a scalar going with every lane.
func (c *compiler) lanes2(t shaderir.Type, args []operand, g func(a, b uint32) uint32) operand {
	o := c.alloc(t)
	n := o.n
	a, b := args[0], args[1]
	sa, sb := stride(a), stride(b)
	return operand{off: o.off, t: t, run: func(s *state) {
		eval(s, a, b)
		for i := range n {
			s.r[o.off+i] = g(s.r[a.off+i*sa], s.r[b.off+i*sb])
		}
	}}
}

func (c *compiler) lanes3(t shaderir.Type, args []operand, g func(a, b, x uint32) uint32) operand {
	o := c.alloc(t)
	n := o.n
	a, b, x := args[0], args[1], args[2]
	sa, sb, sx := stride(a), stride(b), stride(x)
	return operand{off: o.off, t: t, run: func(s *state) {
		eval(s, a, b, x)
		for i := range n {
			s.r[o.off+i] = g(s.r[a.off+i*sa], s.r[b.off+i*sb], s.r[x.off+i*sx])
		}
	}}
}

// reduce compiles a function of vectors that gives a float.
func (c *compiler) reduce(t shaderir.Type, args []operand, g func(s *state, v []operand) float64) operand {
	o := c.alloc(t)
	return operand{off: o.off, t: t, run: func(s *state) {
		eval(s, args...)
		s.r[o.off] = bits(float32(g(s, args)))
	}}
}

func (c *compiler) absSign(f shaderir.BuiltinFunc, t shaderir.Type, a operand) operand {
	o := c.alloc(t)
	n, kind := o.n, t.Scalar()
	var g func(uint32) uint32
	switch {
	case kind == shaderir.Float && f == shaderir.Abs:
		g = func(x uint32) uint32 { return x &^ 0x80000000 }
	case kind == shaderir.Float:
		g = func(x uint32) uint32 {
			v := f32(x)
			switch {
			case v > 0:
				return bits(1)
			case v < 0:
				return bits(-1)
			}
			return bits(0)
		}
	case f == shaderir.Abs:
		g = func(x uint32) uint32 {
			if int32(x) < 0 {
				return -x
			}
			return x
		}
	default:
		g = func(x uint32) uint32 {
			v := int32(x)
			return uint32(boolBits(v > 0)) - uint32(boolBits(v < 0))
		}
	}
	return operand{off: o.off, t: t, run: func(s *state) {
		eval(s, a)
		for i := range n {
			s.r[o.off+i] = g(s.r[a.off+i])
		}
	}}
}
