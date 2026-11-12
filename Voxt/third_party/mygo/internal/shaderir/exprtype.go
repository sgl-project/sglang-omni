package shaderir

import "go/constant"

// TypeEnv tells the types of what expressions refer to: local variables,
// by their index in their function, uniform variables, and the results of
// functions.
type TypeEnv interface {
	LocalType(index int) Type
	UniformType(index int) Type
	FuncReturn(index int) Type
}

// TypeOf returns the type of e, an expression the compiler checked. A
// constant has the type of its kind, bool, float or int, or uint when it
// is Unsigned; a call of a function with output parameters, which only
// statements make, has none.
func TypeOf(e *Expr, env TypeEnv) Type {
	switch e.Type {
	case NumberExpr:
		switch e.Const.Kind() {
		case constant.Bool:
			return Type{Main: Bool}
		case constant.Float:
			return Type{Main: Float}
		case constant.Int:
			if e.Unsigned {
				return Type{Main: Uint}
			}
			return Type{Main: Int}
		}
	case UniformVariable:
		return env.UniformType(e.Index)
	case TextureVariable:
		return Type{Main: Texture}
	case LocalVariable:
		return env.LocalType(e.Index)
	case Unary:
		if e.Op == NotOp {
			return Type{Main: Bool}
		}
		return TypeOf(&e.Exprs[0], env)
	case Binary:
		return binaryType(e, env)
	case Selection:
		return operandType(e.Exprs[1:], env)
	case Call:
		return callType(e, env)
	case FieldSelector:
		base := TypeOf(&e.Exprs[0], env)
		switch m := e.Exprs[1]; m.Type {
		case SwizzlingExpr:
			return Type{Main: VectorOf(base.Scalar(), len(m.Swizzling))}
		case StructMember:
			return base.Sub[m.Index]
		}
	case Index:
		base := TypeOf(&e.Exprs[0], env)
		switch {
		case base.Main == Array:
			return base.Sub[0]
		case base.IsMatrix():
			return Type{Main: VectorOf(Float, base.MatrixSize())}
		default:
			return Type{Main: base.Scalar()}
		}
	}
	return Type{}
}

// operandType returns the type of the first of es that is not a
// constant, or of the first when all are: the type that the operands of
// an operation or a built-in function share.
func operandType(es []Expr, env TypeEnv) Type {
	for i := range es {
		if es[i].Type != NumberExpr {
			return TypeOf(&es[i], env)
		}
	}
	if len(es) == 0 {
		return Type{}
	}
	return TypeOf(&es[0], env)
}

func binaryType(e *Expr, env TypeEnv) Type {
	l, r := TypeOf(&e.Exprs[0], env), TypeOf(&e.Exprs[1], env)
	switch e.Op {
	case LessThanOp, LessThanEqualOp, GreaterThanOp, GreaterThanEqualOp, EqualOp, NotEqualOp,
		VectorEqualOp, VectorNotEqualOp, AndAnd, OrOr:
		return Type{Main: Bool}
	case LeftShift, RightShift:
		return l
	case MatrixMul:
		switch {
		case l.IsMatrix() && r.IsMatrix():
			return l
		case l.IsMatrix() && r.IsFloatVector():
			return r
		case l.IsFloatVector() && r.IsMatrix():
			return l
		case r.IsMatrix():
			return r
		}
		return l
	}
	// A vector or a matrix with a scalar has the vector's or the matrix's
	// type; a constant takes the other operand's.
	lscalar := l.Main == Float || l.Main == Int || l.Main == Uint || l.Main == Bool
	rscalar := r.Main == Float || r.Main == Int || r.Main == Uint || r.Main == Bool
	switch {
	case lscalar && !rscalar:
		return r
	case rscalar && !lscalar:
		return l
	case e.Exprs[0].Type == NumberExpr && e.Exprs[1].Type != NumberExpr:
		return r
	}
	return l
}

func callType(e *Expr, env TypeEnv) Type {
	callee, args := e.Exprs[0], e.Exprs[1:]
	if callee.Type == FunctionExpr {
		return env.FuncReturn(callee.Index)
	}
	switch callee.BuiltinFunc {
	case BoolF, FrontFacing:
		return Type{Main: Bool}
	case IntF, Len, Cap:
		return Type{Main: Int}
	case UintF:
		return Type{Main: Uint}
	case FloatF, Length, Distance, Dot:
		return Type{Main: Float}
	case Vec2F:
		return Type{Main: Vec2}
	case Vec3F, Cross:
		return Type{Main: Vec3}
	case Vec4F, TexelAt:
		return Type{Main: Vec4}
	case IVec2F:
		return Type{Main: IVec2}
	case IVec3F:
		return Type{Main: IVec3}
	case IVec4F:
		return Type{Main: IVec4}
	case UVec2F:
		return Type{Main: UVec2}
	case UVec3F:
		return Type{Main: UVec3}
	case UVec4F:
		return Type{Main: UVec4}
	case Mat2F:
		return Type{Main: Mat2}
	case Mat3F:
		return Type{Main: Mat3}
	case Mat4F:
		return Type{Main: Mat4}
	case Step:
		return TypeOf(&args[1], env)
	case Smoothstep:
		return TypeOf(&args[2], env)
	case Min, Max:
		// A vector comes first when vectors and scalars mix.
		return TypeOf(&args[0], env)
	}
	return operandType(args, env)
}
