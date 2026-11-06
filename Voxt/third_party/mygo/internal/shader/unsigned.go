package shader

import (
	"fmt"
	gconstant "go/constant"
	"go/token"
	"math"

	"github.com/egoist/mygo/internal/shaderir"
)

// blockEnv is a block's types, as the compiler parses it.
type blockEnv struct {
	cs *compileState
	b  *block
}

func (e blockEnv) LocalType(index int) shaderir.Type {
	var chain []*block
	for b := e.b; b != nil; b = b.outer {
		chain = append(chain, b)
	}
	for i := len(chain) - 1; i >= 0; i-- {
		if index < len(chain[i].vars) {
			return chain[i].vars[index].typ
		}
		index -= len(chain[i].vars)
	}
	return shaderir.Type{}
}

func (e blockEnv) UniformType(index int) shaderir.Type { return e.cs.ir.Uniforms[index] }

func (e blockEnv) FuncReturn(index int) shaderir.Type { return e.cs.funcs[index].ir.Return }

// typeConsts marks the expressions of stmts, statements of block b in a
// function returning returnType, that are of type uint or uvecN
// (Expr.Unsigned), integer constants among them, which take the type of
// the operands they go with, or of the variable, the parameter or the
// result they go to. Then it checks that every integer constant fits its
// type, reporting an error at pos for those that do not. Nested blocks are
// not marked, as parseBlock marks each block it parses.
func (cs *compileState) typeConsts(pos token.Pos, b *block, stmts []shaderir.Stmt, returnType shaderir.Type) bool {
	env := blockEnv{cs, b}
	m := &unsignedMarker{env: env}
	for i := range stmts {
		s := &stmts[i]
		switch s.Type {
		case shaderir.ExprStmt:
			m.mark(&s.Exprs[0], shaderir.TypeOf(&s.Exprs[0], env).Scalar())
		case shaderir.Assign:
			t := shaderir.TypeOf(&s.Exprs[0], env).Scalar()
			m.mark(&s.Exprs[0], t)
			m.mark(&s.Exprs[1], t)
		case shaderir.Return:
			for j := range s.Exprs {
				m.mark(&s.Exprs[j], returnType.Scalar())
			}
		case shaderir.If:
			m.mark(&s.Exprs[0], shaderir.Bool)
		case shaderir.For:
			for j := range s.Exprs {
				m.mark(&s.Exprs[j], s.ForVarType.Scalar())
			}
		}
	}

	ok := true
	var check func(e *shaderir.Expr)
	check = func(e *shaderir.Expr) {
		if e.Type == shaderir.NumberExpr && e.Const.Kind() == gconstant.Int {
			if e.Unsigned {
				if x, exact := gconstant.Uint64Val(e.Const); !exact || x > math.MaxUint32 {
					cs.addError(pos, fmt.Sprintf("constant %s overflows uint", e.Const.String()))
					ok = false
				}
			} else if !cs.checkIntConstRange(pos, e.Const) {
				ok = false
			}
		}
		for i := range e.Exprs {
			check(&e.Exprs[i])
		}
	}
	for i := range stmts {
		s := &stmts[i]
		for j := range s.Exprs {
			check(&s.Exprs[j])
		}
		if s.Type == shaderir.For {
			// A uint counter starts and ends at uints, and steps up or down
			// by a uint.
			if s.ForVarType.Main == shaderir.Uint {
				delta := s.ForDelta
				if delta != nil && gconstant.Sign(delta) < 0 {
					delta = gconstant.UnaryOp(token.SUB, delta, 0)
				}
				for _, v := range []gconstant.Value{s.ForInit, s.ForEnd, delta} {
					if !cs.checkUintConstRange(pos, v) {
						ok = false
					}
				}
				continue
			}
			for _, v := range []gconstant.Value{s.ForInit, s.ForEnd, s.ForDelta} {
				if !cs.checkIntConstRange(pos, v) {
					ok = false
				}
			}
		}
	}
	return ok
}

// unsignedMarker marks the expressions of type uint or uvecN.
type unsignedMarker struct {
	env shaderir.TypeEnv
}

// mark marks e and its operands, e an expression whose integer constants
// have the type want.
func (m *unsignedMarker) mark(e *shaderir.Expr, want shaderir.BasicType) {
	switch e.Type {
	case shaderir.NumberExpr:
		e.Unsigned = e.Unsigned || want == shaderir.Uint && e.Const.Kind() == gconstant.Int
		return
	case shaderir.Unary:
		if e.Op == shaderir.NotOp {
			want = shaderir.Bool
		}
		m.mark(&e.Exprs[0], want)
	case shaderir.Binary:
		if e.Op == shaderir.LeftShift || e.Op == shaderir.RightShift {
			m.operands(e.Exprs[:1], want)
			m.operands(e.Exprs[1:], shaderir.Int)
		} else {
			m.operands(e.Exprs, want)
		}
	case shaderir.Selection:
		m.mark(&e.Exprs[0], shaderir.Bool)
		m.operands(e.Exprs[1:], want)
	case shaderir.Call:
		m.call(e)
	case shaderir.FieldSelector:
		m.own(&e.Exprs[0])
	case shaderir.Index:
		m.own(&e.Exprs[0])
		m.mark(&e.Exprs[1], shaderir.Int)
	}
	e.Unsigned = shaderir.TypeOf(e, m.env).Scalar() == shaderir.Uint
}

// own marks e as of its own type.
func (m *unsignedMarker) own(e *shaderir.Expr) {
	m.mark(e, shaderir.TypeOf(e, m.env).Scalar())
}

// operands marks es, operands of one type: constants and untyped shifts
// among them take the type of the others, or want when all are such.
func (m *unsignedMarker) operands(es []shaderir.Expr, want shaderir.BasicType) {
	for i := range es {
		if !flexible(&es[i]) {
			want = shaderir.TypeOf(&es[i], m.env).Scalar()
			break
		}
	}
	for i := range es {
		if flexible(&es[i]) {
			m.mark(&es[i], want)
		} else {
			m.own(&es[i])
		}
	}
}

// flexible reports whether e takes the type of its context: a constant not
// yet marked as a uint, or an untyped shift.
func flexible(e *shaderir.Expr) bool {
	return e.Type == shaderir.NumberExpr && !e.Unsigned || e.Untyped
}

func (m *unsignedMarker) call(e *shaderir.Expr) {
	callee, args := e.Exprs[0], e.Exprs[1:]
	if callee.Type == shaderir.FunctionExpr {
		f := m.env.(blockEnv).cs.funcs[callee.Index].ir
		for i := range args {
			if i < len(f.InParams) {
				m.mark(&args[i], f.InParams[i].Scalar())
			} else {
				m.own(&args[i])
			}
		}
		return
	}
	// A constructor's constants take its kind; other functions' arguments
	// share a type.
	var kind shaderir.BasicType
	switch callee.BuiltinFunc {
	case shaderir.UintF, shaderir.UVec2F, shaderir.UVec3F, shaderir.UVec4F:
		kind = shaderir.Uint
	case shaderir.IntF, shaderir.IVec2F, shaderir.IVec3F, shaderir.IVec4F:
		kind = shaderir.Int
	case shaderir.FloatF, shaderir.Vec2F, shaderir.Vec3F, shaderir.Vec4F, shaderir.Mat2F, shaderir.Mat3F, shaderir.Mat4F:
		kind = shaderir.Float
	case shaderir.BoolF:
		kind = shaderir.Bool
	default:
		m.operands(args, shaderir.Int)
		return
	}
	for i := range args {
		if flexible(&args[i]) {
			m.mark(&args[i], kind)
		} else {
			m.own(&args[i])
		}
	}
}
