// Copyright 2020 The Ebiten Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Modified by the MyGo contributors: uint and uvecN.

package shaderir

import (
	"fmt"
	"strings"
)

type Type struct {
	Main   BasicType
	Sub    []Type
	Length int
}

func (t Type) Equal(rhs *Type) bool {
	if t.Main != rhs.Main {
		return false
	}
	if t.Length != rhs.Length {
		return false
	}
	if len(t.Sub) != len(rhs.Sub) {
		return false
	}
	for i, s := range t.Sub {
		if !s.Equal(&rhs.Sub[i]) {
			return false
		}
	}
	return true
}

func (t Type) String() string {
	switch t.Main {
	case None:
		return "none"
	case Bool:
		return "bool"
	case Int:
		return "int"
	case Float:
		return "float"
	case Vec2:
		return "vec2"
	case Vec3:
		return "vec3"
	case Vec4:
		return "vec4"
	case IVec2:
		return "ivec2"
	case IVec3:
		return "ivec3"
	case IVec4:
		return "ivec4"
	case Mat2:
		return "mat2"
	case Mat3:
		return "mat3"
	case Mat4:
		return "mat4"
	case Uint:
		return "uint"
	case UVec2:
		return "uvec2"
	case UVec3:
		return "uvec3"
	case UVec4:
		return "uvec4"
	case Texture:
		return "texture"
	case Array:
		return fmt.Sprintf("[%d]%s", t.Length, t.Sub[0].String())
	case Struct:
		str := "struct{"
		sub := make([]string, 0, len(t.Sub))
		for _, st := range t.Sub {
			sub = append(sub, st.String())
		}
		str += strings.Join(sub, ",")
		str += "}"
		return str
	default:
		return fmt.Sprintf("?(unknown type: %d)", t.Main)
	}
}

func (t Type) DwordCount() int {
	switch t.Main {
	case Bool:
		// The size of a bool varies by the shader language, but use 1 for simplicity.
		return 1
	case Int:
		return 1
	case Float:
		return 1
	case Vec2:
		return 2
	case Vec3:
		return 3
	case Vec4:
		return 4
	case IVec2:
		return 2
	case IVec3:
		return 3
	case IVec4:
		return 4
	case Mat2:
		return 4
	case Mat3:
		return 9
	case Mat4:
		return 16
	case Uint:
		return 1
	case UVec2:
		return 2
	case UVec3:
		return 3
	case UVec4:
		return 4
	case Array:
		return t.Length * t.Sub[0].DwordCount()
	default: // TODO: Parse a struct correctly
		return -1
	}
}

func (t Type) IsFloatVector() bool {
	switch t.Main {
	case Vec2, Vec3, Vec4:
		return true
	}
	return false
}

func (t Type) IsIntVector() bool {
	switch t.Main {
	case IVec2, IVec3, IVec4:
		return true
	}
	return false
}

// IsUintVector reports whether t is uvec2, uvec3 or uvec4.
func (t Type) IsUintVector() bool {
	switch t.Main {
	case UVec2, UVec3, UVec4:
		return true
	}
	return false
}

// IsIntegerVector reports whether t is a vector of int or uint.
func (t Type) IsIntegerVector() bool { return t.IsIntVector() || t.IsUintVector() }

// IsInteger reports whether t is int, uint, or a vector of either.
func (t Type) IsInteger() bool {
	return t.Main == Int || t.Main == Uint || t.IsIntegerVector()
}

// Scalar returns the type of t's components: float for a float vector or
// a matrix, int or uint for an integer vector, and t's own for a scalar.
func (t Type) Scalar() BasicType {
	switch t.Main {
	case Vec2, Vec3, Vec4, Mat2, Mat3, Mat4:
		return Float
	case IVec2, IVec3, IVec4:
		return Int
	case UVec2, UVec3, UVec4:
		return Uint
	}
	return t.Main
}

// VectorOf returns the vector of n components of type scalar, or scalar
// itself for n 1.
func VectorOf(scalar BasicType, n int) BasicType {
	if n == 1 {
		return scalar
	}
	var v [3]BasicType
	switch scalar {
	case Float:
		v = [3]BasicType{Vec2, Vec3, Vec4}
	case Int:
		v = [3]BasicType{IVec2, IVec3, IVec4}
	case Uint:
		v = [3]BasicType{UVec2, UVec3, UVec4}
	case Bool:
		return None
	}
	if n < 2 || n > 4 {
		return None
	}
	return v[n-2]
}

func (t Type) VectorElementCount() int {
	switch t.Main {
	case UVec2:
		return 2
	case UVec3:
		return 3
	case UVec4:
		return 4
	case Vec2:
		return 2
	case Vec3:
		return 3
	case Vec4:
		return 4
	case IVec2:
		return 2
	case IVec3:
		return 3
	case IVec4:
		return 4
	default:
		return -1
	}
}

func (t Type) IsMatrix() bool {
	switch t.Main {
	case Mat2, Mat3, Mat4:
		return true
	}
	return false
}

func (t Type) MatrixSize() int {
	switch t.Main {
	case Mat2:
		return 2
	case Mat3:
		return 3
	case Mat4:
		return 4
	default:
		return -1
	}
}

type BasicType int

const (
	None BasicType = iota
	Bool
	Int
	Float
	Vec2
	Vec3
	Vec4
	IVec2
	IVec3
	IVec4
	Mat2
	Mat3
	Mat4
	Texture
	Array
	Struct

	// Added by MyGo, after Ebitengine's types.
	Uint
	UVec2
	UVec3
	UVec4
)

func descendantLocalVars(block, target *Block) ([]Type, bool) {
	if block == target {
		return block.LocalVars, true
	}

	var ts []Type
	for _, s := range block.Stmts {
		for _, b := range s.Blocks {
			if ts2, found := descendantLocalVars(b, target); found {
				n := b.LocalVarIndexOffset - block.LocalVarIndexOffset
				ts = append(ts, block.LocalVars[:n]...)
				ts = append(ts, ts2...)
				return ts, true
			}
		}
	}
	return nil, false
}

func localVariableType(p *Program, topBlock, block *Block, absidx int) Type {
	// TODO: Rename this function (truly-local variable?)
	var ts []Type
	for _, f := range p.Funcs {
		if f.Block == topBlock {
			ts = append(f.InParams, f.OutParams...)
			break
		}
	}

	ts2, _ := descendantLocalVars(topBlock, block)
	ts = append(ts, ts2...)
	return ts[absidx]
}

func (p *Program) LocalVariableType(topBlock, block *Block, idx int) Type {
	switch topBlock {
	case p.VertexFunc.Block:
		na := len(p.Attributes)
		nv := len(p.Varyings)
		switch {
		case idx < na:
			return p.Attributes[idx]
		case idx == na:
			return Type{Main: Vec4}
		case idx < na+nv+1:
			return p.Varyings[idx-na-1]
		default:
			return localVariableType(p, topBlock, block, idx-(na+nv+1))
		}
	case p.FragmentFunc.Block:
		nv := len(p.Varyings)
		switch {
		case idx == 0:
			return Type{Main: Vec4}
		case idx < nv+1:
			return p.Varyings[idx-1]
		default:
			return localVariableType(p, topBlock, block, idx-(nv+1))
		}
	default:
		return localVariableType(p, topBlock, block, idx)
	}
}
