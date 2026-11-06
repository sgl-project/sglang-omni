package glsl

import (
	"fmt"

	"github.com/egoist/mygo/internal/shaderir"
)

// UniformLanes returns, for each lane of p's uniforms as package interp
// keeps them (in order, columns of matrices, elements of arrays), its
// offset in bytes in the std140 uniform block of Fragment, and the
// block's size, a multiple of 16: a vector of 3 takes the room of 4, and
// an array's elements and a matrix's columns are 16 bytes apart at least.
func UniformLanes(p *shaderir.Program) (lanes []int, size int) {
	var walk func(t *shaderir.Type, base int)
	walk = func(t *shaderir.Type, base int) {
		switch {
		case t.Main == shaderir.Array:
			s, _ := std140(&t.Sub[0])
			stride := roundUp(s, 16)
			for i := range t.Length {
				walk(&t.Sub[0], base+i*stride)
			}
		case t.IsMatrix():
			n := t.MatrixSize()
			for col := range n {
				for row := range n {
					lanes = append(lanes, base+col*16+row*4)
				}
			}
		default:
			for i := range max(t.VectorElementCount(), 1) {
				lanes = append(lanes, base+4*i)
			}
		}
	}
	for i := range p.Uniforms {
		s, a := std140(&p.Uniforms[i])
		size = roundUp(size, a)
		walk(&p.Uniforms[i], size)
		size += s
	}
	return lanes, roundUp(size, 16)
}

// std140 returns the size and the alignment of t in a std140 block.
func std140(t *shaderir.Type) (size, align int) {
	switch t.Main {
	case shaderir.Bool, shaderir.Int, shaderir.Uint, shaderir.Float:
		return 4, 4
	case shaderir.Vec2, shaderir.IVec2, shaderir.UVec2:
		return 8, 8
	case shaderir.Vec3, shaderir.IVec3, shaderir.UVec3:
		return 12, 16
	case shaderir.Vec4, shaderir.IVec4, shaderir.UVec4:
		return 16, 16
	case shaderir.Mat2, shaderir.Mat3, shaderir.Mat4:
		return t.MatrixSize() * 16, 16
	case shaderir.Array:
		s, _ := std140(&t.Sub[0])
		return t.Length * roundUp(s, 16), 16
	}
	panic(fmt.Sprintf("glsl: no std140 layout for %s", t))
}

func roundUp(x, a int) int { return (x + a - 1) / a * a }
