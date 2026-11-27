// Package shader compiles shaders written in Go's syntax into the
// intermediate representation of package shaderir, which its backends
// translate for each GPU renderer.
//
// It is Ebitengine's Kage compiler (github.com/hajimehoshi/ebiten,
// internal/shader and internal/shaderir at commit 16f18b0), under the
// Apache License 2.0 in LICENSE, forked to compile MyGo's shaders: files
// that came from Ebitengine keep its copyright notice, and say so where
// MyGo changed them.
package shader
