# glass

Liquid Glass for [MyGo](https://github.com/egoist/mygo) apps of native UI,
as macOS 26 and later draw it, on every platform: a material that what is
under an element shows through, frosted and bent along its edges.

```go
import "github.com/egoist/mygo/plugins/glass"

ui.Row(c).Padding(8, 16).Radius(22).Material(glass.Glass{}).Children(func() {
	ui.Text(c, "On glass")
})
```

`glass.ScrollEdge` is macOS's scroll edge effect, which content scrolling
under a bar fades (soft) or frosts (hard) under, and `glass.Blur` blurs
what is under an element without the glass, evenly or, with a mask,
progressively:

```go
ui.Box(c).Absolute().Top(0).Left(0).Right(0).Height(74).PassThrough().Material(glass.ScrollEdge{})
ui.Box(c).Absolute().Top(0).Left(0).Right(0).Height(88).PassThrough().
	Material(glass.Blur{Radius: 6, Mask: &ui.LinearGradient{From: ui.RGB(0, 0, 0), To: ui.Transparent, Angle: 180}})
```

See [the documentation](../../docs/plugins/glass.md), and the Glass page of
`examples/gallery`.

The glass and the blur, which a hard scroll edge draws with, are effects
of MyGo's renderers (`scene.Effect`): their shaders in Metal Shading
Language, HLSL and GLSL (`glass.metal`, `glass.hlsl`, `glass.glsl`, and
`blur.*`), and their twins for the CPU renderer (`pixels.go`, `blur.go`),
which must draw the same pixels; the tests compare each GPU's drawing
with the CPU's.
`go generate` compiles the shaders ahead of time where it can: the Metal
library on macOS, with Xcode (`shaders_darwin.go`), and the Direct3D
bytecode on Windows (`shaders_windows.go`). Renderers compile the source
when the code compiled ahead of time is older than it or than their own
part of the shader.
