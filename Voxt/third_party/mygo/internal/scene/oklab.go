package scene

import "math"

// Oklab returns c, in sRGB, in the Oklab color space, as the renderers
// mix gradients in it: the CPU's from this, the GPU's from the stops'
// values it gives them, or from the same math in their shaders.
func Oklab(c Color) [3]float32 {
	r, g, b := toLinear(float32(c.R)/255), toLinear(float32(c.G)/255), toLinear(float32(c.B)/255)
	l := float32(math.Cbrt(float64(0.4122214708*r + 0.5363325363*g + 0.0514459929*b)))
	m := float32(math.Cbrt(float64(0.2119034982*r + 0.6806995451*g + 0.1073969566*b)))
	s := float32(math.Cbrt(float64(0.0883024619*r + 0.2817188376*g + 0.6299787005*b)))
	return [3]float32{
		0.2104542553*l + 0.7936177850*m - 0.0040720468*s,
		1.9779984951*l - 2.4285922050*m + 0.4505937099*s,
		0.0259040371*l + 0.7827717662*m - 0.8086757660*s,
	}
}

func toLinear(c float32) float32 {
	if c <= 0.04045 {
		return c / 12.92
	}
	return float32(math.Pow(float64((c+0.055)/1.055), 2.4))
}
