package helpertest

import (
	"bytes"
	"hash/fnv"
	"image"
	"image/color"
	"image/png"
	"math"
	"strings"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
)

// Icon is the fake's icon of an app: a rounded square of a colour of its
// bundle id, with a lighter disc, as a size × size PNG.
func Icon(bundleID string, size int) []byte {
	if size <= 0 {
		size = proto.IconSize
	}
	h := fnv.New32a()
	h.Write([]byte(strings.ToLower(bundleID)))
	hue := float64(h.Sum32()%360) / 360
	base := hsv(hue, 0.55, 0.85)
	light := hsv(hue, 0.25, 1)
	img := image.NewNRGBA(image.Rect(0, 0, size, size))
	s := float64(size)
	inset, r := s*0.08, s*0.2 // the square's margin and corner radius, as macOS's
	for y := range size {
		for x := range size {
			px, py := float64(x)+0.5, float64(y)+0.5
			// Distance outside the rounded square (≤ 0 inside).
			dx := math.Max(math.Abs(px-s/2)-(s/2-inset-r), 0)
			dy := math.Max(math.Abs(py-s/2)-(s/2-inset-r), 0)
			out := math.Hypot(dx, dy) - r
			a := math.Min(math.Max(0.5-out, 0), 1)
			if a == 0 {
				continue
			}
			c := base
			if math.Hypot(px-s/2, py-s/2) < s*0.22 {
				c = light
			}
			c.A = uint8(a * 255)
			img.SetNRGBA(x, y, c)
		}
	}
	var buf bytes.Buffer
	png.Encode(&buf, img)
	return buf.Bytes()
}

// hsv is an opaque colour of hue, saturation and value in [0, 1].
func hsv(h, s, v float64) color.NRGBA {
	i := math.Floor(h * 6)
	f := h*6 - i
	p, q, t := v*(1-s), v*(1-f*s), v*(1-(1-f)*s)
	var r, g, b float64
	switch int(i) % 6 {
	case 0:
		r, g, b = v, t, p
	case 1:
		r, g, b = q, v, p
	case 2:
		r, g, b = p, v, t
	case 3:
		r, g, b = p, q, v
	case 4:
		r, g, b = t, p, v
	default:
		r, g, b = v, p, q
	}
	return color.NRGBA{uint8(r * 255), uint8(g * 255), uint8(b * 255), 255}
}

// known reports whether the fake has an app of bundleID: one of Apps,
// Installed or Browsers.
func known(bundleID string) bool {
	for _, a := range Apps {
		if strings.EqualFold(a.BundleID, bundleID) {
			return true
		}
	}
	for _, a := range Installed {
		if strings.EqualFold(a.BundleID, bundleID) {
			return true
		}
	}
	for id := range Browsers {
		if strings.EqualFold(id, bundleID) {
			return true
		}
	}
	return false
}
