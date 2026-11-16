package gpu

import (
	"cmp"
	"image"
	"slices"

	"github.com/egoist/mygo/internal/raster"
	"github.com/egoist/mygo/internal/scene"
)

// clipMaskSlot is a retained clip mask. Masks outlive the frames that
// don't use them, up to idleMaskBytes of them, so a transform that comes
// and goes finds its masks again instead of rasterizing them.
type clipMaskSlot struct {
	shapes []scene.ClipShape
	area   image.Rectangle
	image  *scene.Image
	// frame is the last frame that used the mask, and slot the clip's
	// place among that frame's masked clips.
	frame uint64
	slot  int
}

// idleMaskBytes is how much memory the masks no clip of the frame uses
// keep, the least recently used dropped first.
const idleMaskBytes = 32 << 20

func (b *Builder) prepare(in *Instance, m scene.Affine, c *clip) bool {
	b.preparedMask = 0
	if c.scissor.Empty() || in.Rect[2] <= 0 || in.Rect[3] <= 0 {
		return false
	}
	if _, ok := m.Inverse(); !ok {
		return false
	}
	in.Transform0, in.Transform1, in.Transform2 = [4]float32{1, 0, 0, 0}, [4]float32{0, 1, 0, 0}, [4]float32{}
	if m.Set {
		in.Transform0 = [4]float32{m.A, m.C, m.X, 1}
		in.Transform1 = [4]float32{m.B, m.D, m.Y, 0}
		in.Transform2 = [4]float32{m.G, m.H, 0, 0}
	}
	in.Clip, in.ClipRadii = rect(c.rect), c.radii
	if !c.masked {
		return true
	}
	// A hard clip sharing the drawing's coordinates is cheap to evaluate in
	// the fragment shader. Keep the rounded ancestor's mask stationary even
	// when a text input or custom clipped drawing moves within it.
	if c.shape.Radii == [4]float32{} && c.shape.Transform == m && m.Set && len(b.stack) > 1 {
		parent := &b.stack[len(b.stack)-2]
		if parent.masked {
			b.preparedMask = b.clipTexture(parent, b.clipShapes[:len(b.clipShapes)-1])
			if b.preparedMask == 0 {
				return false
			}
			in.Clip = rect(c.rect)
			in.ClipRadii = [4]float32{float32(parent.scissor.Left), float32(parent.scissor.Top), float32(parent.scissor.Right - parent.scissor.Left), float32(parent.scissor.Bottom - parent.scissor.Top)}
			in.Transform1[3] = 2
			return true
		}
	}
	b.preparedMask = b.clipTexture(c, b.clipShapes)
	if b.preparedMask == 0 {
		return false
	}
	in.Clip = [4]float32{float32(c.scissor.Left), float32(c.scissor.Top), float32(c.scissor.Right - c.scissor.Left), float32(c.scissor.Bottom - c.scissor.Top)}
	in.ClipRadii = [4]float32{}
	in.Transform1[3] = 1
	return true
}

func (b *Builder) clipTexture(c *clip, shapes []scene.ClipShape) uintptr {
	if c.maskTexture != 0 {
		return c.maskTexture
	}
	area := image.Rect(int(c.scissor.Left), int(c.scissor.Top), int(c.scissor.Right), int(c.scissor.Bottom))
	// The mask of the same clip, in this frame or an earlier one; else
	// the one the clip in its place had last frame, which it most likely
	// is, changed; else a new one.
	found, reuse := -1, -1
	for i := range b.maskSlots {
		m := &b.maskSlots[i]
		if m.area == area && slices.Equal(m.shapes, shapes) {
			found = i
			break
		}
		if reuse < 0 && m.frame == b.frame-1 && m.slot == c.maskSlot {
			reuse = i
		}
	}
	if found < 0 {
		found = reuse
		if found < 0 {
			found = len(b.maskSlots)
			b.maskSlots = append(b.maskSlots, clipMaskSlot{})
		}
		m := &b.maskSlots[found]
		m.image = raster.ClipMask(m.image, shapes, area)
		m.area = area
		m.shapes = append(m.shapes[:0], shapes...)
	}
	m := &b.maskSlots[found]
	m.frame, m.slot = b.frame, c.maskSlot
	c.maskTexture = b.imageTexture(m.image)
	return c.maskTexture
}

// dropIdleMasks drops the least recently used masks this frame didn't use
// beyond idle bytes of them.
func (b *Builder) dropIdleMasks(idle int) {
	slices.SortFunc(b.maskSlots, func(x, y clipMaskSlot) int { return cmp.Compare(y.frame, x.frame) })
	for i := range b.maskSlots {
		m := &b.maskSlots[i]
		if m.frame == b.frame {
			continue
		}
		if idle -= len(m.image.Pix); idle < 0 {
			clear(b.maskSlots[i:])
			b.maskSlots = b.maskSlots[:i]
			return
		}
	}
}
