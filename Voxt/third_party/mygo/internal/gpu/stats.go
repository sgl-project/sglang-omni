package gpu

import "fmt"

// Summary describes the last scene built, for logs (MYGO_FRAME_STATS): how
// many instances in how many batches, how many ops were left out as covered,
// how many fills drew as opaque cores, and the batches of each class.
func (b *Builder) Summary() string {
	var classes [NumClasses]int
	opaque := 0
	for i := range b.Batches {
		classes[b.Batches[i].Class]++
		if b.Batches[i].Opaque {
			opaque++
		}
	}
	return fmt.Sprintf("%d instances in %d batches (%d opaque; by class general %d, plain %d, fill %d, shadow %d, glyph %d, subpixel %d, image %d), %d ops culled, %d opaque cores",
		len(b.Instances), len(b.Batches), opaque, classes[ClassGeneral], classes[ClassPlain], classes[ClassFill], classes[ClassShadow], classes[ClassGlyph],
		classes[ClassSubpixel], classes[ClassImage], b.Culled, b.OpaqueCores)
}
