// Command snapcmp compares two directories of UI snapshots (the PNGs and
// texts the snapshot tests write), for refactors that must not change a pixel.
// scripts/snapshots.sh renders the directories (a set each) and runs it.
//
//	go run ./internal/tools/snapcmp [-volatile FILE] [-out DIR] BEFORE AFTER
//	go run ./internal/tools/snapcmp [-volatile FILE] [-out DIR] [-grow 3] BASE_A BASE_B AFTER
//
// With two directories every file must exist in both and be equal: byte
// for byte, or (PNG) pixel for pixel. With three, BASE_A and BASE_B are two
// runs of one commit: the pixels where they differ (run-to-run noise),
// grown by -grow pixels, form a mask, and AFTER passes when it equals A or
// B, or differs from A only inside the mask. Snapshots are reproducible
// since refactor R0, so the two-directory form is the normal one.
//
// -volatile names a file of snapshots allowed to differ a little, a line
// each: a file name pattern (path.Match), the largest difference allowed in
// any color channel (0–255), and why. Pixels within it count as equal in
// the files the pattern matches. Blank lines and # comments are ignored.
//
// -out writes, for each PNG that fails, a picture of the differing region:
// before | after | the differing pixels in red.
//
// Exit status: 0 all equal, 1 differences (listed with the number and the
// bounding box of the differing pixels), 2 usage or I/O errors.
package main

import (
	"bufio"
	"bytes"
	"errors"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
)

func main() {
	var o options
	flag.IntVar(&o.grow, "grow", 3, "three-directory form: grow the noise mask by this many pixels")
	flag.StringVar(&o.out, "out", "", "write a before | after | diff picture of each failing PNG to this directory")
	volatile := flag.String("volatile", "", "file of snapshots allowed to differ a little (pattern, largest channel difference, reason)")
	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: snapcmp [-volatile FILE] [-out DIR] BEFORE AFTER\n       snapcmp [-volatile FILE] [-out DIR] [-grow n] BASE_A BASE_B AFTER")
		flag.PrintDefaults()
	}
	flag.Parse()
	dirs := flag.Args()
	if len(dirs) != 2 && len(dirs) != 3 {
		flag.Usage()
		os.Exit(2)
	}
	if *volatile != "" {
		v, err := readVolatile(*volatile)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		o.volatile = v
	}
	if o.out != "" {
		if err := os.MkdirAll(o.out, 0o755); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
	}
	r, err := compare(dirs, o)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	fmt.Println(r.summary())
	for _, f := range r.fails {
		fmt.Println("FAIL", f)
	}
	if len(r.fails) > 0 {
		os.Exit(1)
	}
}

type options struct {
	grow     int
	out      string
	volatile []volatile
}

// volatile is a line of the -volatile file.
type volatile struct {
	pattern string
	level   int // the largest channel difference allowed, 0–255
}

// tolerance is the channel difference allowed in file name.
func (o options) tolerance(name string) int {
	for _, v := range o.volatile {
		if ok, _ := path.Match(v.pattern, name); ok {
			return v.level
		}
	}
	return 0
}

func readVolatile(file string) ([]volatile, error) {
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var vs []volatile
	sc := bufio.NewScanner(f)
	for n := 1; sc.Scan(); n++ {
		line, _, _ := strings.Cut(sc.Text(), "#")
		fields := strings.Fields(line)
		if len(fields) == 0 {
			continue
		}
		bad := func(why string) error { return fmt.Errorf("%s:%d: %s", file, n, why) }
		if len(fields) < 3 {
			return nil, bad("want: pattern level reason")
		}
		if _, err := path.Match(fields[0], ""); err != nil {
			return nil, bad(err.Error())
		}
		level, err := strconv.Atoi(fields[1])
		if err != nil || level < 0 || level > 255 {
			return nil, bad("the level is a number from 0 to 255")
		}
		vs = append(vs, volatile{fields[0], level})
	}
	return vs, sc.Err()
}

type result struct {
	files, identical, masked, tolerated int
	fails                               []string
}

func (r result) summary() string {
	return fmt.Sprintf("%d files: %d identical, %d within run-to-run noise, %d within the volatile list's tolerance, %d failed",
		r.files, r.identical, r.masked, r.tolerated, len(r.fails))
}

// compare compares the last directory with the first (and, with three,
// the second as another baseline).
func compare(dirs []string, o options) (result, error) {
	var r result
	names := map[string]int{}
	for i, d := range dirs {
		entries, err := os.ReadDir(d)
		if err != nil {
			return r, err
		}
		for _, e := range entries {
			if !e.IsDir() {
				names[e.Name()] |= 1 << i
			}
		}
	}
	all := 1<<len(dirs) - 1
	var list []string
	for n := range names {
		list = append(list, n)
	}
	slices.Sort(list)
	r.files = len(list)
	for _, n := range list {
		if names[n] != all {
			var where []string
			for i, d := range dirs {
				if names[n]&(1<<i) == 0 {
					where = append(where, d)
				}
			}
			r.fails = append(r.fails, fmt.Sprintf("%s: missing in %s", n, strings.Join(where, ", ")))
			continue
		}
		data := make([][]byte, len(dirs))
		for i, d := range dirs {
			b, err := os.ReadFile(filepath.Join(d, n))
			if err != nil {
				return r, err
			}
			data[i] = b
		}
		after := data[len(dirs)-1]
		if bytes.Equal(after, data[0]) || len(dirs) == 3 && bytes.Equal(after, data[1]) {
			r.identical++
			continue
		}
		if filepath.Ext(n) != ".png" {
			r.fails = append(r.fails, n+": differs")
			continue
		}
		st, msg, err := comparePNG(n, dirs, data, o)
		if err != nil {
			return r, err
		}
		switch st {
		case same:
			r.identical++
		case masked:
			r.masked++
		case tolerated:
			r.tolerated++
		default:
			r.fails = append(r.fails, msg)
		}
	}
	return r, nil
}

type status int

const (
	same status = iota
	masked
	tolerated
	failed
)

// comparePNG compares the last of the images with the first, masking the
// noise between the first two when there are three.
func comparePNG(name string, dirs []string, data [][]byte, o options) (status, string, error) {
	imgs := make([]image.Image, len(dirs))
	for i := range dirs {
		img, err := png.Decode(bytes.NewReader(data[i]))
		if err != nil {
			return failed, "", fmt.Errorf("%s: %w", filepath.Join(dirs[i], name), err)
		}
		imgs[i] = img
	}
	a, c := imgs[0], imgs[len(imgs)-1]
	for _, img := range imgs[1:] {
		if img.Bounds() != a.Bounds() {
			return failed, fmt.Sprintf("%s: size %v, other %v", name, a.Bounds().Size(), img.Bounds().Size()), nil
		}
	}
	var noise []bool
	if len(imgs) == 3 {
		noise = mask(a, imgs[1], o.grow)
	}
	diff, count, box, most := differing(a, c, noise, 0)
	switch {
	case count == 0 && noise == nil:
		return same, "", nil // the same pixels, encoded otherwise
	case count == 0:
		return masked, "", nil
	}
	if tol := o.tolerance(name); tol > 0 && most <= tol {
		return tolerated, "", nil
	}
	if o.out != "" {
		if err := writeDiff(filepath.Join(o.out, name), a, c, diff, box); err != nil {
			return failed, "", err
		}
	}
	what := "pixels differ"
	if noise != nil {
		what = "pixels differ outside the noise mask"
	}
	return failed, fmt.Sprintf("%s: %d %s (by up to %d), box %v", name, count, what, most, box), nil
}

// level is how much two pixels differ: the largest difference of their
// channels, 0–255.
func level(x, y image.Image, px, py int) int {
	r1, g1, b1, a1 := x.At(px, py).RGBA()
	r2, g2, b2, a2 := y.At(px, py).RGBA()
	d := 0
	for _, p := range [4][2]uint32{{r1, r2}, {g1, g2}, {b1, b2}, {a1, a2}} {
		d = max(d, abs(int(p[0]>>8)-int(p[1]>>8)), min(1, abs(int(p[0])-int(p[1]))))
	}
	return d
}

func abs(n int) int { return max(n, -n) }

// mask marks the pixels where two baseline runs differ, grown by grow.
func mask(a, b image.Image, grow int) []bool {
	r := a.Bounds()
	w, h := r.Dx(), r.Dy()
	m := make([]bool, w*h)
	for y := range h {
		for x := range w {
			if level(a, b, r.Min.X+x, r.Min.Y+y) == 0 {
				continue
			}
			for dy := -grow; dy <= grow; dy++ {
				for dx := -grow; dx <= grow; dx++ {
					if xx, yy := x+dx, y+dy; xx >= 0 && yy >= 0 && xx < w && yy < h {
						m[yy*w+xx] = true
					}
				}
			}
		}
	}
	return m
}

// differing returns the pixels where a and c differ by more than tol
// outside the mask (nil: none), their number, their bounding box (from 0)
// and the largest difference.
func differing(a, c image.Image, noise []bool, tol int) ([]bool, int, image.Rectangle, int) {
	r := a.Bounds()
	w, h := r.Dx(), r.Dy()
	diff := make([]bool, w*h)
	count, box, most := 0, image.Rectangle{}, 0
	for y := range h {
		for x := range w {
			if noise != nil && noise[y*w+x] {
				continue
			}
			d := level(a, c, r.Min.X+x, r.Min.Y+y)
			if d <= tol {
				continue
			}
			diff[y*w+x] = true
			count++
			most = max(most, d)
			if pt := image.Rect(x, y, x+1, y+1); box.Empty() {
				box = pt
			} else {
				box = box.Union(pt)
			}
		}
	}
	return diff, count, box, most
}

// writeDiff writes before | after | diff of box (padded) side by side.
func writeDiff(file string, a, c image.Image, diff []bool, box image.Rectangle) error {
	r := a.Bounds()
	w := r.Dx()
	box = box.Inset(-24).Intersect(image.Rect(0, 0, r.Dx(), r.Dy()))
	bw, bh := box.Dx(), box.Dy()
	const gap = 8
	pic := image.NewRGBA(image.Rect(0, 0, 3*bw+2*gap, bh))
	draw.Draw(pic, pic.Bounds(), image.NewUniform(color.RGBA{255, 0, 255, 255}), image.Point{}, draw.Src)
	at := box.Min.Add(r.Min)
	draw.Draw(pic, image.Rect(0, 0, bw, bh), a, at, draw.Src)
	draw.Draw(pic, image.Rect(bw+gap, 0, 2*bw+gap, bh), c, at, draw.Src)
	x0 := 2 * (bw + gap)
	for y := range bh {
		for x := range bw {
			g := color.GrayModel.Convert(c.At(at.X+x, at.Y+y)).(color.Gray)
			v := g.Y/3 + 170 // the after picture, faded
			col := color.RGBA{v, v, v, 255}
			if diff[(box.Min.Y+y)*w+box.Min.X+x] {
				col = color.RGBA{230, 0, 0, 255}
			}
			pic.Set(x0+x, y, col)
		}
	}
	f, err := os.Create(file)
	if err != nil {
		return err
	}
	return errors.Join(png.Encode(f, pic), f.Close())
}
