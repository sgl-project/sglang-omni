// Command w0snapcmp compares snapshot directories using W0's measured noise rule.
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
	"strings"
)

type options struct {
	setName    string
	reportPath string
	out        string
	measure    bool
	noisy      map[string]bool
}

type result struct {
	files     int
	identical int
	tolerated int
	observed  []string
	failures  []string
}

func main() {
	var opts options
	flag.StringVar(&opts.setName, "set", "", "snapshot set name used as the prefix in the noisy list (required)")
	flag.StringVar(&opts.reportPath, "report", "", "write every qualifying noisy SET/FILE path to this file")
	flag.StringVar(&opts.out, "out", "", "write a before | after | diff picture for each failing PNG")
	measure := flag.Bool("measure", false, "accept and report qualifying noise without requiring a noisy-list entry")
	noisyPath := flag.String("noisy", "", "file of measured noisy SET/FILE paths")
	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: w0snapcmp -set NAME [-noisy FILE] [-report FILE] [-out DIR] [-measure] BEFORE AFTER")
		flag.PrintDefaults()
	}
	flag.Parse()
	opts.measure = *measure
	if len(flag.Args()) != 2 || opts.setName == "" || path.Clean(opts.setName) != opts.setName || strings.Contains(opts.setName, "/") {
		flag.Usage()
		os.Exit(2)
	}
	if *noisyPath != "" {
		names, err := readNoisy(*noisyPath)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		opts.noisy = names
	}
	if opts.out != "" {
		if err := os.MkdirAll(opts.out, 0o755); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
	}
	result, err := compare(flag.Arg(0), flag.Arg(1), opts)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if opts.reportPath != "" {
		if err := writeReport(opts.reportPath, result.observed); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
	}
	fmt.Printf("%d files: %d identical, %d within measured noise, %d failed\n", result.files, result.identical, result.tolerated, len(result.failures))
	for _, failure := range result.failures {
		fmt.Println("FAIL", failure)
	}
	if len(result.failures) != 0 {
		os.Exit(1)
	}
}

func readNoisy(fileName string) (map[string]bool, error) {
	file, err := os.Open(fileName)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	names := map[string]bool{}
	scanner := bufio.NewScanner(file)
	for lineNumber := 1; scanner.Scan(); lineNumber++ {
		line, _, _ := strings.Cut(scanner.Text(), "#")
		fields := strings.Fields(line)
		if len(fields) == 0 {
			continue
		}
		name := fields[0]
		if len(fields) != 1 || path.Clean(name) != name || path.IsAbs(name) || !strings.Contains(name, "/") || path.Ext(name) != ".png" {
			return nil, fmt.Errorf("%s:%d: want one relative SET/FILE.png path", fileName, lineNumber)
		}
		names[name] = true
	}
	return names, scanner.Err()
}

func writeReport(fileName string, observed []string) error {
	slices.Sort(observed)
	observed = slices.Compact(observed)
	content := ""
	if len(observed) != 0 {
		content = strings.Join(observed, "\n") + "\n"
	}
	return os.WriteFile(fileName, []byte(content), 0o644)
}

func compare(before, after string, opts options) (result, error) {
	var result result
	names := map[string]int{}
	for index, directory := range []string{before, after} {
		entries, err := os.ReadDir(directory)
		if err != nil {
			return result, err
		}
		for _, entry := range entries {
			if !entry.IsDir() {
				names[entry.Name()] |= 1 << index
			}
		}
	}
	orderedNames := make([]string, 0, len(names))
	for name := range names {
		orderedNames = append(orderedNames, name)
	}
	// An empty set compares nothing: the test that renders it was renamed
	// or removed, and go test passes when no test matches.
	if len(orderedNames) == 0 {
		return result, fmt.Errorf("set %s has no snapshots in %s or %s", opts.setName, before, after)
	}
	slices.Sort(orderedNames)
	result.files = len(orderedNames)
	for _, name := range orderedNames {
		if names[name] != 3 {
			missing := before
			if names[name]&1 != 0 {
				missing = after
			}
			result.failures = append(result.failures, fmt.Sprintf("%s: missing in %s", name, missing))
			continue
		}
		beforeBytes, err := os.ReadFile(filepath.Join(before, name))
		if err != nil {
			return result, err
		}
		afterBytes, err := os.ReadFile(filepath.Join(after, name))
		if err != nil {
			return result, err
		}
		if bytes.Equal(beforeBytes, afterBytes) {
			result.identical++
			continue
		}
		qualifiedName := path.Join(opts.setName, name)
		if filepath.Ext(name) != ".png" {
			result.failures = append(result.failures, name+": differs")
			continue
		}
		tolerated, observed, failure, err := comparePNG(name, qualifiedName, beforeBytes, afterBytes, opts)
		if err != nil {
			return result, err
		}
		if observed {
			result.observed = append(result.observed, qualifiedName)
		}
		if tolerated {
			result.tolerated++
		} else {
			result.failures = append(result.failures, failure)
		}
	}
	return result, nil
}

func comparePNG(name, qualifiedName string, beforeBytes, afterBytes []byte, opts options) (bool, bool, string, error) {
	before, err := png.Decode(bytes.NewReader(beforeBytes))
	if err != nil {
		return false, false, "", err
	}
	after, err := png.Decode(bytes.NewReader(afterBytes))
	if err != nil {
		return false, false, "", err
	}
	if before.Bounds().Size() != after.Bounds().Size() {
		return false, false, fmt.Sprintf("%s: size %v, other %v", name, before.Bounds().Size(), after.Bounds().Size()), nil
	}
	bounds := before.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	differentPixels := make([]bool, width*height)
	count, most := 0, 0
	box := image.Rectangle{}
	alphaDiffers := false
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			beforeColor := color.NRGBAModel.Convert(before.At(bounds.Min.X+x, bounds.Min.Y+y)).(color.NRGBA)
			afterColor := color.NRGBAModel.Convert(after.At(bounds.Min.X+x, bounds.Min.Y+y)).(color.NRGBA)
			if beforeColor == afterColor {
				continue
			}
			differentPixels[y*width+x] = true
			count++
			alphaDiffers = alphaDiffers || beforeColor.A != afterColor.A
			for _, channels := range [][2]uint8{{beforeColor.R, afterColor.R}, {beforeColor.G, afterColor.G}, {beforeColor.B, afterColor.B}} {
				most = max(most, abs(int(channels[0])-int(channels[1])))
			}
			point := image.Rect(x, y, x+1, y+1)
			if box.Empty() {
				box = point
			} else {
				box = box.Union(point)
			}
		}
	}
	qualifies := !alphaDiffers && most <= 1
	listed := opts.noisy[qualifiedName]
	if qualifies && (opts.measure || listed) {
		return true, true, "", nil
	}
	if opts.out != "" && count > 0 {
		if err := writeDiff(filepath.Join(opts.out, name), before, after, differentPixels, box); err != nil {
			return false, false, "", err
		}
	}
	switch {
	case count == 0:
		return false, false, name + ": PNG encoding differs outside the measured noise rule", nil
	case alphaDiffers:
		return false, false, fmt.Sprintf("%s: alpha differs outside the measured noise rule, box %v", name, box), nil
	case most > 1:
		return false, false, fmt.Sprintf("%s: %d pixels differ (by up to %d), box %v", name, count, most, box), nil
	default:
		return false, true, fmt.Sprintf("%s: %d pixels differ outside the measured noisy set (by up to %d), box %v", name, count, most, box), nil
	}
}

func abs(number int) int {
	return max(number, -number)
}

func writeDiff(fileName string, before, after image.Image, differentPixels []bool, box image.Rectangle) error {
	bounds := before.Bounds()
	width := bounds.Dx()
	box = box.Inset(-24).Intersect(image.Rect(0, 0, bounds.Dx(), bounds.Dy()))
	boxWidth, boxHeight := box.Dx(), box.Dy()
	const gap = 8
	picture := image.NewRGBA(image.Rect(0, 0, 3*boxWidth+2*gap, boxHeight))
	draw.Draw(picture, picture.Bounds(), image.NewUniform(color.RGBA{255, 0, 255, 255}), image.Point{}, draw.Src)
	at := box.Min.Add(bounds.Min)
	draw.Draw(picture, image.Rect(0, 0, boxWidth, boxHeight), before, at, draw.Src)
	draw.Draw(picture, image.Rect(boxWidth+gap, 0, 2*boxWidth+gap, boxHeight), after, at, draw.Src)
	diffX := 2 * (boxWidth + gap)
	for y := 0; y < boxHeight; y++ {
		for x := 0; x < boxWidth; x++ {
			gray := color.GrayModel.Convert(after.At(at.X+x, at.Y+y)).(color.Gray)
			value := gray.Y/3 + 170
			pixelColor := color.RGBA{value, value, value, 255}
			if differentPixels[(box.Min.Y+y)*width+box.Min.X+x] {
				pixelColor = color.RGBA{230, 0, 0, 255}
			}
			picture.Set(diffX+x, y, pixelColor)
		}
	}
	file, err := os.Create(fileName)
	if err != nil {
		return err
	}
	return errors.Join(png.Encode(file, picture), file.Close())
}
