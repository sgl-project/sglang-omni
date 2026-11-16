package main

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeTestPNG(t *testing.T, directory, name string, width, height int, pixel color.NRGBA) {
	t.Helper()
	picture := image.NewNRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			picture.SetNRGBA(x, y, color.NRGBA{R: 200, G: 200, B: 200, A: 255})
		}
	}
	picture.SetNRGBA(1, 1, pixel)
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, picture); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, name), encoded.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestCompareMeasuredNoise(t *testing.T) {
	before, after := t.TempDir(), t.TempDir()
	base := color.NRGBA{R: 10, G: 20, B: 30, A: 255}
	for _, name := range []string{"same.png", "listed-1.png", "listed-2.png", "outside-1.png", "alpha.png", "size.png"} {
		writeTestPNG(t, before, name, 4, 4, base)
	}
	writeTestPNG(t, after, "same.png", 4, 4, base)
	writeTestPNG(t, after, "listed-1.png", 4, 4, color.NRGBA{R: 11, G: 21, B: 31, A: 255})
	writeTestPNG(t, after, "listed-2.png", 4, 4, color.NRGBA{R: 12, G: 20, B: 30, A: 255})
	writeTestPNG(t, after, "outside-1.png", 4, 4, color.NRGBA{R: 11, G: 20, B: 30, A: 255})
	writeTestPNG(t, after, "alpha.png", 4, 4, color.NRGBA{R: 10, G: 20, B: 30, A: 254})
	writeTestPNG(t, after, "size.png", 5, 4, base)

	noisyPath := filepath.Join(t.TempDir(), "noisy.list")
	noisyList := "# measured noise\nmain/listed-1.png\nmain/listed-2.png\nmain/alpha.png\nmain/size.png\n"
	if err := os.WriteFile(noisyPath, []byte(noisyList), 0o644); err != nil {
		t.Fatal(err)
	}
	noisy, err := readNoisy(noisyPath)
	if err != nil {
		t.Fatal(err)
	}
	result, err := compare(before, after, options{setName: "main", noisy: noisy})
	if err != nil {
		t.Fatal(err)
	}
	if result.files != 6 || result.identical != 1 || result.tolerated != 1 || len(result.failures) != 4 {
		t.Fatalf("unexpected comparison result: %+v", result)
	}
	failures := strings.Join(result.failures, "\n")
	for _, expected := range []string{
		"listed-2.png: 1 pixels differ (by up to 2)",
		"outside-1.png: 1 pixels differ outside the measured noisy set (by up to 1)",
		"alpha.png: alpha differs outside the measured noise rule",
		"size.png: size (4,4), other (5,4)",
	} {
		if !strings.Contains(failures, expected) {
			t.Errorf("missing failure %q in:\n%s", expected, failures)
		}
	}

	reportPath := filepath.Join(t.TempDir(), "observed.list")
	measured, err := compare(before, after, options{setName: "main", measure: true, reportPath: reportPath})
	if err != nil {
		t.Fatal(err)
	}
	if measured.identical != 1 || measured.tolerated != 2 || len(measured.failures) != 3 {
		t.Fatalf("measurement accepted an invalid difference: %+v", measured)
	}
	if err := writeReport(reportPath, measured.observed); err != nil {
		t.Fatal(err)
	}
	report, err := os.ReadFile(reportPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(report) != "main/listed-1.png\nmain/outside-1.png\n" {
		t.Fatalf("unexpected noise report:\n%s", report)
	}
}
