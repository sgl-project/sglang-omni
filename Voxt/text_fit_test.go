package main

import (
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"slices"
	"strings"
	"testing"

	"github.com/egoist/mygo/ui"
)

// Single-line text gets the width it measures and is laid out again at
// that width; on macOS (MyGo 0.4.8, Core Text) a few strings then broke
// early and ended in "…", as "M4B（含章节）" did in the audiobook's Format
// control: Core Text spaces a bracket apart from a letter of another script
// and, between runs of two fonts, the measured width left that room out
// (AudioSmith research/mygo-coretext-truncation.md). MyGo 0.5.0 counts
// that room (penAdvances); long lines still need third_party/mygo-patches
// 0001. textFits tells whether a text keeps to one line at its own width,
// the test of that.

// textFitSizes are the type scale's sizes and the weights views use.
var (
	textFitSizes   = []float32{fsMicro, fsCaption, fsBody, fsSection, fsTitle}
	textFitWeights = []int{400, 500, 600}
)

// misfits returns those of strs, paragraph by paragraph, that break into
// more lines when laid out at their own width, at any size and weight of
// the type scale, with where: in lang, the language the app shows, which
// MyGo lays text out in (showLanguage).
func misfits(t *testing.T, strs []string, lang string) map[string]string {
	t.Helper()
	out := map[string]string{}
	view := func(c *ui.Context) {
		ui.Text(c, "probe")
		for _, s := range strs {
			for _, para := range strings.Split(s, "\n") {
				if strings.TrimSpace(para) == "" {
					continue
				}
				for _, size := range textFitSizes {
					for _, w := range textFitWeights {
						span := ui.Span{Text: para, Size: size, Weight: w}
						w0, h0 := c.MeasureText(0, span)
						if w0 <= 0 {
							continue
						}
						if _, h1 := c.MeasureText(w0, span); h1 > h0+0.5 {
							if _, seen := out[s]; !seen {
								out[s] = fmt.Sprintf("%q at %gpt/%d", para, size, w)
							}
						}
					}
				}
			}
		}
	}
	tt := ui.NewTester(view, 400, 200)
	tt.SetLocale(lang)
	tt.Frame()
	return out
}

// TestTextFitsItsOwnWidth: the strings that showed "…" on the Mac, and
// elements built as views build them — a single-line text and a
// segmented control unconstrained in a row — keep to one line. On the
// Mac with MyGo 0.4.8 the first three failed, unless MyGo carried the fix.
func TestTextFitsItsOwnWidth(t *testing.T) {
	if os.Getenv("VOXTNEW_TEXTFIT_PROBE") == "" {
		t.Skip("probe of MyGo's text engine; set VOXTNEW_TEXTFIT_PROBE=1 (meaningful on macOS)")
	}
	probe := []string{
		"M4B（含章节）", "标题(可选)", "M4B（含章节）尾",
		"标题（可选）", "M4B (含章节)", "M4B · 含章节", "含章节的 M4B", "M4B (chapters)",
		"已完成，共三章", "模型：Kokoro", "“角色”里配音色", "《海上来信》", "【测试】", "「引号」",
	}
	for s, where := range misfits(t, probe, "zh-Hans") {
		t.Errorf("%q breaks at its own width (%s)", s, where)
	}
	// The same through elements: a text that may wrap, unconstrained in a
	// row, is as high as the same text on one line unless it breaks at its
	// own width.
	var one, twin []ui.Element
	hOne, hTwin := make([]float32, len(probe)), make([]float32, len(probe))
	view := func(c *ui.Context) {
		one, twin = one[:0], twin[:0]
		ui.Column(c).Gap(4).AlignItems(ui.Start).Children(func() {
			for i, s := range probe {
				ui.Row(c.Key(i)).AlignItems(ui.Start).Children(func() {
					one = append(one, ui.Text(c.Key("a"), s).SingleLine().Ellipsis("…"))
					twin = append(twin, ui.Text(c.Key("b"), s))
					sel := 0
					ui.Segmented(c.Key("s"), &sel, s, "MP3")
				})
			}
		})
		for i := range one { // the last frame's layout
			hOne[i], hTwin[i] = one[i].Bounds().H, twin[i].Bounds().H
		}
	}
	tt := ui.NewTester(view, 1200, 900)
	tt.SetLocale("zh-Hans")
	tt.Frame()
	tt.Frame()
	tt.Frame()
	for i, s := range probe {
		if h, want := hTwin[i], hOne[i]; h > want+0.5 || want == 0 {
			t.Errorf("%q unconstrained is %.1f high, a line %.1f: it wraps at its own width", s, h, want)
		}
	}
}

// TestTranslationsFit audits every string of the catalogues (the English
// source strings, the Chinese translations) for breaking at their own width, which
// single-line text shows as "…". It runs with VOXTNEW_TEXTFIT_AUDIT=1
// and writes the list to VOXTNEW_TEXTFIT_AUDIT_OUT if set; with
// VOXTNEW_TEXTFIT_AUDIT=strict it fails on any.
func TestTranslationsFit(t *testing.T) {
	mode := os.Getenv("VOXTNEW_TEXTFIT_AUDIT")
	if mode == "" {
		t.Skip("audit; set VOXTNEW_TEXTFIT_AUDIT=1 (meaningful on macOS)")
	}
	load := func(name string) map[string]string {
		b, err := os.ReadFile("internal/i18n/locales/" + name)
		if err != nil {
			t.Fatal(err)
		}
		m := map[string]string{}
		if err := json.Unmarshal(b, &m); err != nil {
			t.Fatal(err)
		}
		return m
	}
	var report strings.Builder
	fmt.Fprintf(&report, "# %s/%s\n", runtime.GOOS, runtime.GOARCH)
	total := 0
	zh, en := load("zh-Hans.json"), load("en.json")
	catalogues := map[string][]string{}
	for k, v := range zh {
		catalogues["zh-Hans"] = append(catalogues["zh-Hans"], v)
		catalogues["en"] = append(catalogues["en"], k) // the English source strings
	}
	for k, v := range en {
		catalogues["en"] = append(catalogues["en"], k, v)
	}
	for _, name := range []string{"zh-Hans", "en"} {
		strs := catalogues[name]
		slices.Sort(strs)
		strs = slices.Compact(strs)
		bad := misfits(t, strs, name)
		keys := make([]string, 0, len(bad))
		for s := range bad {
			keys = append(keys, s)
		}
		slices.Sort(keys)
		fmt.Fprintf(&report, "## %s: %d of %d strings break at their own width\n", name, len(keys), len(strs))
		for _, s := range keys {
			fmt.Fprintf(&report, "%s\t%s\n", bad[s], strings.ReplaceAll(s, "\n", `\n`))
		}
		total += len(keys)
	}
	t.Log(report.String())
	if out := os.Getenv("VOXTNEW_TEXTFIT_AUDIT_OUT"); out != "" {
		if err := os.WriteFile(out, []byte(report.String()), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if mode == "strict" && total > 0 {
		t.Errorf("%d strings break at their own width", total)
	}
}
