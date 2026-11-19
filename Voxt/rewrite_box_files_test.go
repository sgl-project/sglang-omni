package main

import (
	"bytes"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/transfer"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/imgconv"
)

func stubBoxDialog(t *testing.T, paths []string) chan mygo.OpenDialogOptions {
	t.Helper()
	old := boxDialogOpen
	options := make(chan mygo.OpenDialogOptions, 1)
	boxDialogOpen = func(opts mygo.OpenDialogOptions) ([]string, error) {
		options <- opts
		return paths, nil
	}
	t.Cleanup(func() { boxDialogOpen = old })
	return options
}

func boxTestPNG(t *testing.T, side int) []byte {
	t.Helper()
	var out bytes.Buffer
	if err := png.Encode(&out, image.NewRGBA(image.Rect(0, 0, side, side))); err != nil {
		t.Fatal(err)
	}
	return out.Bytes()
}

func TestRewriteBoxAddMenuKeyboardAndVoiceOver(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true, focusText: true}
	tt := hudTester(a.boxView, boxWinW, boxWinH)
	frames(tt)
	tt.Key(0, ui.KeyTab)
	tt.Key(0, ui.KeyEnter)
	frames(tt)
	if !a.box.ui.addOpen {
		t.Fatal("Tab and Enter did not open Add")
	}
	for _, label := range []string{"Choose images…", "Choose file…", "Paste clipboard"} {
		if !slices.Contains(tt.AccessLabels(), label) {
			t.Errorf("menu has no accessible %q control: %q", label, tt.AccessLabels())
		}
	}
	tt.Key(0, ui.KeyEscape)
	frames(tt)
	if a.box.ui.addOpen || !a.box.open {
		t.Fatalf("Escape closed the box instead of only its menu: addOpen %v, open %v", a.box.ui.addOpen, a.box.open)
	}
}

func TestRewriteBoxAddMenuPastesClipboardText(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true}
	tt := hudTester(a.boxView, boxWinW, boxWinH)
	text := "first line\nsecond line\nthird line\nfourth line"
	tt.SetClipboardData(transfer.TextData(text))
	click(t, tt, "Add")
	if err := tt.ClickControl("Paste clipboard"); err != nil {
		t.Fatal(err)
	}
	frames(tt)
	if got := a.boxTextContent(); got != text {
		t.Fatalf("clipboard text %q, want %q", got, text)
	}
	if a.box.ui.addOpen || len(a.box.textSegments) != 1 || !a.box.textSegments[0].Pasted {
		t.Fatalf("clipboard did not land as one closed pasted block: %+v", a.box)
	}
}

func TestBoxChooseImagesLandsLikeDrop(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true}
	drain := queuePosts(a)
	var files []string
	var want [][]byte
	for _, side := range []int{8, 16, 24, 32, 40} {
		encoded := boxJPEGOf(t, side, side)
		path := filepath.Join(t.TempDir(), "image-"+string(rune('a'+side/8-1))+".jpg")
		if err := os.WriteFile(path, encoded, 0o600); err != nil {
			t.Fatal(err)
		}
		files = append(files, path)
		if len(want) < boxMaxImages {
			want = append(want, encoded)
		}
	}
	options := stubBoxDialog(t, files)
	a.boxChooseImages()
	waitFor(t, drain, func() bool { return a.boxPending == 0 && len(a.box.images) == boxMaxImages })
	for index, expected := range want {
		if !bytes.Equal(a.box.images[index], expected) {
			t.Errorf("image %d does not follow selection order", index)
		}
	}
	if !a.box.limit {
		t.Fatal("the fifth selected image did not report the existing cap")
	}
	opts := <-options
	if !opts.Multiple || opts.Directory {
		t.Fatalf("image picker options %+v", opts)
	}
	if got, expected := opts.Filters[0].Extensions, []string{"png", "jpg", "jpeg", "gif"}; !slices.Equal(got, expected) {
		t.Fatalf("image filters %v, want %v", got, expected)
	}
}

func TestBoxChooseTextFileBecomesPastedText(t *testing.T) {
	for _, test := range []struct {
		name     string
		filename string
		text     string
		want     string
	}{
		{name: "long text", filename: "notes.txt", text: "First 中🙂\r\nSecond\r\nThird\r\nFourth", want: "First 中🙂\nSecond\nThird\nFourth"},
		{name: "short markdown", filename: "notes.md", text: "**short**", want: "**short**"},
	} {
		t.Run(test.name, func(t *testing.T) {
			a, _, _ := newTestApp(t)
			a.box = rewriteBox{open: true, shown: true}
			drain := queuePosts(a)
			path := filepath.Join(t.TempDir(), test.filename)
			if err := os.WriteFile(path, []byte(test.text), 0o600); err != nil {
				t.Fatal(err)
			}
			options := stubBoxDialog(t, []string{path})
			a.boxChooseFile()
			waitFor(t, drain, func() bool { return a.boxPending == 0 && a.boxTextContent() == test.want })
			if len(a.box.textSegments) != 1 {
				t.Fatalf("%d text segments, want one file block", len(a.box.textSegments))
			}
			segment := a.box.textSegments[0]
			if !segment.File || !segment.Pasted || segment.Source != test.filename || segment.Text != test.want {
				t.Fatalf("file segment %+v", segment)
			}
			opts := <-options
			if opts.Multiple || opts.Directory {
				t.Fatalf("document picker options %+v", opts)
			}
			if got, expected := opts.Filters[0].Extensions, []string{"txt", "md", "pdf"}; !slices.Equal(got, expected) {
				t.Fatalf("document filters %v, want %v", got, expected)
			}
		})
	}
}

func TestBoxChooseTextPDFBecomesPastedText(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true}
	drain := queuePosts(a)
	path := filepath.Join("testdata", "rewrite-box", "text.pdf")
	old := boxReadPDF
	boxReadPDF = func(gotPath string, maxPages int) (string, [][]byte, int, error) {
		if gotPath != path || maxPages != boxMaxImages {
			t.Fatalf("ReadPDF(%q, %d)", gotPath, maxPages)
		}
		return "First line\nSecond line\nThird line\nFourth line", nil, 2, nil
	}
	t.Cleanup(func() { boxReadPDF = old })
	stubBoxDialog(t, []string{path})
	a.boxChooseFile()
	want := "First line\nSecond line\nThird line\nFourth line"
	waitFor(t, drain, func() bool { return a.boxPending == 0 && a.boxTextContent() == want })
	if len(a.box.images) != 0 || len(a.box.textSegments) != 1 || !a.box.textSegments[0].File {
		t.Fatalf("text PDF landed as images or not as a file block: %+v", a.box)
	}
}

func TestBoxChooseScannedPDFUsesPagesWithinCap(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true}
	existing := boxJPEGOf(t, 8, 8)
	a.boxAttach(existing, "existing.png")
	drain := queuePosts(a)
	path := filepath.Join("testdata", "rewrite-box", "scanned.pdf")
	var pages [][]byte
	var want [][]byte
	for _, side := range []int{16, 24, 32, 40} {
		page := boxTestPNG(t, side)
		pages = append(pages, page)
		if len(want) < boxMaxImages-1 {
			encoded, err := imgconv.JPEG(page)
			if err != nil {
				t.Fatal(err)
			}
			want = append(want, encoded)
		}
	}
	old := boxReadPDF
	boxReadPDF = func(gotPath string, maxPages int) (string, [][]byte, int, error) {
		if gotPath != path || maxPages != boxMaxImages {
			t.Fatalf("ReadPDF(%q, %d)", gotPath, maxPages)
		}
		return "", pages, 6, nil
	}
	t.Cleanup(func() { boxReadPDF = old })
	stubBoxDialog(t, []string{path})
	a.boxChooseFile()
	waitFor(t, drain, func() bool { return a.boxPending == 0 && len(a.box.images) == boxMaxImages })
	if !bytes.Equal(a.box.images[0], existing) {
		t.Fatal("the PDF displaced the existing image")
	}
	for index, expected := range want {
		if !bytes.Equal(a.box.images[index+1], expected) {
			t.Errorf("PDF page %d landed out of order", index+1)
		}
	}
	if got, expected := a.box.err, "Skipped 3 PDF pages beyond the 4-image limit"; got != expected {
		t.Fatalf("skip message %q, want %q", got, expected)
	}
}

func TestRewriteBoxAddMenuPastesShortClipboardText(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true}
	tt := hudTester(a.boxView, boxWinW, boxWinH)
	tt.SetClipboardData(transfer.TextData("short"))
	click(t, tt, "Add")
	if err := tt.ClickControl("Paste clipboard"); err != nil {
		t.Fatal(err)
	}
	frames(tt)
	if got := a.boxTextContent(); got != "short" {
		t.Fatalf("clipboard text %q, want short", got)
	}
}

func TestRewriteBoxAddMenuPastesAtTheCaret(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true, focusText: true}
	tt := hudTester(a.boxView, boxWinW, boxWinH)
	frames(tt)
	tt.Type("hello world")
	frames(tt)
	for range 6 {
		tt.Key(0, ui.KeyLeft)
	}
	frames(tt)
	long := "one\ntwo\nthree\nfour"
	tt.SetClipboardData(transfer.TextData(long))
	click(t, tt, "Add")
	if err := tt.ClickControl("Paste clipboard"); err != nil {
		t.Fatal(err)
	}
	frames(tt)
	if got, want := a.boxTextContent(), "hello"+long+" world"; got != want {
		t.Fatalf("clipboard text %q, want %q", got, want)
	}
	tt.Type("X")
	frames(tt)
	if got, want := a.boxTextContent(), "hello"+long+"X world"; got != want {
		t.Fatalf("text after typing past the collapsed block %q, want %q", got, want)
	}
}

func TestRewriteBoxAddMenuReplacesSelectedText(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true, focusText: true}
	tt := hudTester(a.boxView, boxWinW, boxWinH)
	frames(tt)
	tt.Type("abc")
	tt.Key(ui.Cmd, ui.KeyA)
	frames(tt)
	tt.SetClipboardData(transfer.TextData("short"))
	click(t, tt, "Add")
	if err := tt.ClickControl("Paste clipboard"); err != nil {
		t.Fatal(err)
	}
	frames(tt)
	if got := a.boxTextContent(); got != "short" {
		t.Fatalf("clipboard replacement %q, want short", got)
	}
}

func TestBoxPasteClipboardTextAtEarlierSegment(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true, text: "before "}
	first := "one\ntwo\nthree\nfour"
	if !a.boxPasteText(first) {
		t.Fatal("the fixture paste did not collapse")
	}
	second := "five\nsix\nseven\neight"
	a.boxPasteClipboardTextAt(second, 0, 0, 0)
	if got, want := a.boxTextContent(), second+"before "+first; got != want {
		t.Fatalf("earlier-segment paste %q, want %q", got, want)
	}
	if a.box.ui.pasteCaret != 0 {
		t.Fatalf("caret after a collapsed earlier-segment paste is %d, want 0 in the suffix", a.box.ui.pasteCaret)
	}
}

func TestRewriteBoxAddMenuPasteFollowsEditorAfterBlockRemoval(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true, text: "before "}
	first := "one\ntwo\nthree\nfour"
	second := "five\nsix\nseven\neight"
	if !a.boxPasteText(first) || !a.boxPasteTextAt(0, second, 0, 0) {
		t.Fatal("the fixture pastes did not collapse")
	}
	a.box.ui.editorID = a.box.textSegments[2].ID
	a.box.ui.editorStart, a.box.ui.editorEnd = 0, 0
	a.box.ui.editorActive = true
	a.boxRemoveText(1)
	index, start, end := boxClipboardEditor(a.boxViewState(), &a.box.ui)
	a.boxPasteClipboardTextAt("X", index, start, end)
	if got, want := a.boxTextContent(), "Xbefore "+first; got != want {
		t.Fatalf("paste after removing an earlier block %q, want %q", got, want)
	}
}

func TestRewriteBoxAddMenuActivatesItemWithKeyboard(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true, focusText: true}
	tt := hudTester(a.boxView, boxWinW, boxWinH)
	tt.SetClipboardData(transfer.TextData("short"))
	frames(tt)
	tt.Key(0, ui.KeyTab)
	tt.Key(0, ui.KeyEnter)
	for range 3 {
		tt.Key(0, ui.KeyTab)
	}
	tt.Key(0, ui.KeyEnter)
	frames(tt)
	if got := a.boxTextContent(); got != "short" {
		t.Fatalf("keyboard clipboard paste %q, want short", got)
	}
}

func TestBoxChooseTextFileKeepsExistingTextOrder(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true, text: "before\n"}
	drain := queuePosts(a)
	path := filepath.Join(t.TempDir(), "notes.txt")
	if err := os.WriteFile(path, []byte("file body"), 0o600); err != nil {
		t.Fatal(err)
	}
	stubBoxDialog(t, []string{path})
	a.boxChooseFile()
	waitFor(t, drain, func() bool { return a.boxPending == 0 && a.boxTextContent() == "before\nfile body" })
	if len(a.box.textSegments) != 2 || a.box.textSegments[0].Pasted || !a.box.textSegments[1].File {
		t.Fatalf("file and typed prefix are out of order: %+v", a.box.textSegments)
	}
}

func TestRewriteBoxAddStaysReachableAfterFileBlocks(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true, limit: true, err: "This file is too large: huge.pdf"}
	for range 3 {
		a.boxAddFileText("first\nsecond\nthird\nfourth", "notes.txt")
	}
	tt := hudTester(a.boxView, boxWinW, boxWinH)
	frames(tt)
	for _, text := range []string{"At most 4 images", a.box.err} {
		bounds, ok := tt.Find(text)
		if !ok || bounds.Y < 0 || bounds.Y+bounds.H > boxWinH {
			t.Fatalf("%q is clipped: bounds %+v, present %v", text, bounds, ok)
		}
	}
	if err := tt.ClickControl("Add"); err != nil {
		t.Fatalf("Add is not reachable after three file blocks: %v (controls %q)", err, tt.AccessLabels())
	}
	frames(tt)
	if !a.box.ui.addOpen {
		t.Fatal("Add did not open after three file blocks")
	}
}

func TestBoxReadInputFileRejectsOversizeBeforeReading(t *testing.T) {
	for _, extension := range []string{".txt", ".pdf"} {
		t.Run(extension, func(t *testing.T) {
			a, _, _ := newTestApp(t)
			a.box = rewriteBox{open: true, shown: true}
			drain := queuePosts(a)
			path := filepath.Join(t.TempDir(), "huge"+extension)
			if err := os.WriteFile(path, nil, 0o600); err != nil {
				t.Fatal(err)
			}
			if err := os.Truncate(path, boxMaxInputFileBytes+1); err != nil {
				t.Fatal(err)
			}
			called := false
			old := boxReadPDF
			boxReadPDF = func(string, int) (string, [][]byte, int, error) {
				called = true
				return "", nil, 0, nil
			}
			t.Cleanup(func() { boxReadPDF = old })
			a.boxReadInputFile(path)
			waitFor(t, drain, func() bool { return a.boxPending == 0 && a.box.err != "" })
			if called {
				t.Fatal("an oversize file reached PDF extraction")
			}
			if want := "This file is too large: huge" + extension; a.box.err != want {
				t.Fatalf("error %q, want %q", a.box.err, want)
			}
			if len(a.box.images) != 0 || len(a.box.textSegments) != 0 {
				t.Fatalf("oversize input landed: images %d, segments %d", len(a.box.images), len(a.box.textSegments))
			}
		})
	}
}

func TestBoxTextContentKeepsFilesOnTheirOwnLines(t *testing.T) {
	for _, test := range []struct {
		name  string
		typed string
		files []string
		after string
		want  string
	}{
		{name: "two files without newline", files: []string{"first", "second"}, want: "first\nsecond"},
		{name: "file already ends with newline", files: []string{"first\n", "second\n"}, want: "first\nsecond\n"},
		{name: "typed text around a file", typed: "before", files: []string{"body"}, after: "after", want: "before\nbody\nafter"},
	} {
		t.Run(test.name, func(t *testing.T) {
			a, _, _ := newTestApp(t)
			a.box = rewriteBox{open: true, shown: true, text: test.typed}
			for _, text := range test.files {
				a.boxAddFileText(text, "notes.txt")
			}
			a.box.text = test.after
			if got := a.boxTextContent(); got != test.want {
				t.Fatalf("content %q, want %q", got, test.want)
			}
		})
	}
}

func TestBoxReadInputFileClearsAnEarlierFailure(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true}
	drain := queuePosts(a)
	dir := t.TempDir()
	bad, good := filepath.Join(dir, "notes.docx"), filepath.Join(dir, "notes.txt")
	for _, path := range []string{bad, good} {
		if err := os.WriteFile(path, []byte("body"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	a.boxReadInputFile(bad)
	waitFor(t, drain, func() bool { return a.boxPending == 0 && a.box.err != "" })
	a.boxReadInputFile(good)
	waitFor(t, drain, func() bool { return a.boxPending == 0 && len(a.box.textSegments) == 1 })
	if a.box.err != "" {
		t.Fatalf("error %q stayed after a file landed", a.box.err)
	}
}

func TestBoxReadInputFileCapsTextAtOneMebibyte(t *testing.T) {
	const limit = 1 << 20
	for _, test := range []struct {
		size     int
		accepted bool
	}{{size: limit, accepted: true}, {size: limit + 1, accepted: false}} {
		a, _, _ := newTestApp(t)
		a.box = rewriteBox{open: true, shown: true}
		drain := queuePosts(a)
		path := filepath.Join(t.TempDir(), "big.txt")
		if err := os.WriteFile(path, bytes.Repeat([]byte("a"), test.size), 0o600); err != nil {
			t.Fatal(err)
		}
		a.boxReadInputFile(path)
		waitFor(t, drain, func() bool { return a.boxPending == 0 })
		if accepted := len(a.box.textSegments) == 1; accepted != test.accepted {
			t.Fatalf("%d bytes: accepted %v, want %v (error %q)", test.size, accepted, test.accepted, a.box.err)
		}
		if !test.accepted && a.box.err != "This file is too large: big.txt" {
			t.Fatalf("%d bytes: error %q", test.size, a.box.err)
		}
	}
}
