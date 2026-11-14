package main

import (
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
)

func TestPromptSaveUsesFullSystem(t *testing.T) {
	a, tt := pageTester(t, "/translate")
	click(t, tt, "Your rules")
	click(t, tt, "System prompt for Translate")
	custom := "CUSTOM PROMPT LINE"
	a.fui(hotkey.Translate).prompt = custom
	a.fui(hotkey.Translate).promptDirty = true
	a.savePrompt(hotkey.Translate)
	click(t, tt, "Example user message")
	if p := a.lib.settings.Get().Features.Translate.Prompt; p != custom {
		t.Fatalf("saved %q", p)
	}
	out, err := a.composedPreview(hotkey.Translate)
	if err != nil {
		t.Fatal(err)
	}
	if out.System != custom {
		t.Fatalf("composed system %q", out.System)
	}
}

func TestPromptResetRestoresBuiltin(t *testing.T) {
	a, tt := pageTester(t, "/translate")
	click(t, tt, "Your rules")
	builtin := a.promptBuiltin(hotkey.Translate)
	if strings.TrimSpace(builtin) == "" {
		t.Fatal("no builtin")
	}
	click(t, tt, "System prompt for Translate")
	a.fui(hotkey.Translate).prompt = "TEMP OVERRIDE"
	a.fui(hotkey.Translate).promptDirty = true
	a.savePrompt(hotkey.Translate)
	click(t, tt, "Reset to default prompt")
	if got := a.lib.settings.Get().Features.Translate.Prompt; got != "" {
		t.Fatalf("settings %q", got)
	}
	if strings.TrimSpace(a.fui(hotkey.Translate).prompt) != strings.TrimSpace(builtin) {
		t.Fatalf("ui %q", a.fui(hotkey.Translate).prompt)
	}
}

// TestRulesFieldGrows: the rules field is as high as its text, wrapped
// at its width, however long: a long prompt reads down the page, under
// one scroll bar, the page's. It was a few lines high with a scroll bar of
// its own, then at most 36 lines high, scrolling inside the page.
func TestRulesFieldGrows(t *testing.T) {
	a, tt := pageTester(t, "/transcribe")
	tt.SetSize(windowWidth, 9000) // room for the whole field
	click(t, tt, "Your rules")
	const label = "System prompt for Transcribe"
	u := a.fui(hotkey.Transcribe)
	u.prompt, u.promptDirty = "x", true
	frames(tt)
	short, _ := tt.Find(label)
	u.prompt, u.promptDirty = strings.Repeat("Keep product names in English.\n", 200), true
	frames(tt)
	long, _ := tt.Find(label)
	if line := short.H / promptMinLines; long.H < 150*line {
		t.Errorf("a 200-line prompt's field is %.0f high: %.0f lines", long.H, long.H/line)
	}
}

// TestRulesFieldLetsThePageScroll: a scroll that starts on the page goes
// on scrolling the page when the rules field comes under the pointer, as
// on macOS (MyGo patch 0011: the field took the page's scroll over as
// soon as it reached the pointer); one that starts over the field
// scrolls the page too, the field having nothing of its own to scroll.
func TestRulesFieldLetsThePageScroll(t *testing.T) {
	a, _, tt := newTestApp(t)
	a.navigate("/transcribe")
	frames(tt)
	const label = "System prompt for Transcribe"
	u := a.fui(hotkey.Transcribe)
	u.prompt, u.promptDirty = strings.Repeat("Keep product names in English and write numbers as digits.\n", 200), true
	x, y := float32(windowWidth/2), float32(windowHeight/2)
	tt.Move(x, y)
	if title, ok := tt.Find("Your rules"); ok { // the section at the top, so that it opens in view
		tt.Scroll(x, y, 0, title.Y-60)
		frames(tt)
		tt.Advance(time.Second)
	}
	click(t, tt, "Your rules")
	field, ok := tt.Find(label)
	if !ok {
		t.Fatalf("no rules field: %q", firstTexts(tt))
	}
	tt.Scroll(x, y, 0, field.Y-y-20) // the page: the field's top just under the pointer
	frames(tt)
	before, _ := tt.Find(label)
	for range 4 { // the same gesture, the field under the pointer now
		tt.Scroll(x, y, 0, 60)
	}
	frames(tt)
	after, _ := tt.Find(label)
	if before.Y-after.Y < 200 {
		t.Fatalf("the page stopped when the field came under the pointer: the field from %.0f to %.0f", before.Y, after.Y)
	}
	// The field's top back in view, 20 above the pointer: a new gesture
	// over the field.
	tt.Advance(time.Second)
	tt.Scroll(x, y, 0, -300)
	frames(tt)
	field, _ = tt.Find(label)
	tt.Advance(time.Second)
	tt.Scroll(x, y, 0, field.Y-(y-20))
	frames(tt)
	before, _ = tt.Find(label)
	tt.Advance(time.Second)
	tt.Scroll(x, y, 0, 60)
	frames(tt)
	if again, _ := tt.Find(label); before.Y-again.Y < 30 {
		t.Errorf("a scroll that starts over the field left the page: the field from %.0f to %.0f", before.Y, again.Y)
	}
}

// openLongRules shows Transcribe with 200 lines of rules and their
// section open at the top of the default window.
func openLongRules(t *testing.T, end string) (*app, *ui.Tester) {
	t.Helper()
	a, _, tt := newTestApp(t)
	a.navigate("/transcribe")
	frames(tt)
	u := a.fui(hotkey.Transcribe)
	u.prompt, u.promptDirty = strings.Repeat("Keep product names in English.\n", 200)+end, true
	x, y := float32(windowWidth/2), float32(windowHeight/2)
	if title, ok := tt.Find("Your rules"); ok {
		tt.Scroll(x, y, 0, title.Y-60)
		frames(tt)
		tt.Advance(time.Second)
	}
	click(t, tt, "Your rules")
	return a, tt
}

// TestRulesCaretInView: in the default window the caret stays in view as
// it moves to the end of a long prompt: the page scrolls to it, the field
// having nothing of its own to scroll (MyGo patch 0012).
func TestRulesCaretInView(t *testing.T) {
	_, tt := openLongRules(t, "")
	field, ok := tt.Find("System prompt for Transcribe")
	if !ok {
		t.Fatalf("no rules field: %q", firstTexts(tt))
	}
	at := max(field.Y+10, float32(titleBarHeight+20))
	tt.Press(field.X+20, at)
	tt.Release(field.X+20, at)
	tt.Key(ui.Ctrl, ui.KeyEnd)
	frames(tt)
	if caret, ok := tt.TextCaret(); !ok || caret.Y < float32(titleBarHeight) || caret.Y+caret.H > windowHeight {
		t.Errorf("the caret at the end is at %+v (%v), out of the window's %d..%d", caret, ok, titleBarHeight, windowHeight)
	}
}

// TestRulesFindShowsMatch: Return in Find shows the match in the default
// window, the page scrolling to it, marked in red (#3869); the match
// stayed below the window.
func TestRulesFindShowsMatch(t *testing.T) {
	_, tt := openLongRules(t, "zebra")
	tt.Key(ui.Cmd, ui.KeyF)
	frames(tt)
	tt.Type("zebra")
	tt.Key(0, ui.KeyEnter)
	frames(tt)
	field, ok := tt.Find("System prompt for Transcribe")
	if !ok {
		t.Fatal("no rules field")
	}
	img, red := tt.Image(), tok(ui.LightTheme()).Danger
	seen := 0
	for y := max(int(field.Y)+4, titleBarHeight); y < min(int(field.Y+field.H)-4, windowHeight); y++ {
		for x := int(field.X) + 4; x < int(field.X+field.W)-4; x++ {
			if near(img.RGBAAt(x, y), red) {
				seen++
			}
		}
	}
	if seen < 10 {
		t.Errorf("the match is not in view: %d red pixels in the field's part of the window", seen)
	}
}

// TestRulesFind: ⌘F on a feature page opens Your rules with Find
// focused; Find counts the matches, ignoring case, and Return steps
// through them, back to the first after the last, Shift+Return back.
func TestRulesFind(t *testing.T) {
	a, tt := pageTester(t, "/transcribe")
	u := a.fui(hotkey.Transcribe)
	u.prompt, u.promptDirty = "Use dLLM terms.\nNever expand DLLM.\nplain line", true
	tt.Key(ui.Cmd, ui.KeyF)
	frames(tt)
	if !u.rules || !tt.Focused("Find in the prompt") {
		t.Fatalf("after ⌘F: rules open %v, Find focused %v", u.rules, tt.Focused("Find in the prompt"))
	}
	tt.Type("dllm")
	if !has(tt, "2 matches") {
		t.Fatalf("no count: %q", firstTexts(tt))
	}
	for _, want := range []string{"1 of 2", "2 of 2", "1 of 2"} {
		tt.Key(0, ui.KeyEnter)
		if !has(tt, want) {
			t.Errorf("after Return: no %q (found %d)", want, u.finder.cur)
		}
	}
	for _, want := range []string{"2 of 2", "1 of 2"} {
		tt.Key(ui.Shift, ui.KeyEnter)
		if !has(tt, want) {
			t.Errorf("after Shift+Return: no %q (found %d)", want, u.finder.cur)
		}
	}
	if got := findFolded(u.prompt, "DLLM"); !slices.Equal(got, [][2]int{{4, 8}, {29, 33}}) {
		t.Errorf("matches %v", got)
	}
}

// TestTranslateBuiltinPromptKeepsTarget: the rules field starts from the
// built-in prompt with {{TARGET_LANGUAGE}} in it, so a prompt edited from
// it follows the menu. It had the language's name written in (#3956): the
// edited prompt kept translating into it whatever the menu said.
func TestTranslateBuiltinPromptKeepsTarget(t *testing.T) {
	a, tt := pageTester(t, "/translate")
	builtin := a.promptBuiltin(hotkey.Translate)
	if !strings.Contains(builtin, targetVar) || strings.Contains(builtin, "Simplified Chinese") {
		t.Fatalf("built-in prompt %q", builtin)
	}
	u := a.fui(hotkey.Translate)
	u.prompt, u.promptDirty = builtin+"\nUse British spelling.", true
	a.savePrompt(hotkey.Translate)
	click(t, tt, "Target language for Translate")
	click(t, tt, "Japanese")
	out, err := a.composedPreview(hotkey.Translate)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.System, "into Japanese") || strings.Contains(out.System, "{{") {
		t.Fatalf("edited prompt does not follow the menu: %q", out.System)
	}
	if has(tt, "Your prompt does not use {{TARGET_LANGUAGE}}, so this menu does nothing.") {
		t.Error("warning for a prompt that uses the variable")
	}
}

// TestTranslateWarnsPromptIgnoresTarget: a prompt of one's own without
// {{TARGET_LANGUAGE}} ("into English", in English or Chinese) gets a
// warning on the page and a button that adds it; none for the built-in
// prompt (#3956).
func TestTranslateWarnsPromptIgnoresTarget(t *testing.T) {
	const warning = "Your prompt does not use {{TARGET_LANGUAGE}}, so this menu does nothing."
	a, tt := pageTester(t, "/translate")
	for _, s := range []string{"Target language", "The language model translates into this. Speech recognition writes what you said in your own language. Automatic is English, or Chinese when your main language is English."} {
		if !has(tt, s) {
			t.Errorf("no %q", s)
		}
	}
	if has(tt, warning) {
		t.Fatal("warning with the built-in prompt")
	}
	u := a.fui(hotkey.Translate)
	for _, own := range []string{"Translate Chinese into English.\n\n", "把中文翻译成英语。"} {
		u.prompt, u.promptDirty = own, true
		a.savePrompt(hotkey.Translate)
		frames(tt)
		if !has(tt, warning) {
			t.Fatalf("no warning for %q", own)
		}
		click(t, tt, "Add {{TARGET_LANGUAGE}} to the prompt")
		saved := a.lib.settings.Get().Features.Translate.Prompt
		if want := strings.TrimRight(own, "\n") + "\n\nTranslate the text into {{TARGET_LANGUAGE}}."; saved != want {
			t.Fatalf("saved %q, want %q", saved, want)
		}
		if has(tt, warning) {
			t.Fatalf("warning after adding the variable to %q", own)
		}
		out, err := a.composedPreview(hotkey.Translate)
		if err != nil || !strings.Contains(out.System, "into Simplified Chinese.") {
			t.Fatalf("composed %q, %v", out.System, err)
		}
	}
	// Typed but not saved yet (the field saves 800 ms after the last key):
	// the button keeps it.
	u.prompt, u.promptDirty = "Into English.", true
	a.savePrompt(hotkey.Translate)
	u.prompt, u.promptDirty = "Into English. Keep it short.", true
	frames(tt)
	click(t, tt, "Add {{TARGET_LANGUAGE}} to the prompt")
	if saved := a.lib.settings.Get().Features.Translate.Prompt; saved != "Into English. Keep it short.\n\nTranslate the text into {{TARGET_LANGUAGE}}." {
		t.Fatalf("typed text lost: %q", saved)
	}
	click(t, tt, "Your rules")
	click(t, tt, "Reset to default prompt")
	if has(tt, warning) {
		t.Error("warning after the reset")
	}
}
