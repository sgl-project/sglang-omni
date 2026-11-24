package dict

import (
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// builtinTerm is a term every install knows.
type builtinTerm struct {
	term    string
	aliases []string
	// spelled: a name no ordinary word shares, so another case of it is
	// a misspelling ("Github", "hugging face") and is written as spelled
	// here (matchText keeps one word all in one case: sglang, VOXT_HOME).
	// Not for words like Cursor or prompt, nor for a style (API Key).
	spelled bool
}

// builtin is the vocabulary every install starts with: Voxt, the
// repository it is made in, and AI and developer names its users say
// often, each with what speech recognition writes for it instead
// (Whisper Large v3 Turbo on macOS voices, docs/eval.md "Built-in
// vocabulary (11-23)"). An alias is never an ordinary word or name ("cloud" for
// Claude, "Quinn" for Qwen), so that a sentence without the term stays
// as said. It is not in the store: the user's dictionary stays theirs.
var builtin = []builtinTerm{
	{term: "SGLang Omni", aliases: []string{"SGLOMNI"}, spelled: true},
	{term: "SGLang", aliases: []string{"SG Lang", "SG-Lang", "S.G.Lang", "SGLand", "SGLAN", "SG LAN", "SGLONG", "SG Long", "S G Long", "SG狼"}, spelled: true},
	{term: "Voxt", aliases: []string{"Vox T", "Vox", "Voxed", "Voist", "沃克斯特"}, spelled: true},
	{term: "Qwen", aliases: []string{"Quen", "QEN"}, spelled: true},
	{term: "DeepSeek", aliases: []string{"Deep Seek", "DeepSeq", "Deep Sick", "DeepSick", "DeepSig"}, spelled: true},
	{term: "ChatGPT", aliases: []string{"Chat GPT", "ChatsGPT", "ChadGPT", "Chad GPT", "ChatGTT"}, spelled: true},
	{term: "OpenAI", aliases: []string{"OpenEye"}, spelled: true},
	{term: "GitHub", aliases: []string{"Jithub"}, spelled: true},
	{term: "Hugging Face", aliases: []string{"Hogging Face", "HoggingFace", "Hoggin Face"}, spelled: true},
	{term: "CUDA", aliases: []string{"QDA"}, spelled: true},
	{term: "API key", aliases: []string{"APIQ"}},
	{term: "Gemini", aliases: []string{"Geminar", "Gamini"}},
}

// ReplaceWithBuiltin is Replace over terms, the user's dictionary, then
// the built-in vocabulary. The user's terms win: a built-in alias that
// is one of the user's terms or aliases (ignoring case, enabled or not)
// is left out, so a word the user keeps is never rewritten; a built-in
// term the user turned off is left out; one the user has keeps the
// user's spelling and gains the built-in aliases.
func ReplaceWithBuiltin(text string, terms []store.Term) string {
	all, spelled := withBuiltin(terms)
	text, _ = matchText(text, all, false, spelled)
	return text
}

// withBuiltin is terms followed by the built-in terms they leave (see
// ReplaceWithBuiltin), and the built-in terms written as spelled in any
// case.
func withBuiltin(terms []store.Term) ([]store.Term, map[string]bool) {
	words := map[string]bool{}      // the user's terms and aliases
	spelling := map[string]string{} // an enabled term, as the user writes it
	off := map[string]bool{}        // a term the user turned off
	for _, t := range terms {
		key := strings.ToLower(oneLine(t.Term))
		words[key] = true
		if t.Enabled {
			spelling[key] = oneLine(t.Term)
		} else {
			off[key] = true
		}
		for _, a := range t.Aliases {
			words[strings.ToLower(oneLine(a))] = true
		}
	}
	out := append([]store.Term(nil), terms...)
	spelled := map[string]bool{}
	for _, b := range builtin {
		key := strings.ToLower(b.term)
		user := spelling[key]
		if user == "" && off[key] {
			continue
		}
		var aliases []string
		for _, a := range b.aliases {
			if !words[strings.ToLower(a)] {
				aliases = append(aliases, a)
			}
		}
		switch {
		case user != "" && len(aliases) == 0:
			continue
		case user != "":
			out = append(out, store.Term{Term: user, Aliases: aliases, Enabled: true})
		default:
			out = append(out, store.Term{Term: b.term, Aliases: aliases, Enabled: true})
			spelled[b.term] = b.spelled
		}
	}
	return out, spelled
}
