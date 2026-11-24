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
	// hint: sent to the speech model after the user's terms (BuiltinHints).
	hint bool
}

// sglangAliases are what speech recognition writes for SGLang (Whisper,
// and Qwen3-ASR without the hint: "S. G. Lang", "S G狼").
var sglangAliases = []string{"SG Lang", "SG-Lang", "S.G.Lang", "S.G. Lang", "S. G. Lang", "SGLand", "SG Land", "S G Land", "SGLAN", "SG LAN",
	"SGLONG", "SG Long", "S G Long", "SGLUNG", "SG狼", "S G狼", "S G 龙"}

// builtin is the vocabulary every install starts with: Voxt, the
// repository it is made in, and AI, developer and product names its
// users say often. Aliases are what speech recognition wrote for a name
// (Whisper Large v3 Turbo on macOS voices, docs/eval.md "Built-in
// vocabulary (11-23)"), never an ordinary word, term or name ("cloud"
// for Claude, "Quinn" for Qwen, "Vox" for Voxt, "QDA" for CUDA,
// "Gamini" for Gemini), so that a sentence without the name stays as
// said. The rest only get their spelling back. It is not in the store:
// the user's dictionary stays theirs.
var builtin = append([]builtinTerm{
	{term: "SGLang-Omni", aliases: append([]string{"SGLang Omni", "SGLOMNI"}, suffixed(sglangAliases, " Omni")...), spelled: true},
	{term: "SGLang", aliases: sglangAliases, spelled: true, hint: true},
	{term: "Voxt", aliases: []string{"Vox T", "Voxed", "Voist", "沃克斯特"}, spelled: true, hint: true},
	{term: "Qwen", aliases: []string{"Quen", "QEN"}, spelled: true},
	{term: "DeepSeek", aliases: []string{"Deep Seek", "DeepSeq", "Deep Sick", "DeepSick", "DeepSig"}, spelled: true, hint: true},
	{term: "ChatGPT", aliases: []string{"Chat GPT", "ChatsGPT", "ChadGPT", "Chad GPT", "ChatGTT"}, spelled: true},
	{term: "OpenAI", aliases: []string{"OpenEye"}, spelled: true},
	{term: "GitHub", aliases: []string{"Jithub"}, spelled: true},
	{term: "Hugging Face", aliases: []string{"Hogging Face", "HoggingFace", "Hoggin Face"}, spelled: true},
	{term: "CUDA", spelled: true},
	{term: "API key", aliases: []string{"APIQ", "APIT", "API T", "APIP"}, hint: true},
	{term: "Gemini", aliases: []string{"Geminar"}},
	{term: "Claude Code", spelled: true, hint: true},
	{term: "MCP", aliases: lettersOf("MCP"), spelled: true},
	{term: "LLM", aliases: lettersOf("LLM"), spelled: true},
	{term: "JSON", spelled: true},
	{term: "TypeScript", spelled: true},
	{term: "PR", aliases: []string{"P.R.", "P.R", "P R"}, spelled: true},
	{term: "PRD", aliases: []string{"P.R.D.", "P.R.D", "P R D"}, spelled: true},
}, spelledOnly(
	// AI and coding
	"Anthropic", "GPT-5", "Gemini CLI", "GitHub Copilot", "GLM", "Replit", "PyTorch", "KV cache", "FP8", "INT4",
	"TTS", "ASR", "VLM", "GPU", "MLX", "API", "SDK", "CLI", "YAML", "README", "SSH",
	"JavaScript", "SwiftUI", "Xcode", "npm", "pnpm", "PostgreSQL", "SQLite", "Cloudflare", "Supabase",
	// product work
	"OKR", "KPI", "MVP", "SaaS", "B2B", "PMF", "GMV", "Google Docs",
	// everyday
	"WeChat", "iPhone", "iPad", "MacBook", "YouTube", "TikTok", "Gmail", "Wi-Fi", "PDF", "PPT",
)...)

// spelledOnly is a built-in term for each name that only gets its
// spelling back (an acronym also from its letters spaced out).
func spelledOnly(names ...string) []builtinTerm {
	out := make([]builtinTerm, len(names))
	for i, n := range names {
		out[i] = builtinTerm{term: n, aliases: lettersOf(n), spelled: true}
	}
	return out
}

// lettersOf is an acronym said letter by letter as speech recognition
// writes it ("L L M" for LLM; Qwen3-ASR does), nil for another name.
func lettersOf(name string) []string {
	if len(name) < 2 || len(name) > 4 || strings.ToUpper(name) != name || strings.ContainsFunc(name, func(r rune) bool { return r < 'A' || r > 'Z' }) {
		return nil
	}
	return []string{strings.Join(strings.Split(name, ""), " ")}
}

// suffixed is each of list with suffix after it.
func suffixed(list []string, suffix string) []string {
	out := make([]string, len(list))
	for i, s := range list {
		out[i] = s + suffix
	}
	return out
}

// BuiltinHints are the built-in terms for the speech model, after the
// user's hints: the ones marked hint that the user does not have
// (ignoring case, enabled or not: a term the user turned off stays off).
func BuiltinHints(terms []store.Term) []string {
	user := map[string]bool{}
	for _, t := range terms {
		user[strings.ToLower(oneLine(t.Term))] = true
	}
	var out []string
	for _, b := range builtin {
		if b.hint && !user[strings.ToLower(b.term)] {
			out = append(out, b.term)
		}
	}
	return out
}

// ReplaceWithBuiltin is Replace over terms, the user's dictionary, then
// the built-in vocabulary. The user's terms win: a built-in alias that
// is one of the user's terms or aliases (ignoring case, enabled or not)
// is left out, so a word the user keeps is never rewritten; a built-in
// term the user turned off is left out; one the user has keeps the
// user's spelling and gains the built-in aliases.
func ReplaceWithBuiltin(text string, terms []store.Term) string {
	all, spelled := withBuiltin(terms)
	text, _ = matchText(text, all, false, spelled, len(terms))
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
