package main

// The dictionaries of the dictionary's evals (eval_dictasr_test.go,
// eval_dictllm_test.go) and their helpers.

import (
	"encoding/binary"
	"math"
	"math/rand"
	"slices"
	"strings"
	"testing"
	"unicode"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/eval"
)

// dictSmall is the user's dictionary of 10-19 (the most recently changed
// first, as dict.Hints sends it).
var dictSmall = []string{"Voxt", "Omni", "SGLang"}

// dictMedium is 30 terms: the eval set's, the dict2 clips' and other
// product names.
var dictMedium = []string{"Voxt", "Omni", "SGLang", "VoxtNew", "MLX", "AudioSmith", "Pixcall", "ActNow", "Silero", "Qwen3", "Auralink",
	"陈喆", "韩煜", "周芷若", "欧阳明远", "PyTorch", "Hugging Face", "DeepSeek", "Ollama", "llama.cpp", "TestFlight", "Xcode", "SwiftUI",
	"MyGo", "Dicta", "Grokbot", "Notion", "Figma", "Gemini", "Claude"}

// dictLarge is dictMedium and about a hundred more: tools, people's
// names, Chinese jargon.
func dictLarge() []string {
	return append(slices.Clone(dictMedium),
		"Kubernetes", "Terraform", "Grafana", "Prometheus", "Redis", "PostgreSQL", "ClickHouse", "Kafka", "RabbitMQ",
		"Elasticsearch", "Nginx", "Docker", "GitHub", "GitLab", "Jenkins", "Sentry", "Datadog", "Linear", "Jira", "Confluence", "Slack",
		"Discord", "Telegram", "WeChat", "Feishu", "DingTalk", "Obsidian", "Raycast", "Alfred", "Arc", "Safari", "Chrome", "Firefox",
		"Electron", "Tauri", "Flutter", "React", "Vue", "Svelte", "Next.js", "Nuxt", "Vite", "Webpack", "esbuild", "Bun", "Deno",
		"Node.js", "TypeScript", "Rust", "Golang", "Swift", "Kotlin", "Python", "Jupyter", "NumPy", "pandas", "Transformers", "vLLM",
		"TensorRT", "CUDA", "Metal", "CoreML", "ONNX", "Whisper", "SenseVoice", "Paraformer", "FunASR", "CosyVoice", "Kokoro",
		"Stable Diffusion", "ComfyUI", "Midjourney", "LangChain", "LlamaIndex", "Dify", "Cursor", "Copilot", "Windsurf", "Zed",
		"Vercel", "Supabase", "Tailwind", "Cloudflare", "张伟", "王芳", "李娜", "刘洋", "司马青", "灰度发布", "埋点", "冷启动", "热更新",
		"降本增效", "飞书多维表格", "钉钉宜搭", "小程序", "公众号", "抖音", "小红书", "哔哩哔哩")
}

// padAudio is pcm with sec seconds more at its end: room noise at
// noiseDB dBFS RMS (a microphone's silence after the level stage), or
// digital silence when noiseDB is 0.
func padAudio(pcm []byte, sec, noiseDB float64, seed int64) []byte {
	n := int(sec * asr.SampleRate)
	out := slices.Grow(slices.Clone(pcm), n*2)
	r := rand.New(rand.NewSource(seed))
	amp := 0.0
	if noiseDB != 0 {
		amp = math.Pow(10, noiseDB/20) * 32768
	}
	for range n {
		v := r.NormFloat64() * amp
		out = binary.LittleEndian.AppendUint16(out, uint16(int16(max(-32768, min(32767, v)))))
	}
	return out
}

// foldTerm is text without case, spaces and the width of its characters.
func foldTerm(s string) string {
	return strings.Join(strings.Fields(eval.Normalize(s)), "")
}

// hasTerm says whether text writes term, ignoring case and spaces; a
// term that starts or ends with a letter or digit must not run on into
// another word there ("Arc" is not in "search").
func hasTerm(text, term string) bool {
	f := []rune(foldTerm(term))
	var t []rune
	var spaced []bool // a space before t[i]
	space := false
	for _, r := range eval.Normalize(text) {
		if unicode.IsSpace(r) {
			space = true
			continue
		}
		t, spaced, space = append(t, r), append(spaced, space), false
	}
	if len(f) == 0 {
		return false
	}
	word := func(r rune) bool { return r < 0x2E80 && (unicode.IsLetter(r) || unicode.IsDigit(r)) }
	for i := 0; i+len(f) <= len(t); i++ {
		end := i + len(f)
		if string(t[i:end]) != string(f) {
			continue
		}
		if word(f[0]) && i > 0 && !spaced[i] && word(t[i-1]) || word(f[len(f)-1]) && end < len(t) && !spaced[end] && word(t[end]) {
			continue
		}
		return true
	}
	return false
}

// termsPutIn are the terms out writes that neither of said writes.
func termsPutIn(terms []string, out string, said ...string) []string {
	var extra []string
	for _, term := range terms {
		if hasTerm(out, term) && !slices.ContainsFunc(said, func(s string) bool { return hasTerm(s, term) }) {
			extra = append(extra, term)
		}
	}
	return extra
}

func TestHasTerm(t *testing.T) {
	for _, c := range []struct {
		text, term string
		want       bool
	}{
		{"我们用 Omni。", "Omni", true},
		{"我们用omni", "Omni", true},
		{"Hugging  face hub", "Hugging Face", true},
		{"search the docs", "Arc", false},
		{"the Arc browser", "Arc", true},
		{"Arc。", "Arc", true},
		{"OmniBus", "Omni", false},
		{"请陈喆来", "陈喆", true},
		{"ＳＧＬａｎｇ", "SGLang", true},
	} {
		if got := hasTerm(c.text, c.term); got != c.want {
			t.Errorf("hasTerm(%q, %q) = %v", c.text, c.term, got)
		}
	}
}
