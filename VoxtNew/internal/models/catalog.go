// Package models is VoxtNew's model catalogue and its models on disk: the
// few curated models (each pinned to a Hugging Face commit, with every
// file's size and SHA-256), which of them are installed, their downloads
// (resumable, verified, cancellable) and deletions.
//
// The models live in the ActNow family's model library ($HUB, the Hugging
// Face cache, ~/.cache/huggingface/hub by default; family spec §7), in
// its standard layout, shared with the other ActNow apps: actnowkit's
// hfhub downloads them holding the family's locks and claims them for
// VoxtNew, and actnowkit's models decides what a deletion may remove
// (only what no other app claims, unless the user deletes for all apps).
// The runtimes load a model from its snapshot folder
// ($HUB/models--<owner>--<name>/snapshots/<commit>, Manager.Dir).
//
// The catalogue's ids are stable: settings keep them. What a model is
// called and what it is good at is the UI's (internal/uitext); repos,
// revisions and quantisations stay here.
//
// Imports: nothing internal (actnowkit's hfhub and models).
package models

// Kind is what a model does.
type Kind string

const (
	ASR Kind = "asr" // speech recognition
	LLM Kind = "llm" // a language model
	VAD Kind = "vad" // voice activity detection: a helper model, not chosen by the user
)

// Runtime is what runs a model.
type Runtime string

const (
	Omni Runtime = "omni" // sglang_omni_mlx/native's servers
	MLX  Runtime = "mlx"  // voxtnew-llm, mlx-swift-lm
	GGUF Runtime = "gguf" // voxtnew-llm, llama.cpp
)

// The catalogue's ids.
const (
	ASRAccurate  = "asr.accurate"
	ASRFast      = "asr.fast"
	VADSilero    = "vad.silero"
	LLMBalanced  = "llm.balanced"
	LLMFast      = "llm.fast"
	LLMTranslate = "llm.translate"
	// The family's language models VoxtNew can run when another ActNow
	// app installed them (Borrowed).
	LLMQwen3  = "llm.qwen3"
	LLMGemma4 = "llm.gemma4"
)

// File is one file of a model, at its path in the repo (and in the
// snapshot).
type File struct {
	Path   string `json:"path"`
	Size   int64  `json:"size"`
	SHA256 string `json:"sha256"`
	// LFS: the Hub keeps it in Git LFS, so that its blob is named by its
	// SHA-256 (the others' by their git blob id, which the download
	// checks instead).
	LFS bool `json:"lfs,omitempty"`
}

// Model is one model of the catalogue.
type Model struct {
	ID      string
	Kind    Kind
	Runtime Runtime
	// Repo and Revision are the Hugging Face repo and the commit pinned,
	// the family's pins table's (actnowkit models.PinsOfApp("voxtnew")).
	Repo, Revision string
	// Accept are other commits the runtimes were checked with: one of
	// them on disk is used instead of downloading the pinned one (family
	// spec §7.2).
	Accept []string
	// Files are the files used (the snapshot holds only those when VoxtNew
	// downloads it); nil: the whole snapshot (Borrowed models).
	Files []File
	// Bytes is the size when Files is nil.
	Bytes int64
	// Family and Quant are the family's model series ("qwen3-asr") and
	// quantisation ("8bit"), written into the claims.
	Family, Quant string
	// Default: the recommended model of its kind.
	Default bool
	// Features are the only features it serves ("translate"), nil for
	// every one.
	Features []string
	// Needs are models installed with it (a speech model: the VAD).
	Needs []string
	// Alternatives are models that may stand in for it while it is not
	// installed (the same kind, lighter or another app's), offered to the
	// user, never chosen silently (family spec §7.2, §7.10.1).
	Alternatives []string
	// Borrowed: a model of another ActNow app that VoxtNew can run but
	// does not download: it is on offer only when it is on disk, and is
	// VoxtNew's once the user chose to use it (a claim).
	Borrowed bool
}

// Size is the size of the model's files.
func (m Model) Size() int64 {
	if len(m.Files) == 0 {
		return m.Bytes
	}
	var n int64
	for _, f := range m.Files {
		n += f.Size
	}
	return n
}

// Catalog is every model, the recommended first of each kind.
func Catalog() []Model {
	out := make([]Model, len(catalog))
	copy(out, catalog)
	return out
}

// Lookup is the model of id.
func Lookup(id string) (Model, bool) {
	for _, m := range catalog {
		if m.ID == id {
			return m, true
		}
	}
	return Model{}, false
}

// The pins are the family pins table's (TestFamilyPins); the omni
// models' are also sglang_omni_mlx/native/ci/models.json's (TestOmniPins).
// The language models' files are those Voxt's loaders take (weights,
// configs, tokenizer, chat template): Qwen3.5-4B-OptiQ without its
// optiq/ vision tower and MTP head, Hy-MT2's one GGUF file of the repo.
// The borrowed models are Dicta's and AudioSmith's, whole snapshots.
var catalog = []Model{
	{
		ID: ASRAccurate, Kind: ASR, Runtime: Omni, Default: true, Needs: []string{VADSilero},
		Family: "qwen3-asr", Quant: "8bit", Alternatives: []string{ASRFast},
		Repo: "mlx-community/Qwen3-ASR-1.7B-8bit", Revision: "a8379a2e2f9e313c9292cdf1af4055ab56d50d55",
		Files: []File{
			{"chat_template.json", 1161, "75a8cfca24f00de72d796fbfed6858fc9614ef3dabd8696684cc3bc03a9c58ff", false},
			{"config.json", 7188, "1b76b3b6c655fc54595da025f7a96474ad9fa86363303fbdd61a7d8483ccfaf7", false},
			{"generation_config.json", 142, "1da527824d81e07118facff437e03f2e24a23311e3bdeb2368973fe77e5f275c", false},
			{"merges.txt", 1671853, "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5", false},
			{"model.safetensors", 2463307541, "bf304b009cc7eca79283056f787b44c952d24ac22cec787b39732bba3c23c13c", true},
			{"model.safetensors.index.json", 78968, "0a5d0ec11188602242ff81a9969883d0fdeb98cd5d85cd1413089d897c201af5", false},
			{"preprocessor_config.json", 330, "45e120a4eda2c20c5d7f2ea9354e63536bf35e27aa573fb7cdf78017b378770d", false},
			{"tokenizer_config.json", 12487, "4942d005604266809309cabc9f4e9cb89ce855d59b14681fdc0e1cc62ea26c4c", false},
			{"vocab.json", 2776833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910", false},
		},
	},
	{
		ID: ASRFast, Kind: ASR, Runtime: Omni, Needs: []string{VADSilero},
		Family: "qwen3-asr", Quant: "4bit",
		Repo: "mlx-community/Qwen3-ASR-0.6B-4bit", Revision: "313d850181767edf09f00a9c289becca70e58cd0",
		Files: []File{
			{"chat_template.json", 1161, "75a8cfca24f00de72d796fbfed6858fc9614ef3dabd8696684cc3bc03a9c58ff", false},
			{"config.json", 7187, "923618cf5ca452fda0253a6be5c1a17f94a2e4851d3b98beb45848565587bd72", false},
			{"generation_config.json", 142, "1da527824d81e07118facff437e03f2e24a23311e3bdeb2368973fe77e5f275c", false},
			{"merges.txt", 1671853, "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5", false},
			{"model.safetensors", 708236945, "70c7e67e588062adce4f10796e47ad42ead51c6671eda61a0987eae38ca95ddf", true},
			{"model.safetensors.index.json", 71814, "e3bb80ef0fd42a5be07b04e90c97d60460bbde8af3531e0bfe9100a61404d81a", false},
			{"preprocessor_config.json", 330, "45e120a4eda2c20c5d7f2ea9354e63536bf35e27aa573fb7cdf78017b378770d", false},
			{"tokenizer_config.json", 12487, "4942d005604266809309cabc9f4e9cb89ce855d59b14681fdc0e1cc62ea26c4c", false},
			{"vocab.json", 2776833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910", false},
		},
	},
	{
		ID: VADSilero, Kind: VAD, Runtime: Omni, Default: true, Family: "silero-vad",
		Repo: "mlx-community/silero-vad-v6", Revision: "2ebf4a5e10726a2e78ddd4d70eedfb6f1c33eb06",
		Files: []File{
			{"config.json", 463, "9fe1befb9692a0d4135adadc33f8075ef6d350bd2391b88d750f2c233f97fa0b", false},
			{"model.safetensors", 1237860, "65b6c5f0293cbc44d109e58bef78b474d9c65dedbee814cf0b90ef5f0d9150ff", true},
		},
	},
	{
		ID: LLMBalanced, Kind: LLM, Runtime: MLX, Default: true,
		Family: "qwen3.5", Quant: "4bit", Alternatives: []string{LLMQwen3, LLMGemma4},
		Repo: "mlx-community/Qwen3.5-4B-OptiQ-4bit", Revision: "6cb5bdfd0bf15f484881fb9f1ab6d7c840fddde9",
		Files: []File{
			{"chat_template.jinja", 7756, "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715", false},
			{"config.json", 55543, "aa725a49a83c5503506ff7dcbc57efdaeb7e64a5fa6e4b547d49979a29aec1b3", false},
			{"generation_config.json", 148, "f2130efdf66df1834aa4f7a6e48752e88a8dff41965739c6ffb2bb26c841615e", false},
			{"kv_config.json", 528, "5de3659fa5db489d9ae8ef95eee8963ae6c778099a4de06de7994cce61d82129", false},
			{"model.safetensors", 3269669552, "5d59fea455202270c0f7f71a1902516b06457df5600c3182e1ff9163c33287c1", true},
			{"model.safetensors.index.json", 100945, "67306a1f0b0181d92adfe92fce654b6c91ee0c5c2b51b0de114ced2a2ecdbcbc", false},
			{"optiq_metadata.json", 26224, "38eb38a541e2b169c258630fd4a64b2bf256a024abfc55ff0935f0ec495d4be2", false},
			{"tokenizer.json", 19989343, "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4", true},
			{"tokenizer_config.json", 1135, "b2bed5e033438f09f22b0ce9522115b4807d9bbcc3b82bf23372cb74d93ed081", false},
		},
	},
	{
		ID: LLMFast, Kind: LLM, Runtime: MLX,
		Family: "qwen3.5", Quant: "4bit", Alternatives: []string{LLMQwen3, LLMGemma4},
		Repo: "mlx-community/Qwen3.5-2B-4bit", Revision: "674aaa7240b91e8012fcad5d791b7dfe5ba90207",
		Files: []File{
			{"chat_template.jinja", 7755, "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80", false},
			{"config.json", 3113, "beb7fc5a6e0405fe332821cf1a8ef7b69bb390a8c8933171647de5579debf949", false},
			{"model.safetensors", 1722271785, "713fe7e5d3c3965f7106b0d0ee17615f7869c23c8d327996df8c1196fbcf07d5", true},
			{"model.safetensors.index.json", 81722, "8294c05cca7d53a6c33e3db2b379539bd296d054e0b689711b16b6ac93c7e49d", false},
			{"preprocessor_config.json", 390, "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516", false},
			{"processor_config.json", 1300, "14932921ca485d458a04dafd8069fbb0a4505622a48208d19ed247115801385b", false},
			{"tokenizer.json", 19989343, "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4", true},
			{"tokenizer_config.json", 1139, "e98f1901ac6f0adff67b1d540bfa0c36ac1a0cf59eb72ed78146ef89aafa1182", false},
			{"video_preprocessor_config.json", 385, "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13", false},
			{"vocab.json", 6722759, "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003", false},
		},
	},
	{
		ID: LLMTranslate, Kind: LLM, Runtime: GGUF, Features: []string{"translate"},
		Family: "hy-mt2", Quant: "Q4_K_M",
		Repo: "tencent/Hy-MT2-1.8B-GGUF", Revision: "a0c709d9fac510f2c807aa3af52872340dc37a4a",
		Files: []File{
			{"Hy-MT2-1.8B-Q4_K_M.gguf", 1133080448, "dc5f44fcf1fa496ee7ad725982c0c8c553a4de00259b53af84c4b89fb0c06699", true},
		},
	},
	{
		ID: LLMQwen3, Kind: LLM, Runtime: MLX, Borrowed: true,
		Family: "qwen3", Quant: "4bit", Bytes: 2278969697,
		Repo: "mlx-community/Qwen3-4B-Instruct-2507-4bit", Revision: "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b",
	},
	{
		ID: LLMGemma4, Kind: LLM, Runtime: MLX, Borrowed: true,
		Family: "gemma4", Quant: "4bit", Bytes: 5179239349,
		Repo: "mlx-community/gemma-4-e4b-it-4bit", Revision: "475b9088d29754a3379866cf5aeb6b41acd313c2",
	},
}
