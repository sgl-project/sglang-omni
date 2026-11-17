// Package models is VoxtNew's model catalogue and its models on disk: the
// few curated models (each pinned to a Hugging Face commit, with every
// file's size and SHA-256), which of them are installed, their downloads
// (resumable, verified, cancellable) and deletions.
//
// The models live in the model library ($HUB, by default
// ~/Library/Application Support/SGLang Omni/models, HubIn; the family's
// models.root when VoxtNew follows it, family spec §7), in the Hugging
// Face cache's layout, shared with the other ActNow apps that use it: actnowkit's hfhub downloads them holding the family's locks and
// claims them for VoxtNew, and actnowkit's models decides what a deletion
// may remove (only what no other app claims, unless the user deletes for
// all apps).
// The runtimes load a model from its snapshot folder
// ($HUB/models--<owner>--<name>/snapshots/<commit>, Manager.Dir).
//
// The catalogue's ids are stable: settings keep them. A model's name (its
// family, size and quantisation) is here with its repo and revision; what
// it is good at is the UI's (internal/uitext).
//
// Imports: nothing internal (actnowkit's hfhub and models).
package models

import (
	"iter"
	"slices"
)

// Kind is what a model does.
type Kind string

const (
	ASR Kind = "asr" // speech recognition
	LLM Kind = "llm" // a language model
	VAD Kind = "vad" // voice activity detection: a helper model, not chosen by the user
	Aux Kind = "aux" // files another model loads beside its own (Whisper's tokenizer), never shown
	// A speech model's speculative draft (Model.Draft): a helper model
	// the user adds from the speech model's card, never a speech model
	// of its own.
	Draft Kind = "draft"
)

// Runtime is what runs a model.
type Runtime string

const (
	Omni Runtime = "omni" // sglang_omni_mlx/native's servers
	MLX  Runtime = "mlx"  // the retired language models' (mlx-swift-lm, no longer in Voxt)
	GGUF Runtime = "gguf" // the retired translation model's (llama.cpp, no longer in Voxt)
)

// The catalogue's ids.
const (
	ASRAccurate = "asr.accurate"
	ASRFast     = "asr.fast"
	VADSilero   = "vad.silero"
	// Voxt 1.x's other local speech models.
	ASRQwen6bit     = "asr.qwen3-6bit"
	ASRWhisperTurbo = "asr.whisper-turbo"
	ASRWhisperLarge = "asr.whisper-large"
	ASRWhisperSmall = "asr.whisper-small"
	ASRParakeet     = "asr.parakeet"
	ASRSenseVoice   = "asr.sensevoice"
	ASRCohere       = "asr.cohere"
	ASRNemotron     = "asr.nemotron"
	ASRMoss         = "asr.moss"
	// The Whisper tokenizers, from OpenAI's repos (the MLX conversions
	// have none).
	AuxWhisperLarge = "aux.whisper-large-v3"
	AuxWhisperSmall = "aux.whisper-small"
	LLMBalanced     = "llm.balanced"
	LLMFast         = "llm.fast"
	LLMTranslate    = "llm.translate"
	// The family's language models VoxtNew can run when another ActNow
	// app installed them (Borrowed).
	LLMQwen3  = "llm.qwen3"
	LLMGemma4 = "llm.gemma4"
	// The speech models' DFlash drafts (Draft), one each.
	ASRAccurateDraft = "asr.accurate.draft"
	ASRFastDraft     = "asr.fast.draft"
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
	ID string
	// Name is what the UI calls it, as Voxt 1.x listed its models: the
	// family, its size and quantisation ("Qwen3-ASR 1.7B (8bit)"), so that
	// users can tell which model it is. Not translated.
	Name    string
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
	// Needs are models installed with it (a speech model: the VAD; a
	// Whisper model: its tokenizer, an Aux whose files its server reads
	// from the model's own folder).
	Needs []string
	// Draft is the speech model's speculative draft (a Draft model): a
	// small model that proposes several tokens a step for it to check, so
	// that it decodes faster. The user adds it from the
	// speech model's card; the runtime passes it while it is installed.
	// "" for none.
	Draft string
	// Alternatives are models that may stand in for it while it is not
	// installed (the same kind, lighter or another app's), offered to the
	// user, never chosen silently (family spec §7.2, §7.10.1).
	Alternatives []string
	// Borrowed: a model of another ActNow app that VoxtNew can run but
	// does not download: it is on offer only when it is on disk, and is
	// VoxtNew's once the user chose to use it (a claim).
	Borrowed bool
	// Retired: a model Voxt no longer offers (the local language models:
	// Voxt's are remote since 2.1). It is never downloaded or opened; the
	// Local models page shows it only while Voxt has it on this Mac, to
	// delete it.
	Retired bool
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

// All is every model in Catalog's order, without Catalog's copy: for
// read-only loops that run every frame.
func All() iter.Seq[Model] { return slices.Values(catalog) }

// Lookup is the model of id.
func Lookup(id string) (Model, bool) {
	for _, m := range catalog {
		if m.ID == id {
			return m, true
		}
	}
	return Model{}, false
}

// The pins are the family pins table's where it has the repo
// (TestFamilyPins); the omni models' are also sglang_omni_mlx/native/ci's
// models.json or model_pins.json, the Whisper tokenizers that file's
// (TestOmniPins). The speech models after the first two are Voxt 1.x's
// other local models, with the files their native servers read.
// The language models' files are those Voxt's loaders take (weights,
// configs, tokenizer, chat template): Qwen3.5-4B-OptiQ without its
// optiq/ vision tower and MTP head, Hy-MT2's one GGUF file of the repo.
// The borrowed models are Dicta's and AudioSmith's, whole snapshots.
var catalog = []Model{
	{
		ID: ASRAccurate, Name: "Qwen3-ASR 1.7B (8bit)", Kind: ASR, Runtime: Omni, Default: true, Needs: []string{VADSilero}, Draft: ASRAccurateDraft,
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
		ID: ASRFast, Name: "Qwen3-ASR 0.6B (4bit)", Kind: ASR, Runtime: Omni, Needs: []string{VADSilero}, Draft: ASRFastDraft,
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
		ID: ASRQwen6bit, Name: "Qwen3-ASR 1.7B (6bit)", Kind: ASR, Runtime: Omni, Needs: []string{VADSilero},
		Family: "qwen3-asr", Quant: "6bit",
		Repo: "mlx-community/Qwen3-ASR-1.7B-6bit", Revision: "edd077a475c4da058e25b6e6ec1199115ca1be2b",
		Files: []File{
			{"chat_template.json", 1161, "75a8cfca24f00de72d796fbfed6858fc9614ef3dabd8696684cc3bc03a9c58ff", false},
			{"config.json", 7188, "243c07b95f6ade8db20e73b3aa2b0ba101f8bcce75b82557f7d8e957838bcf25", false},
			{"generation_config.json", 142, "1da527824d81e07118facff437e03f2e24a23311e3bdeb2368973fe77e5f275c", false},
			{"merges.txt", 1671853, "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5", false},
			{"model.safetensors", 2033194557, "cacb094fef227ec2a5908d0136b3ad3384d95e0e2a8a915984ce577e4b12b2ba", true},
			{"model.safetensors.index.json", 78968, "703625934380698dbe06a62aa0e50167ec30fb4d9a383da171af6dd07c56e82d", false},
			{"preprocessor_config.json", 330, "45e120a4eda2c20c5d7f2ea9354e63536bf35e27aa573fb7cdf78017b378770d", false},
			{"tokenizer_config.json", 12487, "4942d005604266809309cabc9f4e9cb89ce855d59b14681fdc0e1cc62ea26c4c", false},
			{"vocab.json", 2776833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910", false},
		},
	},
	{
		ID: ASRWhisperTurbo, Name: "Whisper Large v3 Turbo", Kind: ASR, Runtime: Omni, Needs: []string{VADSilero, AuxWhisperLarge},
		Family: "whisper", Quant: "fp16",
		Repo: "mlx-community/whisper-large-v3-turbo", Revision: "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb",
		Files: []File{
			{"config.json", 268, "b34fc29e4e11e0a25e812775dd67f4dd16fc2c8eb43d28ae25ff7d660ecb6379", false},
			{"weights.safetensors", 1613977612, "951ed3fc1203e6a62467abb2144a96ce7eafca8fa77e3704fdb8635ff3e7f8a6", true},
		},
	},
	{
		ID: ASRWhisperLarge, Name: "Whisper Large v3", Kind: ASR, Runtime: Omni, Needs: []string{VADSilero, AuxWhisperLarge},
		Family: "whisper", Quant: "fp16",
		Repo: "mlx-community/whisper-large-v3-mlx", Revision: "49e6aa286ad60c14352c404340ded53710378a11",
		Files: []File{
			{"config.json", 269, "34982ce6ae286095000f82ae9583b3431639e8b092bf60c961f203745e6500e3", false},
			{"weights.npz", 3083520416, "05ff791ce3630fae47e7c51004e9666204d786246ec07cac6110af768099b40d", true},
		},
	},
	{
		ID: ASRWhisperSmall, Name: "Whisper Small", Kind: ASR, Runtime: Omni, Needs: []string{VADSilero, AuxWhisperSmall},
		Family: "whisper", Quant: "fp16",
		Repo: "mlx-community/whisper-small-mlx", Revision: "45f3915923c7a79a5a5b5a7d909d39aeb0e5630e",
		Files: []File{
			{"config.json", 266, "e8f58e638208af66d5d5d67801259dc7a12d199e971967a9f9d33a8e3635668e", false},
			{"weights.npz", 481307592, "55b6674c9b339702d486e2b1573839a66f8ec8f821ed2886993ef717a86b09f5", true},
		},
	},
	{
		ID: ASRParakeet, Name: "Parakeet TDT 0.6B v3", Kind: ASR, Runtime: Omni, Needs: []string{VADSilero},
		Family: "parakeet", Quant: "fp32",
		Repo: "mlx-community/parakeet-tdt-0.6b-v3", Revision: "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15",
		Files: []File{
			{"config.json", 244093, "f320f1292511f34ec47f513755fe20fd01dbfc09a925d42730e66059a6e1ef4c", false},
			{"model.safetensors", 2508288736, "05e01c7f396c298cf7d23f61da7b504adeab698f0aaeafd9c82d198625464592", true},
		},
	},
	{
		ID: ASRSenseVoice, Name: "SenseVoice Small", Kind: ASR, Runtime: Omni, Needs: []string{VADSilero},
		Family: "sensevoice", Quant: "fp32",
		Repo: "mlx-community/SenseVoiceSmall", Revision: "8ddd966bd96243cff196422f81f0c5d955814792",
		Files: []File{
			{"am.mvn", 11203, "29b3c740a2c0cfc6b308126d31d7f265fa2be74f3bb095cd2f143ea970896ae5", false},
			{"chn_jpn_yue_eng_ko_spectok.bpe.model", 377341, "aa87f86064c3730d799ddf7af3c04659151102cba548bce325cf06ba4da4e6a8", true},
			{"config.json", 517, "32b996111892031dfe4b73751864d93c769391aa213427ea65daf2ecdef7caa5", false},
			{"model.safetensors", 936100124, "7abe356045684ae374ea60e6b8dbc967222d7d1e35cc222be870c94630b1c38a", true},
		},
	},
	{
		ID: ASRCohere, Name: "Cohere Transcribe 03-2026 (fp16)", Kind: ASR, Runtime: Omni, Needs: []string{VADSilero},
		Family: "cohere-transcribe", Quant: "fp16",
		Repo: "beshkenadze/cohere-transcribe-03-2026-mlx-fp16", Revision: "06bbe0d853a93cbbf7c1f40cffccd4c5481459ef",
		Files: []File{
			{"config.json", 3998, "5de7e586cec6d8f51225c8d5fe17a56a3043dda9af8c42f9cb01dd545905eb18", false},
			{"model.safetensors", 4131827448, "1ec6ba9ee27da02b21b3ffdb5183b77020351d3331d05a74ad8d58a09394a2b8", true},
			{"tokenizer.model", 492827, "6d21e6a83b2d0d3e1241a7817e4bef8eb63bcb7cfe4a2675af9a35ff3bbf0e14", true},
			{"tokenizer_config.json", 48141, "0dfeb3eeba07bccaa1b4bf78f3135ad3059acf8d18f681675832b285ac0035b0", false},
		},
	},
	{
		ID: ASRNemotron, Name: "Nemotron 3.5 ASR 0.6B (8bit)", Kind: ASR, Runtime: Omni, Needs: []string{VADSilero},
		Family: "nemotron-asr", Quant: "8bit",
		Repo: "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit", Revision: "7279359e4481b5e9e185a318bd618e429c6d86cd",
		Files: []File{
			{"config.json", 159605, "f30c7bc469fc01fd5483172b4d7c75075030ddb60f347589c44e216b0a5ea9b6", false},
			{"model.safetensors", 755598923, "a64a4da048e7d28dde4cd4ff61ce59308a63314bb5563e73e06c24aae50ea941", true},
		},
	},
	{
		ID: ASRMoss, Name: "MOSS-Transcribe-Diarize", Kind: ASR, Runtime: Omni, Needs: []string{VADSilero},
		Family: "moss-transcribe-diarize", Quant: "bf16",
		Repo: "OpenMOSS-Team/MOSS-Transcribe-Diarize", Revision: "704aa4a9c304e8520be88901e0d1960158ef5b15",
		Files: []File{
			{"config.json", 2335, "2b2b7a6e61334152bdd7ecf8a4da3073b4940a097e193d1d2b22093e77535234", false},
			{"merges.txt", 1671853, "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5", false},
			{"model-00000-of-00001.safetensors", 1817113576, "9a0ceb4ab7330357db3ff583dba8d83625d5b733b00e1d55d6970e11b07026c4", true},
			{"processor_config.json", 292, "a978c2dd54a65b576c3dae4b654fe9bcbac1184c6db2df0afb2c90fcdc872ae7", false},
			{"tokenizer.json", 11423222, "bcf03774334462d6e34b5005cb11120a62275f146ee2953e68731ecdbce84fbb", true},
			{"tokenizer_config.json", 503, "61d04c96104177240688396655ae3f7cf38ce2ea036db867a1d2b6883e27c3d5", false},
			{"vocab.json", 2776833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910", false},
		},
	},
	{
		ID: AuxWhisperLarge, Name: "Whisper Large v3 tokenizer", Kind: Aux, Runtime: Omni, Family: "whisper",
		Repo: "openai/whisper-large-v3", Revision: "06f233fe06e710322aca913c1bc4249a0d71fce1",
		Files: []File{
			{"generation_config.json", 3903, "fbdfa70135de9b1d31553393f14e80aaeb1936ea36576b2ba864055943c09d23", false},
			{"merges.txt", 493869, "2df2990a395e35e8dfbc7511e08c12d56018d8d04691e0133e5d63b21e154dc6", false},
			{"tokenizer_config.json", 282843, "844b642c73a91359722f47b35705f7174686df33d252695d8572cf9ac03a6389", false},
			{"vocab.json", 1036558, "e2aa043ef015641d363d8288e7c241c85e36a5c761fb303598e0710233344387", false},
		},
	},
	{
		ID: AuxWhisperSmall, Name: "Whisper Small tokenizer", Kind: Aux, Runtime: Omni, Family: "whisper",
		Repo: "openai/whisper-small", Revision: "973afd24965f72e36ca33b3055d56a652f456b4d",
		Files: []File{
			{"generation_config.json", 3868, "71565b8ef50d0bf7a1193ed4bbed195b94e70c18894d81bba2f1233dcec3ab53", false},
			{"merges.txt", 493869, "2df2990a395e35e8dfbc7511e08c12d56018d8d04691e0133e5d63b21e154dc6", false},
			{"tokenizer_config.json", 282683, "2a4c4281cf9f51ac6ccc406fdc711a087afe6530f671fa7b80953edc498275ce", false},
			{"vocab.json", 835550, "8f680bba319e01a653d2e8a5dbc17a9157179e0576e6ce74ce0c06356c6e24f9", false},
		},
	},
	{
		ID: VADSilero, Name: "Silero VAD v6", Kind: VAD, Runtime: Omni, Default: true, Family: "silero-vad",
		Repo: "mlx-community/silero-vad-v6", Revision: "2ebf4a5e10726a2e78ddd4d70eedfb6f1c33eb06",
		Files: []File{
			{"config.json", 463, "9fe1befb9692a0d4135adadc33f8075ef6d350bd2391b88d750f2c233f97fa0b", false},
			{"model.safetensors", 1237860, "65b6c5f0293cbc44d109e58bef78b474d9c65dedbee814cf0b90ef5f0d9150ff", true},
		},
	},
	{
		ID: ASRAccurateDraft, Name: "Qwen3-ASR 1.7B DFlash draft", Kind: Draft, Runtime: Omni, Family: "qwen3-asr-dflash", Quant: "bf16",
		Repo: "wilsonzheng0327/Qwen3-ASR-1.7B-dflash", Revision: "23a0db8ae3f0724ccb7227c969a57651ddade2e0",
		Files: []File{
			{"config.json", 1858, "890b34fd509f74fa8aeceaaea921d5d9a7110f177b75185aa54d4edb9f6af4e1", false},
			{"model.safetensors", 725684336, "00c32b078e3613073c0437e69519eb7811cbf97fcbd0066d05d62be7273670b8", true},
		},
	},
	{
		ID: ASRFastDraft, Name: "Qwen3-ASR 0.6B DFlash draft", Kind: Draft, Runtime: Omni, Family: "qwen3-asr-dflash", Quant: "bf16",
		Repo: "wilsonzheng0327/Qwen3-ASR-0.6B-dflash", Revision: "75da55f3170c28ce70379b96921c1cf9a331d008",
		Files: []File{
			{"config.json", 1858, "de349d2f8494363a3a3641ca6554a3841e001f769de089974a5349c9580931da", false},
			{"model.safetensors", 262451248, "bbfc427b9e57cb3663d23ab796d05ca82070bbdfe4c42dc9a5813430a0ace219", true},
		},
	},
	{
		ID: LLMBalanced, Name: "Qwen3.5 4B OptiQ (4bit)", Kind: LLM, Retired: true, Runtime: MLX, Default: true,
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
		ID: LLMFast, Name: "Qwen3.5 2B (4bit)", Kind: LLM, Retired: true, Runtime: MLX,
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
		ID: LLMTranslate, Name: "Hy-MT2 1.8B (Q4_K_M)", Kind: LLM, Retired: true, Runtime: GGUF, Features: []string{"translate"},
		Family: "hy-mt2", Quant: "Q4_K_M",
		Repo: "tencent/Hy-MT2-1.8B-GGUF", Revision: "a0c709d9fac510f2c807aa3af52872340dc37a4a",
		Files: []File{
			{"Hy-MT2-1.8B-Q4_K_M.gguf", 1133080448, "dc5f44fcf1fa496ee7ad725982c0c8c553a4de00259b53af84c4b89fb0c06699", true},
		},
	},
	{
		ID: LLMQwen3, Name: "Qwen3 4B Instruct 2507 (4bit)", Kind: LLM, Retired: true, Runtime: MLX, Borrowed: true,
		Family: "qwen3", Quant: "4bit", Bytes: 2278969697,
		Repo: "mlx-community/Qwen3-4B-Instruct-2507-4bit", Revision: "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b",
	},
	{
		ID: LLMGemma4, Name: "Gemma 4 E4B (4bit)", Kind: LLM, Retired: true, Runtime: MLX, Borrowed: true,
		Family: "gemma4", Quant: "4bit", Bytes: 5179239349,
		Repo: "mlx-community/gemma-4-e4b-it-4bit", Revision: "475b9088d29754a3379866cf5aeb6b41acd313c2",
	},
}
