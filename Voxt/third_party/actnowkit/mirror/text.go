package mirror

import "github.com/hehehai/actnowkit/family"

// Reference wordings of the download source setting (network.hfEndpoints,
// spec §9.6), which covers models (Hugging Face) and tools (GitHub), for
// the apps' catalogues. Apps whose downloads are only models keep naming
// Hugging Face hosts only.
var (
	// TextSourceAuto: the automatic choice (the default).
	TextSourceAuto = family.Text{EN: "Automatic (the fastest of the official sites and mirrors)", ZH: "自动（官方站点和镜像中最快的）"}
	// TextSourceOfficial: "official".
	TextSourceOfficial = family.Text{EN: "Official sites only (huggingface.co, github.com)", ZH: "仅官方站点（huggingface.co、github.com）"}
	// TextSourceMirror: "mirror".
	TextSourceMirror = family.Text{EN: "Mirrors only (hf-mirror.com, mirror.actnow.dev)", ZH: "仅镜像（hf-mirror.com、mirror.actnow.dev）"}
	// TextSourceNote: under the setting.
	TextSourceNote = family.Text{
		EN: "Models come from Hugging Face, ffmpeg and yt-dlp from GitHub. Every download is checked (signature or checksum), so a mirror cannot change what you get.",
		ZH: "模型来自 Hugging Face，ffmpeg 和 yt-dlp 来自 GitHub。所有下载都会校验（签名或校验和），镜像无法改动你拿到的文件。",
	}
)
