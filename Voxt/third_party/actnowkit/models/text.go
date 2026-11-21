package models

import "github.com/hehehai/actnowkit/family"

// Reference texts of the model library (spec §7.10). Apps translate them
// with their own i18n; placeholders are filled with Text.Fill.
var (
	// TextReuseTitle / TextReuseBy: the first-run reuse card (§7.10.1).
	TextReuseTitle  = family.Text{EN: "{count} models are already on this Mac", ZH: "已在这台 Mac 上找到 {count} 个模型"}
	TextReuseBy     = family.Text{EN: "Installed by {app} · {size} · no download needed", ZH: "由 {app} 安装 · {size} · 无需下载"}
	TextUseThem     = family.Text{EN: "Use them", ZH: "直接使用"}
	TextChooseModel = family.Text{EN: "Choose models…", ZH: "选择模型…"}
	TextStillToGet  = family.Text{EN: "{size} still to download", ZH: "还需下载 {size}"}
	TextInstalledBy = family.Text{EN: "Installed · from {app}", ZH: "已安装 · 来自 {app}"}
	// TextAlternative: the recommended model is missing, a lighter one of
	// the same series is there (§7.10.1). Default button: download.
	TextAlternative = family.Text{
		EN: "{want} isn't installed, but {have} (from {app}) is. Use it for now?",
		ZH: "未安装 {want}，但已有 {have}（来自 {app}）。先用它吗？",
	}
	TextUseAlternative = family.Text{EN: "Use {have}", ZH: "先用 {have}"}
	TextDownloadWanted = family.Text{EN: "Download {want}", ZH: "下载 {want}"}

	// Models and storage pages (§7.10.2).
	TextSharedWith   = family.Text{EN: "Shared with {apps}", ZH: "与 {apps} 共用"}
	TextFolderShared = family.Text{EN: "Shared by all ActNow apps", ZH: "由所有 ActNow app 共用"}
	TextOnlyThisApp  = family.Text{EN: "Only {app}", ZH: "仅 {app}"}
	TextSharedOthers = family.Text{EN: "Shared with other ActNow apps", ZH: "与其他 ActNow app 共享"}
	TextStillUsedBy  = family.Text{EN: "Still used by {app}, kept", ZH: "仍被 {app} 使用，已保留"}
	TextOrphans      = family.Text{
		EN: "{count} models ({size}) are only used by apps that are no longer installed.",
		ZH: "有 {count} 个模型（{size}）只被已卸载的 app 使用。",
	}
	TextCleanUp = family.Text{EN: "Clean up", ZH: "清理"}

	// The delete dialog when another app uses the model (§7.10.3).
	TextRemoveTitle = family.Text{EN: "Remove \"{model}\" from {app}?", ZH: "从 {app} 移除“{model}”？"}
	TextRemoveBody  = family.Text{
		EN: "{apps} also uses this model, so its files ({size}) stay on this Mac.",
		ZH: "{apps} 也在使用这个模型，文件（{size}）会保留在这台 Mac 上。",
	}
	TextRemoveFromApp   = family.Text{EN: "Remove from {app}", ZH: "从 {app} 移除"}
	TextDeleteForAll    = family.Text{EN: "Delete for all apps…", ZH: "为所有 app 删除…"}
	TextDeleteForAllMsg = family.Text{
		EN: "{apps} will need to download it again ({size}).",
		ZH: "{apps} 再次使用时需要重新下载（{size}）。",
	}
	TextDelete = family.Text{EN: "Delete", ZH: "删除"}
	TextCancel = family.Text{EN: "Cancel", ZH: "取消"}
	// TextWipeKeeps: "erase all data" with models, when some are shared (§7.6 step 3).
	TextWipeKeeps = family.Text{
		EN: "These models are also used by other ActNow apps and will be kept: {models} ({size})",
		ZH: "以下模型也被其他 ActNow app 使用，将被保留：{models}（{size}）",
	}
	// TextDeletedIn: after another app deleted a model for all apps (§7.6).
	TextDeletedIn = family.Text{EN: "Deleted in {app} · Download again", ZH: "已在 {app} 中删除 · 重新下载"}
	// TextBusy: the repo lock is held (§7.4).
	TextBusy = family.Text{
		EN: "{app} is downloading or using this model. Try again later.",
		ZH: "{app} 正在下载或使用这个模型，请稍后再试。",
	}
	// TextDownloadingIn: another app downloads the model (§7.10.4).
	TextDownloadingIn = family.Text{EN: "Downloading in {app}… {percent}%", ZH: "正在由 {app} 下载… {percent}%"}
	// TextLicenceAcceptedIn: a licence accepted in another app (§7.9).
	TextLicenceAcceptedIn = family.Text{
		EN: "You accepted this licence in {app} on {date}.",
		ZH: "你已在 {app} 中接受了此许可（{date}）。",
	}
	TextContinue = family.Text{EN: "Continue", ZH: "继续"}
	// TextMoveModels: the family keeps models elsewhere (§15.2 step 5).
	TextMoveModels = family.Text{
		EN: "Other ActNow apps keep models in {path}. Move this app's models there to share them?",
		ZH: "其他 ActNow app 把模型放在 {path}。要把本 app 的模型移过去以便共用吗？",
	}
	TextMoveButton   = family.Text{EN: "Move models…", ZH: "移动模型…"}
	TextKeepSeparate = family.Text{EN: "Keep separate", ZH: "保持分开"}

	// v1.0.9: the library in $FAMILY/models/hub, Time Machine, size check.

	// TextBackupModels / TextBackupModelsHint: the storage page's switch
	// for models.backup (§7.12), off by default.
	TextBackupModels     = family.Text{EN: "Back up models with Time Machine", ZH: "用 Time Machine 备份模型"}
	TextBackupModelsHint = family.Text{
		EN: "Models can be downloaded again, so they are left out of backups. Applies to all ActNow apps.",
		ZH: "模型可以重新下载，所以默认不放进备份。对所有 ActNow app 生效。",
	}
	// TextImported: once, after Import found models (§7.1.1).
	TextImported = family.Text{
		EN: "Found {count} models downloaded earlier ({size}). They are now in the ActNow model folder; nothing was downloaded again.",
		ZH: "找到之前下载的 {count} 个模型（{size}），已放进 ActNow 模型文件夹，无需重新下载。",
	}
	// TextFolderSource: the model folder row's second line (§7.10.2):
	// where the path comes from.
	TextFolderDefault = family.Text{EN: "Default · Shared by all ActNow apps", ZH: "默认位置 · 由所有 ActNow app 共用"}
	TextFolderChosen  = family.Text{EN: "Chosen in {app} · Shared by all ActNow apps", ZH: "在 {app} 中设置 · 由所有 ActNow app 共用"}
	// TextDamaged: Check found files of the wrong size (§7.5).
	TextDamaged = family.Text{
		EN: "Some files are incomplete. Download again to repair (only the damaged files are fetched).",
		ZH: "部分文件不完整。重新下载即可修复（只下载损坏的文件）。",
	}
	// TextUnclaimed: Decide is unsure because an app's claims are missing
	// (§7.6, v1.0.9); the files stay.
	TextUnclaimed = family.Text{
		EN: "{apps} may still use this model, so its files stay. Open {apps} once and try again.",
		ZH: "{apps} 可能还在使用这个模型，文件已保留。打开一次 {apps} 后再试。",
	}
)
