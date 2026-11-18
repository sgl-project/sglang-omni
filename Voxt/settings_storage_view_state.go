package main

import (
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

type settingsStorageCategoryViewState struct {
	ID        string `json:"id"`
	Title     string `json:"title"`
	Size      string `json:"size"`
	Directory string `json:"directory"`
}

type settingsStorageReportViewState struct {
	DataSize   string                             `json:"data_size"`
	FreeSize   string                             `json:"free_size"`
	Categories []settingsStorageCategoryViewState `json:"categories"`
}

type settingsStorageUsageViewState struct {
	Title string `json:"title"`
	Size  string `json:"size"`
}

type settingsStorageViewState struct {
	DataDirectory        string                          `json:"data_directory"`
	DataPath             string                          `json:"data_path"`
	RecordingsPath       string                          `json:"recordings_path"`
	PreviousPath         string                          `json:"previous_path"`
	FailedMove           string                          `json:"failed_move"`
	CanOpen              bool                            `json:"can_open"`
	Report               *settingsStorageReportViewState `json:"report"`
	MoveError            string                          `json:"move_error"`
	Moving               bool                            `json:"moving"`
	MovePath             string                          `json:"move_path"`
	MoveProgress         float64                         `json:"move_progress"`
	MoveProgressText     string                          `json:"move_progress_text"`
	CanCancelMove        bool                            `json:"can_cancel_move"`
	ResetAsk             string                          `json:"reset_ask"`
	ResetText            string                          `json:"reset_text"`
	Erasing              bool                            `json:"erasing"`
	KeptModelsText       string                          `json:"kept_models_text"`
	ModelDirectory       string                          `json:"model_directory"`
	ModelPath            string                          `json:"model_path"`
	ModelNote            string                          `json:"model_note"`
	CanChangeModelFolder bool                            `json:"can_change_model_folder"`
	ShowModelBackup      bool                            `json:"show_model_backup"`
	ModelBackup          bool                            `json:"model_backup"`
	ModelBackupHint      string                          `json:"model_backup_hint"`
	ModelUsage           []settingsStorageUsageViewState `json:"model_usage"`
	ModelMoving          bool                            `json:"model_moving"`
	ModelMoveText        string                          `json:"model_move_text"`
	CanCancelModelMove   bool                            `json:"can_cancel_model_move"`
}

type settingsStorageStateInput struct {
	HomeDir              string
	DataDirectory        string
	RecordingsDirectory  string
	PreviousDirectory    string
	FailedMove           string
	CanOpen              bool
	Report               *storageReport
	MoveError            string
	Moving               bool
	MoveDone             int64
	MoveTotal            int64
	MoveTo               string
	CanCancelMove        bool
	ResetAsk             string
	ResetText            string
	Erasing              bool
	KeptModelsText       string
	ModelDirectory       string
	FamilyOthers         bool
	CanChangeModelFolder bool
	ModelBackup          bool
	ModelMoving          bool
	ModelMoveTo          string
	CanCancelModelMove   bool
}

func buildSettingsStorageViewState(translator *i18n.Translator, input settingsStorageStateInput) settingsStorageViewState {
	state := settingsStorageViewState{
		DataDirectory: input.DataDirectory, DataPath: uitext.ShortPathAt(input.DataDirectory, input.HomeDir),
		PreviousPath: uitext.ShortPathAt(input.PreviousDirectory, input.HomeDir),
		FailedMove:   input.FailedMove, CanOpen: input.CanOpen, MoveError: input.MoveError,
		Moving: input.Moving, MovePath: uitext.ShortPathAt(input.MoveTo, input.HomeDir),
		MoveProgressText: uitext.HumanBytes(input.MoveDone) + " / " + uitext.HumanBytes(input.MoveTotal),
		CanCancelMove:    input.CanCancelMove, ResetAsk: input.ResetAsk, ResetText: input.ResetText,
		Erasing: input.Erasing, KeptModelsText: input.KeptModelsText,
		ModelDirectory: input.ModelDirectory, ModelPath: uitext.ShortPathAt(input.ModelDirectory, input.HomeDir),
		CanChangeModelFolder: input.CanChangeModelFolder, ShowModelBackup: input.CanChangeModelFolder,
		ModelBackup: input.ModelBackup, ModelMoving: input.ModelMoving, CanCancelModelMove: input.CanCancelModelMove,
	}
	if input.RecordingsDirectory != "" {
		state.RecordingsPath = uitext.ShortPathAt(input.RecordingsDirectory, input.HomeDir)
	}
	if input.MoveTotal > 0 {
		state.MoveProgress = float64(input.MoveDone) / float64(input.MoveTotal)
	}
	if input.ModelMoving {
		state.ModelMoveText = translator.T("Moving the models to %s, then Voxt restarts. The old folder is left as it is.", uitext.ShortPathAt(input.ModelMoveTo, input.HomeDir))
		if input.FamilyOthers {
			state.ModelMoveText = translator.T("Moving every app's models to %s, then Voxt restarts. The old folders are left as they are.", uitext.ShortPathAt(input.ModelMoveTo, input.HomeDir))
		}
	}
	state.ModelBackupHint = translator.T("Models can be downloaded again, so Time Machine leaves them out.")
	if input.FamilyOthers {
		state.ModelBackupHint = translator.T("Models can be downloaded again, so Time Machine leaves them out. Applies to all apps that use this model library.")
	}
	if input.Report == nil {
		return state
	}
	report := &settingsStorageReportViewState{
		DataSize: uitext.HumanBytes(input.Report.data), FreeSize: uitext.HumanBytes(input.Report.free),
	}
	for _, category := range input.Report.cats {
		report.Categories = append(report.Categories, settingsStorageCategoryViewState{
			ID: category.id, Title: category.title, Size: uitext.HumanBytes(category.bytes), Directory: category.dir,
		})
	}
	state.Report = report
	if input.FamilyOthers && input.Report.shared {
		state.ModelNote = translator.T("Shared by all apps that use this model library")
	}
	state.ModelUsage = []settingsStorageUsageViewState{{Title: translator.T("Used by Voxt"), Size: uitext.HumanBytes(input.Report.modelsMine)}}
	if input.FamilyOthers && input.Report.shared || input.Report.modelsShared > 0 {
		state.ModelUsage = []settingsStorageUsageViewState{
			{Title: translator.T("Only Voxt"), Size: uitext.HumanBytes(input.Report.modelsMine)},
			{Title: translator.T("Shared with other apps that use this model library"), Size: uitext.HumanBytes(input.Report.modelsShared)},
		}
	}
	return state
}
