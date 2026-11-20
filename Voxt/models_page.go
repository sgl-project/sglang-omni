package main

// The Speech models section in Settings (/settings/models; /models before):
// the speech models of this Mac, remote speech services, retired local
// language models, shared-family status, downloads and deletion. The
// section's view is modelsSettingsSection (shell_settings.go) over
// models_view_state.go; its dialogs show over Settings (shell_controller.go).
// Controllers own the model library and runtime work (models_controller.go).

// modelsPath is the Speech models section's path in Settings.
const modelsPath = "/settings/models"
