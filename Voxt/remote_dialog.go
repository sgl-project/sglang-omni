package main

// The remote service dialog's view. Its controller owns discovery,
// validation, credentials and persistence.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

func remoteProviderDialogView(c *ui.Context, state *remoteDialogViewState, actions remoteDialogActions, local *providerDialogLocalUI) {
	if state == nil || !state.Open {
		return
	}
	t := c.Theme()
	if dialogFrame(c, "remote-dialog", &local.open, state.Name, state.Subtitle, 560, func() {
		ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
			caption(c, state.What).Grow(1).Basis(px(t, 240))
			if state.HasKeyURL && remoteLink(c, "remote-keyurl", i18n.T("Get a key")) {
				actions.openRemoteKeyURL()
			}
		})
		if state.ShowAppID {
			fieldInfo(c, i18n.T("App ID"), i18n.T("From your app in the Volcengine speech console. Use its Access Token as the key below. Keys from the newer console need no App ID."), func() {
				ui.TextInput(c.Key("remote-appid"), &local.appID).Placeholder(i18n.T("Optional")).Label(i18n.T("App ID"))
			})
		}
		if state.ChatGPT != nil {
			remoteChatGPTFields(c, state.ChatGPT, actions)
		}
		if state.Key != nil {
			fieldInfo(c, i18n.T("API key"), state.Key.Info, func() {
				ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
					input := ui.TextInput(c.Key("remote-key"), &local.key).Placeholder(state.Key.Placeholder).Label(i18n.T("API key")).Grow(1).MinWidth(0)
					if !local.showKey {
						input.Password()
					}
					if local.key != "" {
						show := i18n.T("Show")
						if local.showKey {
							show = i18n.T("Hide")
						}
						if ghostButton(c.Key("remote-key-show"), show).Label(i18n.T("Show or hide the key")).Clicked() {
							local.showKey = !local.showKey
						}
					}
					if state.Key.HasSaved && !local.removeKey && ghostButton(c.Key("remote-key-remove"), i18n.T("Remove")).Label(i18n.T("Remove saved key")).Clicked() {
						local.removeKey, local.key = true, ""
					}
				})
				if state.Key.Warning != "" {
					caption(c, state.Key.Warning).TextColor(tok(t).Warning)
				}
			})
		}
		if state.Endpoint != nil {
			fieldInfo(c, i18n.T("Address"), state.Endpoint.Info, func() {
				ui.TextInput(c.Key("remote-endpoint"), &local.endpoint).Placeholder(state.Endpoint.Placeholder).Label(i18n.T("Address"))
			})
		}
		fieldInfo(c, i18n.T("Model"), "", func() {
			ui.Select(c.Key("remote-model"), &local.model, state.ModelOptions).Label(i18n.T("Model")).Width(px(t, 320))
			if local.model == state.OtherModel {
				ui.TextInput(c.Key("remote-model-custom"), &local.customModel).Placeholder(i18n.T("Model ID")).Label(i18n.T("Model ID"))
			}
		})
		if state.RecognitionPrompt {
			fieldInfo(c, i18n.T("Recognition prompt"), i18n.T("Optional. Added after the speech context on the Speech models page. You can use {{USER_MAIN_LANGUAGE}}, {{USER_OTHER_LANGUAGES}} and {{DICTIONARY_TERMS}}."), func() {
				ui.TextArea(c.Key("remote-asr-prompt"), &local.recognitionPrompt).Label(i18n.T("Recognition prompt")).
					Placeholder(i18n.T("For example: Technical talk; prefer product names from the dictionary.")).MinHeight(px(t, 72))
			})
		}
		if state.Generation != nil {
			remoteGenerationFields(c, state.Generation, &local.gen)
		}
		if local.err != "" {
			notice(c, "remote-dlg-err", noticeError, local.err, local.errMore, nil, func() { local.err, local.errMore = "", "" })
		}
	}, func() {
		if ui.Button(c.Key("remote-dlg-test"), i18n.T("Test connection")).Disabled(!state.CanTest).Shrink(0).Clicked() {
			actions.testDialog()
		}
		ui.Row(c.Key("remote-dlg-status")).Gap(px(t, sp1)).AlignItems(ui.Center).Grow(1).Shrink(1).Basis(0).MinWidth(0).Children(func() {
			remoteTestStatus(c, state.Test, true)
		})
		if state.ShowRemove && ghostButton(c.Key("remote-dlg-remove"), i18n.T("Remove")).Label(i18n.T("Remove %s", state.Name)).Disabled(state.Saving).Shrink(0).Clicked() {
			actions.removeProvider()
		}
		if ghostButton(c.Key("remote-dlg-cancel"), i18n.T("Cancel")).Shrink(0).Clicked() {
			local.open = false
		}
		if primaryButton(c.Key("remote-dlg-save"), i18n.T("Save")).Disabled(state.Saving).Shrink(0).Clicked() {
			actions.saveProviderDialog()
		}
	}).Dismissed() && actions.remoteDialogUnsaved() {
		local.open = true
	}
}

// remoteLink matches MyGo's link presentation while routing the intent
// through the area's actions instead of performing I/O in the view.
func remoteLink(c *ui.Context, key, label string) bool {
	element := ui.Text(c.Key(key), label).FontSize(px(c.Theme(), fsCaption)).TextColor(c.Theme().Accent).Cursor(ui.CursorPointer).Focusable().Role(ui.RoleLink)
	if element.Hovered() {
		element.Underline()
	}
	return element.Clicked()
}
