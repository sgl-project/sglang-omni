package main

// The ChatGPT account controls in the remote service dialog.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

func remoteChatGPTFields(c *ui.Context, state *remoteChatGPTViewState, actions remoteDialogActions) {
	t := c.Theme()
	info := i18n.Platform("The sign-in is kept only in this Mac's Keychain. Signing out deletes it.", "The sign-in is kept only in this PC's Credential Manager. Signing out deletes it.")
	fieldInfo(c, i18n.T("ChatGPT account"), info, func() {
		ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
			switch {
			case state.Running:
				ui.Spinner(c).Size(px(t, 14), px(t, 14)).Shrink(0)
				caption(c, i18n.T("Waiting for the browser…")).Grow(1).MinWidth(0)
				if ghostButton(c.Key("chatgpt-cancel"), i18n.T("Cancel")).Label(i18n.T("Cancel sign-in")).Clicked() {
					actions.cancelChatGPTLogin()
				}
			case state.SignedIn:
				caption(c, i18n.T("Signed in as %s", state.Email)).Grow(1).MinWidth(0)
				if remoteLink(c, "chatgpt-usage", i18n.T("Manage usage")) {
					actions.openChatGPTUsage()
				}
				if ghostButton(c.Key("chatgpt-signout"), i18n.T("Sign out")).Label(i18n.T("Sign out of ChatGPT")).Clicked() {
					actions.signOutChatGPT()
				}
			default:
				ui.Spacer(c)
				if ui.Button(c.Key("chatgpt-signin"), i18n.T("Sign in with ChatGPT")).Clicked() {
					actions.signInChatGPT()
				}
			}
		})
		if state.Error != "" {
			notice(c, "chatgpt-err", noticeError, state.Error, state.ErrorMore, nil, actions.dismissChatGPTError)
		}
	})
}
