package main

// Thinking-menu rows drawn from the cached JSON-safe options.

import "github.com/egoist/mygo/ui"

func thinkMenuView(context *ui.Context, options []sessionMenuOption, actions sessionOverlayActions) {
	for index, option := range options {
		if menuRow(context.Key(index), option.Label, option.Selected) {
			actions.chooseThinking(option.Thinking)
		}
	}
}
