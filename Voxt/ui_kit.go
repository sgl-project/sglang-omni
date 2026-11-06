package main

// The shared UI kit of the actnow family, as AudioSmith 0.3 made it (its
// research/ui-design-spec.md §3–§4; docs/architecture.md §3): buttons by rank, the "⋯" overflow menu,
// text marks and chips, the page header, flat cards, collapsible
// sections, form rows, search and filter bars, empty states, notes on
// the page, and side panels that become drawers when the page is
// narrow. Pages build from these rather than styling MyGo's widgets one
// by one.
//
// Ranks: one primaryButton per view (the core action, in ink); ui.Button
// for the other actions that need a label (secondary: grey face); ghost
// buttons for quiet actions in toolbars and lists; iconButton, with its
// tooltip, wherever an icon says enough; and the rest in moreMenu. The
// round ink playButton plays a recording; playToggle is the small play
// icon of a row in a list.
//
// Notes: banner is a line of state on the subtle fill with any icon;
// notice is a banner by kind (info, done, warning, error) with an ⓘ for
// more and an × to dismiss it.
//
// Files of the kit:
//
//	ui_kit.go      this: the overview
//	ui_tokens.go   the design tokens (tok)
//	ui_buttons.go  buttons by rank, icon buttons and toggles, "⋯", play buttons
//	ui_marks.go    text, marks and chips; notes on the page (banner, notice)
//	ui_layout.go   the page header, cards, collapsible, form rows, search and filter bars, empty states
//	ui_panels.go   side panels (inline or a drawer) and their state
//	ui_forms.go    segmented controls, the number input, popovers, the dialog frame
