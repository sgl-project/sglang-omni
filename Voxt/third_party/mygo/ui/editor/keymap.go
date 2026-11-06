package editor

import "github.com/egoist/mygo/ui"

// Command is something a code editor does, which a key binding runs:
// st.Run(ui.ToggleComment).
type Command uint16

// Commands of code editors, as Zed's editor actions.
const (
	NoCommand Command = iota
	// Moving the carets, and with Select, extending the selections.
	MoveLeft
	MoveRight
	MoveUp
	MoveDown
	MoveWordLeft
	MoveWordRight
	MoveSubwordLeft
	MoveSubwordRight
	MoveLineStart
	MoveLineEnd
	MoveDocStart
	MoveDocEnd
	MoveParagraphUp
	MoveParagraphDown
	MovePageUp
	MovePageDown
	MoveToBracket
	SelectLeft
	SelectRight
	SelectUp
	SelectDown
	SelectWordLeft
	SelectWordRight
	SelectSubwordLeft
	SelectSubwordRight
	SelectLineStart
	SelectLineEnd
	SelectDocStart
	SelectDocEnd
	SelectParagraphUp
	SelectParagraphDown
	SelectPageUp
	SelectPageDown
	SelectToBracket
	// Selections.
	SelectAll
	SelectLine
	SelectNext
	SelectPrevious
	SelectAllMatches
	AddCursorAbove
	AddCursorBelow
	SelectLarger
	SelectSmaller
	UndoSelection
	RedoSelection
	SplitSelectionIntoLines
	Cancel
	// Editing.
	Undo
	Redo
	Cut
	Copy
	Paste
	Backspace
	Delete
	DeleteWordLeft
	DeleteWordRight
	DeleteSubwordLeft
	DeleteSubwordRight
	DeleteToLineStart
	DeleteToLineEnd
	CutToLineEnd
	DeleteLine
	Newline
	NewlineAbove
	NewlineBelow
	MoveLinesUp
	MoveLinesDown
	DuplicateLinesUp
	DuplicateLinesDown
	DuplicateSelection
	JoinLines
	Transpose
	Indent
	Outdent
	Tab
	Backtab
	ToggleComment
	ToggleBlockComment
	SortLines
	SortLinesCaseInsensitive
	ReverseLines
	UniqueLines
	Uppercase
	Lowercase
	TitleCase
	SnakeCase
	KebabCase
	CamelCase
	PascalCase
	TrimTrailingWhitespace
	// Folding.
	Fold
	Unfold
	ToggleFold
	FoldAll
	UnfoldAll
	FoldRecursive
	UnfoldRecursive
	FoldSelection
	FoldLevel1
	FoldLevel2
	FoldLevel3
	FoldLevel4
	FoldLevel5
	FoldLevel6
	FoldLevel7
	FoldLevel8
	FoldLevel9
	// Finding.
	Find
	FindReplace
	FindNext
	FindPrevious
	UseSelectionForFind
	FindInSelection
	// Going places.
	GoToLine
	ScrollCursorCenter
	ScrollLineUp
	ScrollLineDown
	GoToDefinition
	ShowCompletions
	ShowHover
	// Changes against the DiffBase, or the hunks of a diff.
	NextHunk
	PreviousHunk
	RevertHunk
	// The view.
	ToggleSoftWrap
	ToggleLineNumbers
	commandCount
)

var commandNames = [...]string{
	"none", "moveLeft", "moveRight", "moveUp", "moveDown", "moveWordLeft", "moveWordRight", "moveSubwordLeft", "moveSubwordRight",
	"moveLineStart", "moveLineEnd", "moveDocStart", "moveDocEnd", "moveParagraphUp", "moveParagraphDown", "movePageUp", "movePageDown", "moveToBracket",
	"selectLeft", "selectRight", "selectUp", "selectDown", "selectWordLeft", "selectWordRight", "selectSubwordLeft", "selectSubwordRight",
	"selectLineStart", "selectLineEnd", "selectDocStart", "selectDocEnd", "selectParagraphUp", "selectParagraphDown", "selectPageUp", "selectPageDown", "selectToBracket",
	"selectAll", "selectLine", "selectNext", "selectPrevious", "selectAllMatches", "addCursorAbove", "addCursorBelow", "selectLarger", "selectSmaller",
	"undoSelection", "redoSelection", "splitSelectionIntoLines", "cancel",
	"undo", "redo", "cut", "copy", "paste", "backspace", "delete", "deleteWordLeft", "deleteWordRight", "deleteSubwordLeft", "deleteSubwordRight",
	"deleteToLineStart", "deleteToLineEnd", "cutToLineEnd", "deleteLine", "newline", "newlineAbove", "newlineBelow", "moveLinesUp", "moveLinesDown",
	"duplicateLinesUp", "duplicateLinesDown", "duplicateSelection", "joinLines", "transpose", "indent", "outdent", "tab", "backtab",
	"toggleComment", "toggleBlockComment", "sortLines", "sortLinesCaseInsensitive", "reverseLines", "uniqueLines",
	"uppercase", "lowercase", "titleCase", "snakeCase", "kebabCase", "camelCase", "pascalCase", "trimTrailingWhitespace",
	"fold", "unfold", "toggleFold", "foldAll", "unfoldAll", "foldRecursive", "unfoldRecursive", "foldSelection",
	"foldLevel1", "foldLevel2", "foldLevel3", "foldLevel4", "foldLevel5", "foldLevel6", "foldLevel7", "foldLevel8", "foldLevel9",
	"find", "findReplace", "findNext", "findPrevious", "useSelectionForFind", "findInSelection",
	"goToLine", "scrollCursorCenter", "scrollLineUp", "scrollLineDown", "goToDefinition", "showCompletions", "showHover",
	"nextHunk", "previousHunk", "revertHunk", "toggleSoftWrap", "toggleLineNumbers",
}

// String returns the command's name, as "toggleComment".
func (c Command) String() string {
	if int(c) < len(commandNames) {
		return commandNames[c]
	}
	return "unknown"
}

// CommandNamed returns the command of a name String returns.
func CommandNamed(name string) (Command, bool) {
	for i, n := range commandNames {
		if n == name {
			return Command(i), true
		}
	}
	return NoCommand, false
}

// Binding runs Command for a key: Mods+Key, followed by
// ThenMods+ThenKey when ThenKey is set, as the chords Cmd+K Cmd+0. A
// binding to NoCommand takes a key away from the editor.
type Binding struct {
	Mods     ui.Modifiers
	Key      ui.Key
	ThenMods ui.Modifiers
	ThenKey  ui.Key
	Command  Command
}

// DefaultBindings returns the key bindings of code editors on this
// platform: Zed's on macOS, and the same with Control, as Visual Studio
// Code's, elsewhere.
func DefaultBindings() []Binding {
	if isMac() {
		return macCodeBindings()
	}
	return pcCodeBindings()
}

func macCodeBindings() []Binding {
	cmd, alt, ctrl, sh := ui.Super, ui.Alt, ui.Ctrl, ui.Shift
	b := []Binding{
		{0, ui.KeyLeft, 0, 0, MoveLeft}, {0, ui.KeyRight, 0, 0, MoveRight},
		{0, ui.KeyUp, 0, 0, MoveUp}, {0, ui.KeyDown, 0, 0, MoveDown},
		{ctrl, ui.KeyB, 0, 0, MoveLeft}, {ctrl, ui.KeyF, 0, 0, MoveRight},
		{ctrl, ui.KeyP, 0, 0, MoveUp}, {ctrl, ui.KeyN, 0, 0, MoveDown},
		{alt, ui.KeyLeft, 0, 0, MoveWordLeft}, {alt, ui.KeyRight, 0, 0, MoveWordRight},
		{ctrl | alt, ui.KeyLeft, 0, 0, MoveSubwordLeft}, {ctrl | alt, ui.KeyRight, 0, 0, MoveSubwordRight},
		{cmd, ui.KeyLeft, 0, 0, MoveLineStart}, {0, ui.KeyHome, 0, 0, MoveLineStart}, {ctrl, ui.KeyA, 0, 0, MoveLineStart},
		{cmd, ui.KeyRight, 0, 0, MoveLineEnd}, {0, ui.KeyEnd, 0, 0, MoveLineEnd}, {ctrl, ui.KeyE, 0, 0, MoveLineEnd},
		{cmd, ui.KeyUp, 0, 0, MoveDocStart}, {cmd, ui.KeyHome, 0, 0, MoveDocStart},
		{cmd, ui.KeyDown, 0, 0, MoveDocEnd}, {cmd, ui.KeyEnd, 0, 0, MoveDocEnd},
		{ctrl, ui.KeyUp, 0, 0, MoveParagraphUp}, {ctrl, ui.KeyDown, 0, 0, MoveParagraphDown},
		{0, ui.KeyPageUp, 0, 0, MovePageUp}, {0, ui.KeyPageDown, 0, 0, MovePageDown},
		{cmd | sh, ui.KeyBackslash, 0, 0, MoveToBracket}, {ctrl, ui.KeyM, 0, 0, MoveToBracket},
		{sh, ui.KeyLeft, 0, 0, SelectLeft}, {sh, ui.KeyRight, 0, 0, SelectRight},
		{sh, ui.KeyUp, 0, 0, SelectUp}, {sh, ui.KeyDown, 0, 0, SelectDown},
		{alt | sh, ui.KeyLeft, 0, 0, SelectWordLeft}, {alt | sh, ui.KeyRight, 0, 0, SelectWordRight},
		{ctrl | alt | sh, ui.KeyLeft, 0, 0, SelectSubwordLeft}, {ctrl | alt | sh, ui.KeyRight, 0, 0, SelectSubwordRight},
		{cmd | sh, ui.KeyLeft, 0, 0, SelectLineStart}, {sh, ui.KeyHome, 0, 0, SelectLineStart},
		{cmd | sh, ui.KeyRight, 0, 0, SelectLineEnd}, {sh, ui.KeyEnd, 0, 0, SelectLineEnd},
		{cmd | sh, ui.KeyUp, 0, 0, SelectDocStart}, {cmd | sh, ui.KeyHome, 0, 0, SelectDocStart},
		{cmd | sh, ui.KeyDown, 0, 0, SelectDocEnd}, {cmd | sh, ui.KeyEnd, 0, 0, SelectDocEnd},
		{ctrl | sh, ui.KeyUp, 0, 0, SelectParagraphUp}, {ctrl | sh, ui.KeyDown, 0, 0, SelectParagraphDown},
		{sh, ui.KeyPageUp, 0, 0, SelectPageUp}, {sh, ui.KeyPageDown, 0, 0, SelectPageDown},
		{cmd, ui.KeyA, 0, 0, SelectAll}, {cmd, ui.KeyL, 0, 0, SelectLine},
		{cmd, ui.KeyD, 0, 0, SelectNext}, {ctrl | cmd, ui.KeyD, 0, 0, SelectPrevious},
		{cmd | sh, ui.KeyL, 0, 0, SelectAllMatches}, {cmd, ui.KeyF2, 0, 0, SelectAllMatches},
		{cmd | alt, ui.KeyUp, 0, 0, AddCursorAbove}, {cmd | alt, ui.KeyDown, 0, 0, AddCursorBelow},
		{ctrl | sh, ui.KeyRight, 0, 0, SelectLarger}, {ctrl | cmd, ui.KeyRight, 0, 0, SelectLarger},
		{ctrl | sh, ui.KeyLeft, 0, 0, SelectSmaller}, {ctrl | cmd, ui.KeyLeft, 0, 0, SelectSmaller},
		{cmd, ui.KeyU, 0, 0, UndoSelection}, {cmd | sh, ui.KeyU, 0, 0, RedoSelection},
		{0, ui.KeyEscape, 0, 0, Cancel},
		{cmd, ui.KeyZ, 0, 0, Undo}, {cmd | sh, ui.KeyZ, 0, 0, Redo},
		{cmd, ui.KeyX, 0, 0, Cut}, {cmd, ui.KeyC, 0, 0, Copy}, {cmd, ui.KeyV, 0, 0, Paste},
		{0, ui.KeyBackspace, 0, 0, Backspace}, {sh, ui.KeyBackspace, 0, 0, Backspace}, {ctrl, ui.KeyH, 0, 0, Backspace},
		{0, ui.KeyDelete, 0, 0, Delete}, {ctrl, ui.KeyD, 0, 0, Delete},
		{alt, ui.KeyBackspace, 0, 0, DeleteWordLeft}, {ctrl, ui.KeyW, 0, 0, DeleteWordLeft},
		{alt, ui.KeyDelete, 0, 0, DeleteWordRight},
		{ctrl | alt, ui.KeyBackspace, 0, 0, DeleteSubwordLeft}, {ctrl | alt, ui.KeyDelete, 0, 0, DeleteSubwordRight},
		{cmd, ui.KeyBackspace, 0, 0, DeleteToLineStart}, {cmd, ui.KeyDelete, 0, 0, DeleteToLineEnd},
		{ctrl, ui.KeyK, 0, 0, CutToLineEnd},
		{cmd | sh, ui.KeyK, 0, 0, DeleteLine},
		{0, ui.KeyEnter, 0, 0, Newline}, {sh, ui.KeyEnter, 0, 0, Newline},
		{cmd, ui.KeyEnter, 0, 0, NewlineBelow}, {cmd | sh, ui.KeyEnter, 0, 0, NewlineAbove},
		{alt, ui.KeyUp, 0, 0, MoveLinesUp}, {alt, ui.KeyDown, 0, 0, MoveLinesDown},
		{alt | sh, ui.KeyUp, 0, 0, DuplicateLinesUp}, {alt | sh, ui.KeyDown, 0, 0, DuplicateLinesDown},
		{ctrl, ui.KeyJ, 0, 0, JoinLines}, {ctrl, ui.KeyT, 0, 0, Transpose},
		{cmd, ui.KeyBracketRight, 0, 0, Indent}, {cmd, ui.KeyBracketLeft, 0, 0, Outdent},
		{0, ui.KeyTab, 0, 0, Tab}, {sh, ui.KeyTab, 0, 0, Backtab},
		{cmd, ui.KeySlash, 0, 0, ToggleComment}, {cmd, ui.KeyK, cmd, ui.KeySlash, ToggleBlockComment},
		{cmd | alt, ui.KeyBracketLeft, 0, 0, Fold}, {cmd | alt, ui.KeyBracketRight, 0, 0, Unfold},
		{cmd, ui.KeyK, cmd, ui.KeyL, ToggleFold}, {cmd, ui.KeyK, cmd, ui.Key0, FoldAll}, {cmd, ui.KeyK, cmd, ui.KeyJ, UnfoldAll},
		{cmd, ui.KeyK, cmd, ui.KeyBracketLeft, FoldRecursive}, {cmd, ui.KeyK, cmd, ui.KeyBracketRight, UnfoldRecursive},
		{cmd, ui.KeyF, 0, 0, Find}, {cmd | alt, ui.KeyF, 0, 0, FindReplace}, {cmd | sh, ui.KeyH, 0, 0, FindReplace},
		{cmd, ui.KeyG, 0, 0, FindNext}, {cmd | sh, ui.KeyG, 0, 0, FindPrevious},
		{cmd, ui.KeyE, 0, 0, UseSelectionForFind}, {cmd | alt, ui.KeyL, 0, 0, FindInSelection},
		{ctrl, ui.KeyG, 0, 0, GoToLine}, {ctrl, ui.KeyL, 0, 0, ScrollCursorCenter},
		{ctrl, ui.KeyPageUp, 0, 0, ScrollLineUp}, {ctrl, ui.KeyPageDown, 0, 0, ScrollLineDown},
		{0, ui.KeyF12, 0, 0, GoToDefinition}, {ctrl, ui.KeySpace, 0, 0, ShowCompletions},
		{cmd, ui.KeyK, cmd, ui.KeyI, ShowHover},
		{cmd, ui.KeyF8, 0, 0, NextHunk}, {cmd | sh, ui.KeyF8, 0, 0, PreviousHunk}, {cmd | alt, ui.KeyZ, 0, 0, RevertHunk},
		{cmd, ui.KeyK, 0, ui.KeyZ, ToggleSoftWrap}, {cmd, ui.KeySemicolon, 0, 0, ToggleLineNumbers},
	}
	for i, k := range []ui.Key{ui.Key1, ui.Key2, ui.Key3, ui.Key4, ui.Key5, ui.Key6, ui.Key7, ui.Key8, ui.Key9} {
		b = append(b, Binding{cmd, ui.KeyK, cmd, k, FoldLevel1 + Command(i)})
	}
	return b
}

func pcCodeBindings() []Binding {
	alt, ctrl, sh := ui.Alt, ui.Ctrl, ui.Shift
	b := []Binding{
		{0, ui.KeyLeft, 0, 0, MoveLeft}, {0, ui.KeyRight, 0, 0, MoveRight},
		{0, ui.KeyUp, 0, 0, MoveUp}, {0, ui.KeyDown, 0, 0, MoveDown},
		{ctrl, ui.KeyLeft, 0, 0, MoveWordLeft}, {ctrl, ui.KeyRight, 0, 0, MoveWordRight},
		{alt, ui.KeyLeft, 0, 0, MoveSubwordLeft}, {alt, ui.KeyRight, 0, 0, MoveSubwordRight},
		{0, ui.KeyHome, 0, 0, MoveLineStart}, {0, ui.KeyEnd, 0, 0, MoveLineEnd},
		{ctrl, ui.KeyHome, 0, 0, MoveDocStart}, {ctrl, ui.KeyEnd, 0, 0, MoveDocEnd},
		{0, ui.KeyPageUp, 0, 0, MovePageUp}, {0, ui.KeyPageDown, 0, 0, MovePageDown},
		{ctrl | sh, ui.KeyBackslash, 0, 0, MoveToBracket},
		{sh, ui.KeyLeft, 0, 0, SelectLeft}, {sh, ui.KeyRight, 0, 0, SelectRight},
		{sh, ui.KeyUp, 0, 0, SelectUp}, {sh, ui.KeyDown, 0, 0, SelectDown},
		{ctrl | sh, ui.KeyLeft, 0, 0, SelectWordLeft}, {ctrl | sh, ui.KeyRight, 0, 0, SelectWordRight},
		{alt | sh, ui.KeyLeft, 0, 0, SelectSmaller}, {alt | sh, ui.KeyRight, 0, 0, SelectLarger},
		{sh, ui.KeyHome, 0, 0, SelectLineStart}, {sh, ui.KeyEnd, 0, 0, SelectLineEnd},
		{ctrl | sh, ui.KeyHome, 0, 0, SelectDocStart}, {ctrl | sh, ui.KeyEnd, 0, 0, SelectDocEnd},
		{sh, ui.KeyPageUp, 0, 0, SelectPageUp}, {sh, ui.KeyPageDown, 0, 0, SelectPageDown},
		{ctrl, ui.KeyA, 0, 0, SelectAll}, {ctrl, ui.KeyL, 0, 0, SelectLine},
		{ctrl, ui.KeyD, 0, 0, SelectNext}, {ctrl | sh, ui.KeyL, 0, 0, SelectAllMatches},
		{ctrl | alt, ui.KeyUp, 0, 0, AddCursorAbove}, {ctrl | alt, ui.KeyDown, 0, 0, AddCursorBelow},
		{ctrl, ui.KeyU, 0, 0, UndoSelection}, {ctrl | sh, ui.KeyU, 0, 0, RedoSelection},
		{0, ui.KeyEscape, 0, 0, Cancel},
		{ctrl, ui.KeyZ, 0, 0, Undo}, {ctrl | sh, ui.KeyZ, 0, 0, Redo}, {ctrl, ui.KeyY, 0, 0, Redo},
		{ctrl, ui.KeyX, 0, 0, Cut}, {ctrl, ui.KeyC, 0, 0, Copy}, {ctrl, ui.KeyV, 0, 0, Paste},
		{sh, ui.KeyDelete, 0, 0, Cut}, {ctrl, ui.KeyInsert, 0, 0, Copy}, {sh, ui.KeyInsert, 0, 0, Paste},
		{0, ui.KeyBackspace, 0, 0, Backspace}, {sh, ui.KeyBackspace, 0, 0, Backspace},
		{0, ui.KeyDelete, 0, 0, Delete},
		{ctrl, ui.KeyBackspace, 0, 0, DeleteWordLeft}, {ctrl, ui.KeyDelete, 0, 0, DeleteWordRight},
		{alt, ui.KeyBackspace, 0, 0, DeleteSubwordLeft}, {alt, ui.KeyDelete, 0, 0, DeleteSubwordRight},
		{ctrl | sh, ui.KeyBackspace, 0, 0, DeleteToLineStart}, {ctrl | sh, ui.KeyDelete, 0, 0, DeleteToLineEnd},
		{ctrl | sh, ui.KeyK, 0, 0, DeleteLine},
		{0, ui.KeyEnter, 0, 0, Newline}, {sh, ui.KeyEnter, 0, 0, Newline},
		{ctrl, ui.KeyEnter, 0, 0, NewlineBelow}, {ctrl | sh, ui.KeyEnter, 0, 0, NewlineAbove},
		{alt, ui.KeyUp, 0, 0, MoveLinesUp}, {alt, ui.KeyDown, 0, 0, MoveLinesDown},
		{alt | sh, ui.KeyUp, 0, 0, DuplicateLinesUp}, {alt | sh, ui.KeyDown, 0, 0, DuplicateLinesDown},
		{ctrl, ui.KeyJ, 0, 0, JoinLines},
		{ctrl, ui.KeyBracketRight, 0, 0, Indent}, {ctrl, ui.KeyBracketLeft, 0, 0, Outdent},
		{0, ui.KeyTab, 0, 0, Tab}, {sh, ui.KeyTab, 0, 0, Backtab},
		{ctrl, ui.KeySlash, 0, 0, ToggleComment}, {ctrl | sh, ui.KeyA, 0, 0, ToggleBlockComment},
		{ctrl | sh, ui.KeyBracketLeft, 0, 0, Fold}, {ctrl | sh, ui.KeyBracketRight, 0, 0, Unfold},
		{ctrl, ui.KeyK, ctrl, ui.KeyL, ToggleFold}, {ctrl, ui.KeyK, ctrl, ui.Key0, FoldAll}, {ctrl, ui.KeyK, ctrl, ui.KeyJ, UnfoldAll},
		{ctrl, ui.KeyK, ctrl, ui.KeyBracketLeft, FoldRecursive}, {ctrl, ui.KeyK, ctrl, ui.KeyBracketRight, UnfoldRecursive},
		{ctrl, ui.KeyF, 0, 0, Find}, {ctrl, ui.KeyH, 0, 0, FindReplace},
		{0, ui.KeyF3, 0, 0, FindNext}, {sh, ui.KeyF3, 0, 0, FindPrevious},
		{ctrl, ui.KeyE, 0, 0, UseSelectionForFind}, {ctrl | alt, ui.KeyL, 0, 0, FindInSelection},
		{ctrl, ui.KeyG, 0, 0, GoToLine},
		{ctrl, ui.KeyUp, 0, 0, ScrollLineUp}, {ctrl, ui.KeyDown, 0, 0, ScrollLineDown},
		{0, ui.KeyF12, 0, 0, GoToDefinition}, {ctrl, ui.KeySpace, 0, 0, ShowCompletions},
		{ctrl, ui.KeyK, ctrl, ui.KeyI, ShowHover},
		{alt, ui.KeyF5, 0, 0, NextHunk}, {alt | sh, ui.KeyF5, 0, 0, PreviousHunk}, {ctrl | alt, ui.KeyZ, 0, 0, RevertHunk},
		{alt, ui.KeyZ, 0, 0, ToggleSoftWrap},
	}
	for i, k := range []ui.Key{ui.Key1, ui.Key2, ui.Key3, ui.Key4, ui.Key5, ui.Key6, ui.Key7, ui.Key8, ui.Key9} {
		b = append(b, Binding{ctrl, ui.KeyK, ctrl, k, FoldLevel1 + Command(i)})
	}
	return b
}

// keymap finds the binding of keys: the app's first, then the
// defaults.
type keymap struct {
	app, def []Binding
}

var defaultBindingList []Binding

func defaultCodeBindings() []Binding {
	if defaultBindingList == nil {
		defaultBindingList = DefaultBindings()
	}
	return defaultBindingList
}

// lookup finds the binding of mods+key, after the first key of a chord
// when chord is set. prefix reports that the key starts a chord.
func (km keymap) lookup(chord *Binding, mods ui.Modifiers, key ui.Key) (b *Binding, prefix bool) {
	for _, list := range [][]Binding{km.app, km.def} {
		for i := range list {
			c := &list[i]
			if chord != nil {
				if c.ThenKey != 0 && c.Mods == chord.Mods && c.Key == chord.Key && c.ThenMods == mods && c.ThenKey == key {
					return c, false
				}
				continue
			}
			if c.Mods == mods && c.Key == key {
				if c.ThenKey != 0 {
					prefix = true
					continue
				}
				return c, false
			}
		}
		if b != nil || prefix {
			break
		}
	}
	return nil, prefix
}

// isMac reports whether the keys are those of macOS, as ui's shortcuts
// decide.
func isMac() bool { return ui.Cmd == ui.Super }
