// Package editor is a code editor and a diff viewer for MyGo's native UI,
// on package ui.
//
// View shows a code editor of a Buffer, as Zed's: syntax highlighting,
// several carets and selections, soft wraps, folding, finding and
// replacing, completions, hovers, diagnostics and inlay hints from the
// app or a language server, and the commands and key bindings of code
// editors:
//
//	st := &editor.State{Buffer: editor.NewBuffer(src)}
//	st.Buffer.SetLanguage(editor.LanguageFor("main.go"))
//	// In the view:
//	editor.View(c, st).Fill()
//
// DiffView shows the changes between two buffers, as Pierre's diffs:
// side by side or unified, with the words changed highlighted, unchanged
// lines collapsed, annotations under lines, and the new side editable.
//
// A Language highlights its text with a Tokenizer, a line at a time, or
// with a Highlighter of its own, as a parser's: the editor gives it the
// edits of the text and asks it for the tokens of the lines it shows,
// and uses what it knows of the text's structure (SyntaxSelector,
// Folder, BracketMatcher, Indenter) for selections, folds, brackets and
// indentation.
//
// Editors lay out and paint only the lines in view: a buffer of a million
// lines scrolls and edits as one of ten.
package editor
