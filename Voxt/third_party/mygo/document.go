package mygo

import (
	"errors"
	"fmt"
	"github.com/egoist/mygo/internal/l10n"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"time"
	"unicode"
)

// DocumentType is a kind of file that Documents reads and writes.
type DocumentType struct {
	// Name describes the files, e.g. "Markdown Document". Open and save
	// dialogs show it.
	Name string
	// Extensions are the file name extensions, without dots, e.g.
	// []string{"md", "markdown"}. New documents are saved with the first.
	Extensions []string
}

// Autosave is how documents save their changes.
type Autosave int

const (
	// AutosaveInPlace writes a document's changes to its file a moment
	// after the user stops editing, when its window loses the focus,
	// and when it closes, as macOS apps do: saved documents never ask to
	// be saved. Untitled documents ask when they close (Delete, Cancel or
	// Save), and so do documents that can't be written (read-only,
	// changed on disk).
	AutosaveInPlace Autosave = iota
	// AutosaveOff writes documents when the user saves them, and asks to
	// save their changes when they close, as Windows and Linux apps
	// traditionally do.
	AutosaveOff
)

// DocumentsOptions configures NewDocuments. Read, Write, Window and at
// least one Type are required.
type DocumentsOptions[T any] struct {
	// Types are the kinds of files the documents are. Open dialogs show
	// them, and new documents are saved as the first.
	Types []DocumentType
	// New returns the content of a new, untitled document; nil gives the
	// zero value of T.
	New func() T
	// Read reads a document's content from its file. It runs on a
	// goroutine of its own.
	Read func(r io.Reader) (T, error)
	// Write writes a document's content to its file. It runs on a
	// goroutine of its own, with the content as it was when saving
	// started: edits made meanwhile stay unsaved. A content that is a
	// value the app replaces with Document.Set (a string, a struct of
	// values, a persistent snapshot such as ui.TextSnapshot) can be
	// written while the app goes on; a mutable one needs Snapshot.
	Write func(w io.Writer, content T) error
	// Snapshot, when set, copies a mutable content before it is written,
	// on the main thread, where the app changes it. It is called with the
	// content and returns a copy that nothing changes, which Write gets.
	Snapshot func(content T) T
	// Equal, when set, compares contents: a document whose content is
	// Equal to what its file has is not edited, so that undoing back to
	// it, or typing a character and deleting it, leaves the document
	// unedited. Without it, every Set and Changed edits the document
	// until it is saved.
	Equal func(a, b T) bool
	// Window creates the window that shows a document, on the main
	// thread. MyGo sets its title, represented file and edited state,
	// asks to save before it closes, and closes the document with it.
	Window func(doc *Document[T]) *Window

	// Autosave is how documents save their changes: AutosaveInPlace
	// (the default), or AutosaveOff.
	Autosave Autosave
	// AutosaveDelay is how long after the last edit a document is saved
	// in place: 5 seconds by default. A document being edited without a
	// pause is saved at least every 30 seconds.
	AutosaveDelay time.Duration
	// AskOnQuit asks to save the documents with unsaved changes when the
	// app quits, rather than keeping them and opening them again, still
	// edited, the next time the app starts, as VS Code and Zed do.
	AskOnQuit bool
	// NoUntitled opens no untitled document when the app starts without
	// documents to open, or (macOS) when its Dock icon is clicked while
	// it has no windows.
	NoUntitled bool
}

// Documents opens, creates and keeps track of the documents of one kind,
// and their windows, as AppKit's NSDocumentController and SwiftUI's
// DocumentGroup do. Create it with NewDocuments, before App.Run. Its
// methods are safe from any goroutine.
//
// Documents gives an app the File menu's commands (the RoleNew,
// RoleOpen, RoleSave… menu roles), autosave, a recovery copy of every
// unsaved change, documents opened from the file manager (OnOpenFile),
// the documents changed by other apps, and asks to save before closing
// and quitting where it must. See the documents guide.
type Documents[T any] struct {
	kind *docKind
	opts DocumentsOptions[T]
}

// ErrCanceled is what a document's command returns when the user cancels
// it, in a dialog it showed.
var ErrCanceled = errors.New("mygo: canceled")

// ErrDocumentChanged is what Document.Save returns when another app
// changed the document's file since it was read or saved; SaveAnyway
// writes it all the same.
var ErrDocumentChanged = errors.New("mygo: the file was changed by another application")

// NewDocuments creates the controller of the documents of one kind. Call
// it before App.Run: when the app starts it opens the files it was
// started with, the documents a crash or a quit left unsaved, or else a
// new untitled document. An app may have several kinds of documents; the
// first is the one RoleNew creates.
func NewDocuments[T any](opts DocumentsOptions[T]) *Documents[T] {
	switch {
	case len(opts.Types) == 0 || len(opts.Types[0].Extensions) == 0:
		panic("mygo: NewDocuments needs a type with an extension")
	case opts.Read == nil || opts.Write == nil:
		panic("mygo: NewDocuments needs Read and Write")
	case opts.Window == nil:
		panic("mygo: NewDocuments needs Window")
	}
	for _, t := range opts.Types {
		for _, e := range t.Extensions {
			if e == "" || strings.ContainsAny(e, `./\`) {
				panic(fmt.Sprintf("mygo: NewDocuments: invalid extension %q", e))
			}
		}
	}
	if opts.AutosaveDelay <= 0 {
		opts.AutosaveDelay = 5 * time.Second
	}
	ds := &Documents[T]{opts: opts}
	ds.kind = newDocKind(docKindConfig{
		types:         opts.Types,
		autosave:      opts.Autosave,
		autosaveDelay: opts.AutosaveDelay,
		askOnQuit:     opts.AskOnQuit,
		noUntitled:    opts.NoUntitled,
		create: func(content any, name string) (*docBase, error) {
			v, _ := content.(T)
			return &ds.newDocument(v, name).docBase, nil
		},
		read: func(r io.Reader) (any, error) { return opts.Read(r) },
		empty: func() any {
			if opts.New != nil {
				return opts.New()
			}
			var zero T
			return zero
		},
	})
	return ds
}

// New creates an untitled document with the content of
// DocumentsOptions.New and opens its window.
func (ds *Documents[T]) New() *Document[T] {
	needsApp("Documents.New")
	b := ds.kind.newUntitled()
	return b.ops.(*Document[T])
}

// Open opens the document at path in its window, or focuses the window
// of the document if it is open already.
func (ds *Documents[T]) Open(path string) (*Document[T], error) {
	needsApp("Documents.Open")
	b, err := ds.kind.open(path)
	if err != nil {
		return nil, err
	}
	return b.ops.(*Document[T]), nil
}

// OpenDialog asks which documents to open, and opens them.
func (ds *Documents[T]) OpenDialog() ([]*Document[T], error) {
	needsApp("Documents.OpenDialog")
	paths, err := Dialog.Open(OpenDialogOptions{Parent: FocusedWindow(), Filters: ds.kind.filters(), Multiple: true})
	if err != nil || len(paths) == 0 {
		return nil, err
	}
	var docs []*Document[T]
	for _, p := range paths {
		d, err := ds.Open(p)
		if err != nil {
			return docs, err
		}
		docs = append(docs, d)
	}
	return docs, nil
}

// Documents returns the open documents, in the order they opened.
func (ds *Documents[T]) Documents() []*Document[T] {
	var docs []*Document[T]
	for _, b := range ds.kind.list() {
		docs = append(docs, b.ops.(*Document[T]))
	}
	return docs
}

// ForWindow returns the document w shows, or nil.
func (ds *Documents[T]) ForWindow(w *Window) *Document[T] {
	if b := documentFor(w); b != nil && b.kind == ds.kind {
		return b.ops.(*Document[T])
	}
	return nil
}

// Document is an open document: its content, the file it was read from
// and is saved to, and its window. Its methods are safe from any
// goroutine.
type Document[T any] struct {
	docBase
	ds *Documents[T]
	// content and saved are guarded by docBase.mu; saved is the content
	// the file has, for DocumentsOptions.Equal, when savedKnown: a
	// document restored from its recovery copy doesn't know it.
	content, saved T
	savedKnown     bool
}

func (ds *Documents[T]) newDocument(content T, name string) *Document[T] {
	d := &Document[T]{ds: ds, content: content, saved: content, savedKnown: true}
	d.ops = d
	d.kind = ds.kind
	d.name = name
	return d
}

// Content returns the document's content.
func (d *Document[T]) Content() T {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.content
}

// Set replaces the document's content, which edits it, unless
// DocumentsOptions.Equal finds it equal to what its file has.
func (d *Document[T]) Set(content T) {
	d.mu.Lock()
	d.content = content
	d.changes++
	edited := true
	if eq := d.ds.opts.Equal; eq != nil && d.savedKnown {
		edited = !eq(content, d.saved)
	}
	d.mu.Unlock()
	d.changed(edited)
}

// Changed tells that the content changed without being replaced, for a
// mutable content (see DocumentsOptions.Snapshot). It edits the document.
func (d *Document[T]) Changed() {
	d.mu.Lock()
	d.changes++
	d.mu.Unlock()
	d.changed(true)
}

// SetEdited sets whether the document has unsaved changes, for an app
// that knows better, as one that keeps the undo history of its content:
// false makes the content the one its file has, without writing it.
func (d *Document[T]) SetEdited(edited bool) {
	d.mu.Lock()
	d.changes++
	if !edited {
		d.savedAt = d.changes
		d.saved, d.savedKnown = d.content, true
	}
	d.mu.Unlock()
	d.changed(edited)
}

// Window returns the window that shows the document.
func (d *Document[T]) Window() *Window {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.window
}

// capture takes what a write of the document writes, and the change it
// is at: on the main thread with a Snapshot, else as it is now.
func (d *Document[T]) capture() docSnapshot {
	take := func() docSnapshot {
		d.mu.Lock()
		v, at := d.content, d.changes
		d.mu.Unlock()
		if snap := d.ds.opts.Snapshot; snap != nil {
			v = snap(v)
		}
		write := d.ds.opts.Write
		return docSnapshot{at: at, value: v, write: func(w io.Writer) error { return write(w, v) }}
	}
	if d.ds.opts.Snapshot != nil {
		return onMainValue(take)
	}
	return take()
}

// markSaved records that the file has s. d.mu is held.
func (d *Document[T]) markSaved(s docSnapshot) {
	v := s.value.(T)
	d.saved, d.savedKnown = v, true
	d.edited = d.changes != s.at
	if eq := d.ds.opts.Equal; eq != nil && d.edited {
		d.edited = !eq(d.content, v)
	}
}

// replace makes v, read from the file, the content. d.mu is held.
func (d *Document[T]) replace(v any) {
	d.content, d.saved, d.savedKnown = v.(T), v.(T), true
	d.changes++
	d.savedAt = d.changes
	d.edited = false
}

// restore makes v, a recovery copy, the content, edited. d.mu is held.
func (d *Document[T]) restore(v any) {
	d.content = v.(T)
	d.savedKnown = false
	d.changes++
	d.edited = true
}

func (d *Document[T]) makeWindow() *Window { return d.ds.opts.Window(d) }

// docSnapshot is what a write of a document writes: the content at
// change at.
type docSnapshot struct {
	at    uint64
	value any
	write func(io.Writer) error
}

// docOps is what the generic Document gives the rest of the document
// machinery.
type docOps interface {
	capture() docSnapshot
	markSaved(s docSnapshot)
	replace(v any)
	restore(v any)
	makeWindow() *Window
}

// DocumentState describes a document, as Document.State returns it and
// pages of its window get it.
type DocumentState struct {
	// Name is the document's file name, or a name such as "Untitled 2"
	// for an untitled document.
	Name string `json:"name"`
	// Path is the document's file, "" while it is untitled.
	Path string `json:"path"`
	// Edited reports whether the document has unsaved changes.
	Edited bool `json:"edited"`
	// ReadOnly reports that the document's file can't be written: saving
	// asks for another file.
	ReadOnly bool `json:"readOnly"`
	// Conflict reports that another app changed the document's file while
	// the document had unsaved changes: saving asks which to keep.
	Conflict bool `json:"conflict"`
	// Deleted reports that the document's file was deleted: saving
	// writes it again.
	Deleted bool `json:"deleted"`
}

// docBase is the state of a document that doesn't depend on its
// content's type.
type docBase struct {
	kind *docKind
	ops  docOps
	// id names the document's recovery copy.
	id string

	// mu guards the fields below, and the content of the Document.
	mu   sync.Mutex
	path string
	// name is the name of an untitled document; untitled its number,
	// which other untitled documents don't take.
	name     string
	untitled int
	// changes counts the changes of the content; savedAt is the change
	// the file has, backedUp the one the recovery copy has (0: none).
	changes, savedAt, backedUp uint64
	edited                     bool
	// stamp is the file as the document last read or wrote it.
	stamp                       fileStamp
	readOnly, conflict, deleted bool
	window                      *Window
	closed                      bool
	// discarded tells that the user chose not to save the changes of the
	// closing document; watched is the file watched for it.
	discarded bool
	watched   string
	// dirtySince is when the oldest change not autosaved was made.
	dirtySince time.Time
	timer      *time.Timer

	// io serializes the reads and writes of the document's file and
	// recovery copy.
	io sync.Mutex

	// Main thread only.
	closing bool
	// notifying tells that a state change waits for the main thread.
	notifying bool

	onState  listeners[func(DocumentState)]
	onReload listeners[func()]
}

// Path returns the document's file, "" while it is untitled.
func (b *docBase) Path() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.path
}

// Name returns the document's file name, or the name of an untitled
// document, such as "Untitled 2".
func (b *docBase) Name() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.nameLocked()
}

func (b *docBase) nameLocked() string {
	if b.path != "" {
		return filepath.Base(b.path)
	}
	return b.name
}

// IsEdited reports whether the document has unsaved changes.
func (b *docBase) IsEdited() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.edited
}

// State describes the document.
func (b *docBase) State() DocumentState {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.stateLocked()
}

func (b *docBase) stateLocked() DocumentState {
	return DocumentState{
		Name: b.nameLocked(), Path: b.path, Edited: b.edited,
		ReadOnly: b.readOnly, Conflict: b.conflict, Deleted: b.deleted,
	}
}

// OnStateChange is called on the main thread when the document's state
// changes: it is edited or saved, moved, renamed, or its file changed.
func (b *docBase) OnStateChange(fn func(DocumentState)) (off func()) {
	return b.onState.add(fn, false)
}

// OnReload is called on the main thread after the document's content was
// read from its file again: reverted, or changed by another app. Views
// that keep a copy of the content take it again.
func (b *docBase) OnReload(fn func()) (off func()) { return b.onReload.add(fn, false) }

// changed follows an edit of the content: edited is the new edited state.
func (b *docBase) changed(edited bool) {
	b.mu.Lock()
	was := b.edited
	b.edited = edited
	switch {
	case !edited:
		b.dirtySince = time.Time{}
	case b.dirtySince.IsZero():
		b.dirtySince = time.Now()
	}
	b.scheduleLocked()
	stale := !edited && b.backedUp != 0
	b.mu.Unlock()
	if stale {
		// Back to what the file has: the recovery copy is of no use.
		docWork.Go(func() { b.kind.recovery.remove(b) })
	}
	if was != edited {
		b.notify()
	}
}

// notify tells the window, the menus, the listeners and the pages that
// the state changed, on the main thread, once for changes in a row.
func (b *docBase) notify() {
	postMain(func() {
		if b.notifying {
			return
		}
		b.notifying = true
		postMain(func() {
			b.notifying = false
			s := b.State()
			b.mu.Lock()
			w := b.window
			b.mu.Unlock()
			if w != nil && !w.IsDestroyed() {
				w.SetDocumentEdited(s.Edited)
				w.SetRepresentedFile(s.Path)
				w.SetTitle(documentTitle(s.Name))
				emitDocumentState(w, s)
			}
			fire1(&b.onState, s)
			refreshDocumentMenus()
		})
	})
}

// documentTitle is the title of a document's window: its name, then the
// app's on Windows and Linux, as Notepad and gedit show them.
func documentTitle(name string) string {
	if runtime.GOOS == "darwin" || runtime.GOOS == "js" {
		return name
	}
	return name + " - " + App.Name()
}

// Save writes the document to its file, as File > Save does: an untitled
// document asks where, a read-only one asks for another file, and one
// whose file another app changed asks whether to overwrite it. It returns
// ErrCanceled when the user cancels.
func (b *docBase) Save() error { return b.saveInteractive(false) }

// SaveAs asks where to save the document, then writes it there; the
// document then is that file. A path saves there without asking.
func (b *docBase) SaveAs(path string) error {
	if path == "" {
		return b.saveInteractive(true)
	}
	_, err := b.write(path, writeSaveAs)
	return err
}

// SaveAnyway writes the document to its file even when another app
// changed the file since the document read it.
func (b *docBase) SaveAnyway() error {
	path := b.Path()
	if path == "" {
		return b.saveInteractive(true)
	}
	_, err := b.write(path, writeOverwrite)
	return err
}

// Revert reads the document's file again, discarding the unsaved changes.
func (b *docBase) Revert() error {
	path := b.Path()
	if path == "" {
		return errors.New("mygo: an untitled document has no file to revert to")
	}
	return b.reload(path)
}

// Rename renames the document's file to name, in its folder, as File >
// Rename does; "" asks for the name. An untitled document is saved
// instead.
func (b *docBase) Rename(name string) error { return b.rename(name) }

// MoveTo moves the document's file into the folder dir, as File > Move
// To does; "" asks for the folder.
func (b *docBase) MoveTo(dir string) error { return b.moveTo(dir) }

// Duplicate opens an untitled copy of the document in a new window, as
// File > Duplicate does.
func (d *Document[T]) Duplicate() *Document[T] { return d.duplicate().ops.(*Document[T]) }

// Close closes the document's window, which asks to save the document
// first when it must.
func (b *docBase) Close() {
	b.mu.Lock()
	w := b.window
	b.mu.Unlock()
	if w != nil {
		w.Close()
	}
}

// IsReadOnly reports whether the document's file can't be written.
func (b *docBase) IsReadOnly() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.readOnly
}

// HasConflict reports whether another app changed the document's file
// while the document had unsaved changes.
func (b *docBase) HasConflict() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.conflict
}

// autosaves reports whether the document is saved in place. b.mu is held.
func (b *docBase) autosavesLocked() bool {
	return b.kind.autosave == AutosaveInPlace && b.path != "" && !b.readOnly && !b.conflict && !b.deleted
}

// scheduleLocked arms the timer that autosaves the document or writes its
// recovery copy after the edits stop. b.mu is held.
func (b *docBase) scheduleLocked() {
	if !b.edited || b.closed {
		if b.timer != nil {
			b.timer.Stop()
		}
		return
	}
	delay, most := time.Second, 5*time.Second
	if b.autosavesLocked() {
		delay, most = b.kind.autosaveDelay, 30*time.Second
	}
	if left := most - time.Since(b.dirtySince); left < delay {
		delay = max(left, 0)
	}
	if b.timer == nil {
		b.timer = time.AfterFunc(delay, func() { docWork.Go(b.persist) })
	} else {
		b.timer.Reset(delay)
	}
}

// persist saves the document in place, or else writes its recovery copy.
// It runs on a goroutine.
func (b *docBase) persist() {
	b.mu.Lock()
	in, closed := b.autosavesLocked(), b.closed
	path := b.path
	b.mu.Unlock()
	if closed {
		return
	}
	if in {
		if _, err := b.write(path, writeAutosave); err == nil {
			return
		}
	}
	b.backup()
}

// docWork tracks the writes of recovery copies that nothing waits for,
// for tests.
var docWork sync.WaitGroup

// untitledName returns the name of untitled document n, in the language
// the app shows: "Untitled", "Untitled 2".
func untitledName(n int) string {
	if n == 1 {
		return menuMessages().Untitled()
	}
	return menuMessages().UntitledNumbered(n)
}

// untitledNumber returns the number of an untitled document by its name,
// as untitledName writes it in the language the app shows or in English
// (the app's language may have changed since it was named), or 0.
func untitledNumber(name string) int { return untitledNumberIn(name, appLanguage(), "en") }

// untitledNumberIn is untitledNumber with the names of locales, whose
// digits it reads: "Untitled ۲" in Persian is 2.
func untitledNumberIn(name string, locales ...string) int {
	for _, locale := range locales {
		if name == l10n.In(locale).Untitled() {
			return 1
		}
		// The message around the number.
		format := l10n.Catalog.Format(locale, "untitledNumbered", map[string]any{"n": "\x00"})
		before, after, ok := strings.Cut(format, "\x00")
		if !ok || !strings.HasPrefix(name, before) || !strings.HasSuffix(name, after) || len(name) < len(before)+len(after) {
			continue
		}
		if n, ok := decimalValue(name[len(before) : len(name)-len(after)]); ok && n > 1 {
			return n
		}
	}
	return 0
}

// decimalValue reads a number written in the decimal digits of any
// script, as "12" or "۱۲".
func decimalValue(s string) (int, bool) {
	n := 0
	for _, r := range s {
		d, ok := digitValue(r)
		if !ok || n > (1<<31)/10 {
			return 0, false
		}
		n = n*10 + d
	}
	return n, s != ""
}

// digitValue returns the value of a decimal digit of Unicode: the digits
// of a script are runs of ten, from zero, in the table of Nd.
func digitValue(r rune) (int, bool) {
	for _, rg := range unicode.Nd.R16 {
		if lo, hi := rune(rg.Lo), rune(rg.Hi); r >= lo && r <= hi {
			return int(r-lo) % 10, true
		}
	}
	for _, rg := range unicode.Nd.R32 {
		if lo, hi := rune(rg.Lo), rune(rg.Hi); r >= lo && r <= hi {
			return int(r-lo) % 10, true
		}
	}
	return 0, false
}

// docKind is what Documents of any content type share: options, the open
// documents, and the recovery copies.
type docKind struct {
	docKindConfig
	recovery *recoveryStore

	mu   sync.Mutex
	docs []*docBase
}

type docKindConfig struct {
	types         []DocumentType
	autosave      Autosave
	autosaveDelay time.Duration
	askOnQuit     bool
	noUntitled    bool
	// create makes a document with content, untitled with name; read
	// reads a content; empty is the content of a new document.
	create func(content any, name string) (*docBase, error)
	read   func(r io.Reader) (any, error)
	empty  func() any
}

func newDocKind(c docKindConfig) *docKind {
	k := &docKind{docKindConfig: c}
	k.recovery = newRecoveryStore(k)
	registerDocKind(k)
	return k
}

func (k *docKind) list() []*docBase {
	k.mu.Lock()
	defer k.mu.Unlock()
	return slices.Clone(k.docs)
}

// filters are the dialogs' filters of the kind's types.
func (k *docKind) filters() []FileFilter {
	var fs []FileFilter
	for _, t := range k.types {
		fs = append(fs, FileFilter{Name: t.Name, Extensions: t.Extensions})
	}
	return fs
}

// handles reports whether the kind opens path, by its extension.
func (k *docKind) handles(path string) bool {
	ext := strings.TrimPrefix(strings.ToLower(filepath.Ext(path)), ".")
	for _, t := range k.types {
		for _, e := range t.Extensions {
			if strings.EqualFold(e, ext) {
				return true
			}
		}
	}
	return false
}

// nextUntitled returns the smallest number no untitled document has.
func (k *docKind) nextUntitled() int {
	k.mu.Lock()
	defer k.mu.Unlock()
	used := map[int]bool{}
	for _, b := range k.docs {
		b.mu.Lock()
		if b.path == "" {
			used[b.untitled] = true
		}
		b.mu.Unlock()
	}
	n := 1
	for used[n] {
		n++
	}
	return n
}

// newUntitled creates an untitled document and its window.
func (k *docKind) newUntitled() *docBase {
	n := k.nextUntitled()
	b, _ := k.create(k.empty(), untitledName(n))
	b.untitled = n
	b.id = randomText()
	k.show(b)
	return b
}

// open opens the document at path, or focuses it when it is open.
func (k *docKind) open(path string) (*docBase, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	if b := openDocument(abs); b != nil {
		b.focus()
		return b, nil
	}
	v, stamp, err := k.readFile(abs)
	if err != nil {
		return nil, err
	}
	if b := openDocument(abs); b != nil { // opened meanwhile
		b.focus()
		return b, nil
	}
	b, err := k.create(v, "")
	if err != nil {
		return nil, err
	}
	b.path, b.stamp, b.id = abs, stamp, randomText()
	b.readOnly = !fileWritable(abs)
	k.show(b)
	addRecent(abs)
	return b, nil
}

// readFile reads the content of the file at path, off the main thread.
func (k *docKind) readFile(path string) (v any, stamp fileStamp, err error) {
	read := func() {
		var f *os.File
		if f, err = openShared(path); err != nil {
			return
		}
		defer f.Close()
		var info os.FileInfo
		if info, err = f.Stat(); err != nil {
			return
		}
		if info.IsDir() {
			err = fmt.Errorf("mygo: %s is a directory", path)
			return
		}
		stamp = stampOf(info)
		v, err = k.read(f)
	}
	offMain(func() struct{} { read(); return struct{}{} })
	return v, stamp, err
}

// show registers b and opens its window.
func (k *docKind) show(b *docBase) {
	k.mu.Lock()
	k.docs = append(k.docs, b)
	k.mu.Unlock()
	onMain(func() {
		w := b.ops.makeWindow()
		if w == nil {
			panic("mygo: DocumentsOptions.Window returned no window")
		}
		b.attach(w)
	})
	b.mu.Lock()
	b.scheduleLocked()
	b.mu.Unlock()
	b.watch()
	b.notify()
}
