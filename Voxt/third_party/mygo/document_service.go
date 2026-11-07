package mygo

import (
	"context"
	"errors"
)

// DocumentService gives the pages of document windows their documents,
// through the generated TypeScript client: bind it with BindAs, under the
// name the client calls it by,
//
//	mygo.BindAs("Note", notes.Service())
//
// and a page reads and replaces the content of the document its window
// shows, and saves it:
//
//	const text = await Note.content();
//	await Note.setContent(text + "!");
//	onDocumentState((state) => { title.textContent = state.name }); // mygo-runtime
//
// Its methods fail for windows that show no document of its kind.
type DocumentService[T any] struct{ ds *Documents[T] }

// Service returns the service that gives pages their documents.
func (ds *Documents[T]) Service() *DocumentService[T] { return &DocumentService[T]{ds} }

var errNoDocument = errors.New("mygo: the window shows no document of this kind")

func (s *DocumentService[T]) doc(ctx context.Context) (*Document[T], error) {
	if d := s.ds.ForWindow(CallerWindow(ctx)); d != nil {
		return d, nil
	}
	return nil, errNoDocument
}

// Content returns the content of the window's document.
func (s *DocumentService[T]) Content(ctx context.Context) (T, error) {
	d, err := s.doc(ctx)
	if err != nil {
		var zero T
		return zero, err
	}
	return d.Content(), nil
}

// SetContent replaces the content of the window's document, which edits
// it.
func (s *DocumentService[T]) SetContent(ctx context.Context, content T) error {
	d, err := s.doc(ctx)
	if err == nil {
		d.Set(content)
	}
	return err
}

// State describes the window's document; onDocumentState of mygo-runtime
// tells its changes.
func (s *DocumentService[T]) State(ctx context.Context) (DocumentState, error) {
	d, err := s.doc(ctx)
	if err != nil {
		return DocumentState{}, err
	}
	return d.State(), nil
}

// Save saves the window's document, as File > Save does.
func (s *DocumentService[T]) Save(ctx context.Context) error {
	return s.do(ctx, func(d *Document[T]) error { return d.Save() })
}

// SaveAs asks where to save the window's document, and saves it there.
func (s *DocumentService[T]) SaveAs(ctx context.Context) error {
	return s.do(ctx, func(d *Document[T]) error { return d.SaveAs("") })
}

// Duplicate opens a copy of the window's document in a new window.
func (s *DocumentService[T]) Duplicate(ctx context.Context) error {
	return s.do(ctx, func(d *Document[T]) error { d.Duplicate(); return nil })
}

// Rename asks for a new name of the window's document's file.
func (s *DocumentService[T]) Rename(ctx context.Context) error {
	return s.do(ctx, func(d *Document[T]) error { return d.Rename("") })
}

// MoveTo asks where to move the window's document's file.
func (s *DocumentService[T]) MoveTo(ctx context.Context) error {
	return s.do(ctx, func(d *Document[T]) error { return d.MoveTo("") })
}

// Revert asks whether to discard the unsaved changes of the window's
// document, and reads its file again.
func (s *DocumentService[T]) Revert(ctx context.Context) error {
	return s.do(ctx, func(d *Document[T]) error { return d.revert() })
}

func (s *DocumentService[T]) do(ctx context.Context, fn func(*Document[T]) error) error {
	d, err := s.doc(ctx)
	if err != nil {
		return err
	}
	return fn(d)
}
