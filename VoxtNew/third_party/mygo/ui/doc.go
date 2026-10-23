// Package ui builds user interfaces that MyGo draws itself, on the GPU
// where it can, for windows that show no web page:
//
//	type counter struct{ n int }
//
//	func (s *counter) view(c *ui.Context) {
//		ui.Column(c).Fill().Center().Gap(12).Children(func() {
//			ui.Text(c, fmt.Sprint(s.n)).FontSize(40).Bold()
//			if ui.PrimaryButton(c, "Increment").Clicked() {
//				s.n++
//			}
//		})
//	}
//
//	mygo.NewWindow(mygo.WindowOptions{Title: "Counter", Content: ui.View(s.view)})
//
// A view is a function of the app's state that builds the interface of one
// frame on the main thread; MyGo calls it again after input, after
// Window.Update or Invalidate, and while something animates. Elements lay
// out their children with flexbox or in a Grid, take their colors from a
// Theme that follows the system's appearance, and answer questions about
// input, such as Clicked, where they are built. What an element keeps
// between frames (focus, scrolling, text being edited, animations) follows
// its position among its siblings, or its Key.
// Constructors return checked Element values that expire before the next
// build pass; one frame can contain several passes. Keep elements and custom
// parts out of app state and background goroutines. Handle provides persistent
// control identity for focus, commands and committed bounds; FocusBind connects
// desired focus to app data, and FocusedValue reads actual focus.
//
// Views receive a stable *Context for their window, with its parent scope
// changing inside Children callbacks. Capture Services for persistent
// clipboard/URL callbacks and background redraws instead of storing Context.
// A control takes the user's input as its constructor returns, as Dear
// ImGui's and egui's widgets do: what it binds, a local copy too, holds
// the edit when the next statement reads it, and inline actions see the
// edits of the controls built before them. A container takes its input
// with its children, as the call that built it returns. Input follows the
// options of the frame the user acted on: Disabled, ReadOnly or Step set
// in a build govern the input of the frames after it. Changed and
// Submitted report input; OnClick, OnShortcut, OnChange and OnSubmit run
// after construction, before the rebuild.
// Input queries such as Clicked remain available.
//
// Widgets take the theme's look, whose Spacing sizes them all. Each is
// built on a base without a look, such as ButtonBase, CheckboxBase,
// TabsBase or SelectBase, which handles the pointer, the keyboard, the
// focus and accessibility: style its elements for a design of your own.
//
// Tester runs views in tests, without a window.
package ui
