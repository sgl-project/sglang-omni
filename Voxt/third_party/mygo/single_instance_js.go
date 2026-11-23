//go:build js

package mygo

// RequestSingleInstanceLock makes sure a single instance of the app runs.
// Browsers run an instance in every page that loads the app, unaware of
// the others: it returns true.
func (a *Application) RequestSingleInstanceLock() bool { return true }

// OnSecondInstance is called in the first instance when another instance
// was started, which browsers never tell.
func (a *Application) OnSecondInstance(fn func(args []string, workingDir string)) (off func()) {
	return func() {}
}
