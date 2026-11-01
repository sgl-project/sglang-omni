// Package core holds what every VoxtNew package shares: the event bus,
// the error sentinels, ULIDs, and the few entities of the shell
// (notifications, log lines) with the service interfaces behind them.
//
// core is a leaf: it imports no other internal package, so every other
// package may import it.
//
// Rules for everything in this package:
//
//   - Service methods are safe to call from any goroutine and may block on
//     I/O: never call them from a MyGo build pass. The UI calls them from a
//     goroutine and learns about the outcome from the Bus, applying it with
//     Window.Update.
//   - Every service method that may block takes a context.Context first.
//   - Entities are plain values. Services return copies.
//   - Errors wrap the sentinels in errors.go, so callers test with errors.Is.
package core
