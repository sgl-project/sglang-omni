// Package uitext makes the words the UI shows from the domain's values:
// shortcuts' names and problems, the models' errors, a session's state, a
// history entry's details, sizes and times, the menu bar's short labels.
// Pure functions of i18n and internal types, without MyGo: package main's
// pages call them, and their tests run without a window.
//
// It is the one internal package that imports i18n (internal/archtest):
// the domain returns codes and errors, and the UI says them. Every
// message is a literal at its i18n.T call, so the catalogue's key parity
// (internal/i18n) finds it here as anywhere.
package uitext
