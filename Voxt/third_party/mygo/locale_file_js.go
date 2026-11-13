//go:build js

package mygo

// Browsers keep the app's language in localStorage (the backend's
// SaveLocale): there is no file.

func readLocaleFile() string { return "" }

func writeLocaleFile(string) {}
