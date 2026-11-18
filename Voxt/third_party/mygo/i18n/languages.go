package i18n

import (
	"slices"
	"sync"
)

// languages are the languages an app ships, as the catalogs it registers
// declare them: the source language of the first first, then the others in
// the order they came.
var languages struct {
	mu   sync.Mutex
	list []string
}

// declare adds the languages of a catalog to those the app ships, its
// source language first when it is the first catalog.
func declare(source string, others []string) {
	languages.mu.Lock()
	defer languages.mu.Unlock()
	for _, tag := range append([]string{source}, others...) {
		if tag != "" && !slices.Contains(languages.list, tag) {
			languages.list = append(languages.list, tag)
		}
	}
}

// Languages returns the languages of the catalogs the app registered, the
// source language of the first one first, or nil when it registered none.
func Languages() []string {
	languages.mu.Lock()
	defer languages.mu.Unlock()
	return slices.Clone(languages.list)
}
