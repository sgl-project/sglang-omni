//go:build windows

package main

import (
	"os"
	"testing"
)

// denyLegacyRead empties the database: an exclusive read handle does not
// stop a later open here, but a byte read of an empty file ends in EOF, so
// legacyReadable finds it not readable, as a database that cannot be
// opened.
func denyLegacyRead(t *testing.T, path string) {
	t.Helper()
	if err := os.WriteFile(path, nil, 0o644); err != nil {
		t.Fatal(err)
	}
}
