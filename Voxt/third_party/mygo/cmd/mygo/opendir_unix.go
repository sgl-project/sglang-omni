//go:build !windows

package main

import "os"

// openDir opens a directory to list it.
func openDir(name string) (*os.File, error) { return os.Open(name) }
