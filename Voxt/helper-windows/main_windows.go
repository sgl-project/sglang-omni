//go:build windows

package main

import (
	"fmt"
	"os"
)

func main() {
	if err := runHelper(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
