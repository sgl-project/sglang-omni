package main

import (
	"os"
)

func main() {
	run := Run
	if len(os.Args) == 2 && os.Args[1] == "--connector" { run = RunConnector }
	if err := run(os.Stdin, os.Stdout, Option{}); err != nil {
		os.Exit(1)
	}
}
