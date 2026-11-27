package main

// `Voxt dictionary import FILE` (docs/dictionary-import.md): the terms of
// a CSV, text or Voxt 1.x file merged into the dictionary, as Dictionary →
// Import… does. The command reads the file itself, so the app needs no
// access to the user's folders, and sends it to the running app over
// internal/dictapi's socket. Its messages are English, as the app's other
// developer commands.

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/sgl-project/sglang-omni/Voxt/internal/dictapi"
)

const dictionaryUsage = `usage:
  Voxt dictionary import [--dry-run] [--json] FILE
FILE is CSV (term, aliases, note, enabled), text (.txt: a term per line,
"Term = alias, alias") or a Voxt 1.x dictionary export (.json).
--dry-run says what the import would do and saves nothing.`

func runDictionaryCLI(args []string) int {
	return dictionaryCLI(args, os.Stdout, os.Stderr, filepath.Join(filepath.Dir(historySocketPath()), dictapi.SocketName), openDictionaryFile)
}

func openDictionaryFile(path string) (io.ReadCloser, error) {
	return os.Open(path)
}

func readDictionaryData(reader io.Reader) ([]byte, error) {
	return io.ReadAll(io.LimitReader(reader, dictapi.MaxBytes+1))
}

func dictionaryCLI(args []string, stdout, stderr io.Writer, socket string, openFile func(string) (io.ReadCloser, error)) int {
	if len(args) == 0 || args[0] != "import" {
		fmt.Fprintln(stderr, dictionaryUsage)
		return exitUsage
	}
	fs := flag.NewFlagSet("dictionary import", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	dryRun := fs.Bool("dry-run", false, "")
	asJSON := fs.Bool("json", false, "")
	// Options may come after FILE too: flag stops at the first argument.
	var files []string
	for rest := args[1:]; ; {
		if err := fs.Parse(rest); err != nil {
			fmt.Fprintln(stderr, dictionaryUsage)
			return exitUsage
		}
		if fs.NArg() == 0 {
			break
		}
		files, rest = append(files, fs.Arg(0)), fs.Args()[1:]
	}
	if len(files) != 1 {
		fmt.Fprintln(stderr, dictionaryUsage)
		return exitUsage
	}
	in := files[0]
	file, err := openFile(in)
	if err != nil {
		fmt.Fprintf(stderr, "Cannot read %s: %v\n", in, err)
		return exitUnreadable
	}
	data, readErr := readDictionaryData(file)
	closeErr := file.Close()
	if readErr != nil {
		err = readErr
	} else {
		err = closeErr
	}
	if err != nil {
		fmt.Fprintf(stderr, "Cannot read %s: %v\n", in, err)
		return exitUnreadable
	}
	res, err := dictapi.Import(socket, filepath.Base(in), data, *dryRun)
	switch {
	case errors.Is(err, dictapi.ErrUnavailable):
		fmt.Fprintln(stderr, "Voxt is not running. Open Voxt, then run the command again.")
		return exitUnavailable
	case err != nil:
		fmt.Fprintf(stderr, "The import stopped: %v\n", err)
		return exitInterrupted
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetEscapeHTML(false)
		if err := enc.Encode(res); err != nil {
			fmt.Fprintf(stderr, "Cannot write the import result: %v\n", err)
			return exitInterrupted
		}
		return exitOK
	}
	verb := "Imported"
	if res.DryRun {
		verb = "Would import"
	}
	fmt.Fprintf(stdout, "%s: %d new, %d updated, %d rows skipped.\n", verb, res.Added, res.Updated, res.Skipped)
	return exitOK
}
