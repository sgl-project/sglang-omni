package main

// `Voxt history export|status` (docs/history-access.md): a documented,
// read-only command that talks to the running app over the connection of
// internal/historyapi. It opens no database and takes no lock; its
// messages are English, as the app's other developer commands.

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/historyapi"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// Exit codes of the command, part of its contract.
const (
	exitOK          = 0
	exitUsage       = 2
	exitUnavailable = 3
	exitInterrupted = 4
)

const historyUsage = `usage:
  Voxt history export (--day YYYY-MM-DD | --since T [--until T])
                      [--kind transcribe|translate|rewrite] [--format jsonl|md] [--as NAME]
  Voxt history status`

func runHistoryCLI(args []string) int {
	return historyCLI(args, os.Stdout, os.Stderr, time.Local, historySocketPath())
}

// historySocketPath is the socket in the folder nameApp makes
// PathUserData: this command runs before and without any of it.
func historySocketPath() string {
	dir, err := os.UserConfigDir()
	if err != nil {
		return ""
	}
	return filepath.Join(dir, "VoxtNew", historyapi.SocketName)
}

func historyCLI(args []string, stdout, stderr io.Writer, loc *time.Location, socket string) int {
	if len(args) == 0 {
		fmt.Fprintln(stderr, historyUsage)
		return exitUsage
	}
	switch args[0] {
	case "export":
		return historyExport(args[1:], stdout, stderr, loc, socket)
	case "status":
		if len(args) != 1 {
			fmt.Fprintln(stderr, historyUsage)
			return exitUsage
		}
		st, err := historyapi.FetchStatus(socket)
		if err != nil {
			return historyFailed(stderr, err)
		}
		enc := json.NewEncoder(stdout)
		enc.SetEscapeHTML(false)
		if err := enc.Encode(st); err != nil {
			return historyFailed(stderr, err)
		}
		return exitOK
	}
	fmt.Fprintln(stderr, historyUsage)
	return exitUsage
}

func historyExport(args []string, stdout, stderr io.Writer, loc *time.Location, socket string) int {
	fs := flag.NewFlagSet("history export", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	day := fs.String("day", "", "")
	sinceArg := fs.String("since", "", "")
	untilArg := fs.String("until", "", "")
	kind := fs.String("kind", "", "")
	format := fs.String("format", historyapi.FormatJSONL, "")
	as := fs.String("as", "", "")
	if err := fs.Parse(args); err != nil || fs.NArg() != 0 {
		fmt.Fprintln(stderr, historyUsage)
		return exitUsage
	}
	since, until, err := historyRange(*day, *sinceArg, *untilArg, loc)
	switch {
	case err != nil:
		fmt.Fprintf(stderr, "%v\n%s\n", err, historyUsage)
		return exitUsage
	case *kind != "" && !slices.Contains(store.Kinds, store.Kind(*kind)):
		fmt.Fprintf(stderr, "unknown kind %q\n%s\n", *kind, historyUsage)
		return exitUsage
	case *format != historyapi.FormatJSONL && *format != historyapi.FormatMarkdown:
		fmt.Fprintf(stderr, "unknown format %q\n%s\n", *format, historyUsage)
		return exitUsage
	}
	req := historyapi.Request{Since: since.UnixMilli(), Kind: *kind, As: *as}
	if !until.IsZero() {
		req.Until = until.UnixMilli()
	}
	rd := historyapi.NewRenderer(stdout, *format)
	if _, err := historyapi.Export(socket, req, rd.Row); err != nil {
		return historyFailed(stderr, err)
	}
	return exitOK
}

// historyRange is the half-open range the arguments ask for. A day is
// the calendar's: time.Date keeps 23- and 25-hour days right.
func historyRange(day, since, until string, loc *time.Location) (from, to time.Time, err error) {
	switch {
	case day != "" && (since != "" || until != ""):
		return from, to, errors.New("--day cannot go with --since or --until")
	case day != "":
		d, err := time.ParseInLocation("2006-01-02", day, loc)
		if err != nil {
			return from, to, fmt.Errorf("bad --day %q", day)
		}
		return d, time.Date(d.Year(), d.Month(), d.Day()+1, 0, 0, 0, 0, loc), nil
	case since == "":
		return from, to, errors.New("say which entries: --day or --since")
	}
	if from, err = historyTime(since, loc); err != nil {
		return from, to, fmt.Errorf("bad --since %q", since)
	}
	if until != "" {
		if to, err = historyTime(until, loc); err != nil {
			return from, to, fmt.Errorf("bad --until %q", until)
		}
	}
	return from, to, nil
}

// historyTime reads RFC 3339, or a date meaning its local midnight.
func historyTime(s string, loc *time.Location) (time.Time, error) {
	if t, err := time.Parse(time.RFC3339, s); err == nil {
		return t, nil
	}
	return time.ParseInLocation("2006-01-02", s, loc)
}

// historyFailed says what went wrong on stderr and picks the exit code.
func historyFailed(stderr io.Writer, err error) int {
	if errors.Is(err, historyapi.ErrUnavailable) {
		fmt.Fprintln(stderr, i18n.Platform("Voxt is not answering: it is not running, or Settings → Privacy → Let programs on this Mac read the history is off. Open Voxt, or turn the setting on.", "Voxt is not answering: it is not running, or Settings → Privacy → Let programs on this PC read the history is off. Open Voxt, or turn the setting on."))
		return exitUnavailable
	}
	fmt.Fprintf(stderr, "The read stopped before the end: %v\n", err)
	return exitInterrupted
}
