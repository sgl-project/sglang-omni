package historyapi

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// Formats of an export.
const (
	FormatJSONL    = "jsonl"
	FormatMarkdown = "md"
)

// Renderer writes rows in a format, one at a time.
type Renderer struct {
	w      io.Writer
	format string
	enc    *json.Encoder
	day    string // the Markdown section open
}

// NewRenderer writes format to w.
func NewRenderer(w io.Writer, format string) *Renderer {
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	return &Renderer{w: w, format: format, enc: enc}
}

// Row writes one row.
func (r *Renderer) Row(x Row) error {
	if r.format == FormatMarkdown {
		return r.markdown(x)
	}
	return r.enc.Encode(x)
}

// markdown groups by the day of the row's own time; a Mac moved across
// time zones keeps each row on the day it was said.
func (r *Renderer) markdown(x Row) error {
	if day := x.CreatedAt.Format("2006-01-02"); day != r.day {
		if r.day != "" {
			if _, err := io.WriteString(r.w, "\n"); err != nil {
				return err
			}
		}
		r.day = day
		if _, err := fmt.Fprintf(r.w, "## %s\n\n", day); err != nil {
			return err
		}
	}
	prefix := x.CreatedAt.Format("15:04")
	if x.App != "" {
		prefix += " · " + x.App + ":"
	} else {
		prefix += " ·"
	}
	_, err := fmt.Fprintf(r.w, "- %s %s\n", prefix, strings.ReplaceAll(x.Text, "\n", "\n  "))
	return err
}

// WriteFile exports q to dst in format and reports how many rows; the
// file appears whole or not at all, readable only by this user.
func WriteFile(ctx context.Context, h store.History, q store.HistoryQuery, selection bool, dst, format string) (int, error) {
	f, err := os.CreateTemp(filepath.Dir(dst), ".voxt-export-*") // 0600
	if err != nil {
		return 0, err
	}
	defer os.Remove(f.Name())
	bw := bufio.NewWriter(f)
	rd := NewRenderer(bw, format)
	n := 0
	err = Walk(ctx, h, q, func() bool { return selection }, func(x Row) error {
		n++
		return rd.Row(x)
	})
	if err == nil {
		err = bw.Flush()
	}
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return 0, err
	}
	return n, os.Rename(f.Name(), dst)
}
