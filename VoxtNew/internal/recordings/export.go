package recordings

// Exporting recordings: copies under readable names, a folder per
// feature as Voxt's export (transcription/, translation/, rewrite/),
// each file named by when it was said and its first words.

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"unicode"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

// nameWords is how many characters of the text a name keeps at most.
const nameWords = 32

// kindFolder is the folder of a feature's recordings in an export
// (Voxt's names).
func kindFolder(k store.Kind) string {
	switch k {
	case store.KindTranslate:
		return "translation"
	case store.KindRewrite:
		return "rewrite"
	}
	return "transcription"
}

// ExportName is where an entry's recording goes in an export, relative
// to its folder: its feature's folder, then the local date and time
// (Voxt's yyyyMMdd-HHmmss) and the first words of its text, letters and
// digits only (Voxt's sanitising: anything else is "-"), or the feature
// without a text: "transcription/20261017-143005-Lets-move-the-design.wav".
func ExportName(e store.Entry) string {
	words := ""
	for _, s := range []string{e.Result, e.RawText, e.PromptText, e.SourceText} {
		if words = sanitize(s); words != "" {
			break
		}
	}
	if words == "" {
		words = kindFolder(e.Kind)
	}
	return kindFolder(e.Kind) + "/" + e.CreatedAt.Local().Format("20060102-150405") + "-" + words + ".wav"
}

// sanitize keeps the start of s as a file name's words: letters and
// digits, runs of anything else as one "-", at most nameWords
// characters, cut after a whole word when one ends in the second half.
func sanitize(s string) string {
	var b []rune
	dash := false
	for _, r := range strings.TrimSpace(s) {
		switch {
		case r == '\'' || r == '’':
			continue // "Let's" is "Lets"
		case unicode.IsLetter(r) || unicode.IsDigit(r):
			if dash && len(b) > 0 {
				b = append(b, '-')
			}
			b, dash = append(b, r), false
		default:
			dash = true
		}
		if len(b) > nameWords {
			break
		}
	}
	if len(b) > nameWords {
		cut := nameWords
		for i := nameWords; i > nameWords/2; i-- {
			if b[i] == '-' {
				cut = i
				break
			}
		}
		b = b[:cut]
	}
	return strings.Trim(string(b), "-")
}

// ExportResult is what an export did: the recordings copied, the
// entries whose recording was not found, and the copies that failed.
type ExportResult struct {
	Exported, Skipped, Failed int
}

// Export copies the recordings of entries into dir (ExportName; a name
// taken by another file gets "-2", "-3"; the same recording exported
// before is left as it is), checking each copy. progress gets the
// entries done and their number. It stops at a cancel; other failures
// count in Failed and the export goes on.
func Export(ctx context.Context, hs store.History, entries []store.Entry, dir string, progress func(done, total int)) (ExportResult, error) {
	var res ExportResult
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return res, err
	}
	used := map[string]bool{}
	for i, e := range entries {
		if progress != nil {
			progress(i, len(entries))
		}
		if err := ctx.Err(); err != nil {
			return res, err
		}
		if e.Audio == "" || filepath.Base(e.Audio) != e.Audio {
			continue
		}
		src := hs.AudioPath(e.Audio)
		if fi, err := os.Stat(src); err != nil || !fi.Mode().IsRegular() {
			res.Skipped++
			continue
		}
		dst, same := freeName(dir, ExportName(e), src, used)
		used[dst] = true
		if same {
			res.Exported++
			continue
		}
		if err := exportOne(src, dst); err != nil {
			res.Failed++
			continue
		}
		res.Exported++
	}
	if progress != nil {
		progress(len(entries), len(entries))
	}
	return res, nil
}

// freeName is the path of name in dir that no other file takes, and
// whether it holds src's bytes already.
func freeName(dir, name, src string, used map[string]bool) (path string, same bool) {
	base := filepath.Join(dir, filepath.FromSlash(strings.TrimSuffix(name, ".wav")))
	for n := 1; ; n++ {
		p := base + ".wav"
		if n > 1 {
			p = fmt.Sprintf("%s-%d.wav", base, n)
		}
		if used[p] {
			continue
		}
		if _, err := os.Lstat(p); err != nil {
			return p, false
		}
		if sameFile(src, p) {
			return p, true
		}
	}
}

// exportOne copies src to dst through a temporary file.
func exportOne(src, dst string) error {
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	tmp := dst + ".part"
	if _, err := copyFile(src, tmp); err != nil {
		os.Remove(tmp)
		return err
	}
	if err := os.Rename(tmp, dst); err != nil {
		os.Remove(tmp)
		return err
	}
	return nil
}
