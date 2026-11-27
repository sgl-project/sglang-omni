package editor

import (
	"errors"
	"fmt"
	"github.com/egoist/mygo/internal/diff"
	"strconv"
	"strings"
)

// FilePatch is the change of one file in a unified diff, as git diff and
// diff -u print them.
type FilePatch struct {
	// OldName and NewName are the file's names before and after, without
	// git's a/ and b/; OldName is empty for a file added, NewName for one
	// deleted.
	OldName, NewName string
	// OldMode and NewMode are git's file modes, as "100644", when given.
	OldMode, NewMode string
	// Binary marks a change git shows no lines of.
	Binary bool
	Hunks  []PatchHunk
}

// PatchHunk is a hunk of a FilePatch: lines from OldStart (from 1) of
// the old file, OldLines of them, and NewStart and NewLines of the new.
type PatchHunk struct {
	OldStart, OldLines int
	NewStart, NewLines int
	// Section is what follows the second @@, as the function the hunk is
	// in.
	Section string
	// Lines are the hunk's lines with their first character: a space for
	// a line of both, - for one deleted, + for one added. A line "\ No
	// newline at end of file" follows the line it is about.
	Lines []string
}

// ParsePatch reads the files a unified diff changes, in order: those of
// git diff and git format-patch, with renames, modes and new and deleted
// files, and those of diff -u.
func ParsePatch(text string) ([]FilePatch, error) {
	// Lines split at \n alone: the lines of a file with \r\n endings keep
	// their \r, as Apply needs, and only the headers lose it.
	lines := strings.Split(text, "\n")
	var files []FilePatch
	var cur *FilePatch
	start := func() *FilePatch {
		files = append(files, FilePatch{})
		return &files[len(files)-1]
	}
	for i := 0; i < len(lines); i++ {
		line := strings.TrimSuffix(lines[i], "\r")
		switch {
		case strings.HasPrefix(line, "diff --git "):
			cur = start()
			if a, b, ok := gitNames(line[len("diff --git "):]); ok {
				cur.OldName, cur.NewName = a, b
			}
		case cur != nil && strings.HasPrefix(line, "new file mode "):
			cur.OldName, cur.NewMode = "", strings.TrimPrefix(line, "new file mode ")
		case cur != nil && strings.HasPrefix(line, "deleted file mode "):
			cur.NewName, cur.OldMode = "", strings.TrimPrefix(line, "deleted file mode ")
		case cur != nil && strings.HasPrefix(line, "old mode "):
			cur.OldMode = strings.TrimPrefix(line, "old mode ")
		case cur != nil && strings.HasPrefix(line, "new mode "):
			cur.NewMode = strings.TrimPrefix(line, "new mode ")
		case cur != nil && strings.HasPrefix(line, "rename from "):
			cur.OldName = unquoteName(strings.TrimPrefix(line, "rename from "))
		case cur != nil && strings.HasPrefix(line, "rename to "):
			cur.NewName = unquoteName(strings.TrimPrefix(line, "rename to "))
		case cur != nil && strings.HasPrefix(line, "copy from "):
			cur.OldName = unquoteName(strings.TrimPrefix(line, "copy from "))
		case cur != nil && strings.HasPrefix(line, "copy to "):
			cur.NewName = unquoteName(strings.TrimPrefix(line, "copy to "))
		case cur != nil && (strings.HasPrefix(line, "Binary files ") || line == "GIT binary patch"):
			cur.Binary = true
		case strings.HasPrefix(line, "--- ") && i+1 < len(lines) && strings.HasPrefix(lines[i+1], "+++ "):
			if cur == nil || len(cur.Hunks) > 0 {
				cur = start()
			}
			if a := patchName(line[4:], "a/"); a != "" || cur.OldName == "" {
				cur.OldName = a
			}
			if b := patchName(strings.TrimSuffix(lines[i+1], "\r")[4:], "b/"); b != "" || cur.NewName == "" {
				cur.NewName = b
			}
			i++
		case strings.HasPrefix(line, "@@ "):
			if cur == nil {
				cur = start()
			}
			h, err := parseHunkHeader(line)
			if err != nil {
				return files, fmt.Errorf("line %d: %w", i+1, err)
			}
			// The hunk's lines, as many as its header counts.
			oldLeft, newLeft := h.OldLines, h.NewLines
			for i+1 < len(lines) && (oldLeft > 0 || newLeft > 0 || strings.HasPrefix(lines[i+1], "\\")) {
				l := lines[i+1]
				switch {
				case strings.HasPrefix(l, "\\"):
				case strings.HasPrefix(l, "+"):
					newLeft--
				case strings.HasPrefix(l, "-"):
					oldLeft--
				case strings.HasPrefix(l, " "), l == "", l == "\r":
					// An empty line is a blank line of both, as some tools
					// strip the space.
					if !strings.HasPrefix(l, " ") {
						l = " " + l
					}
					oldLeft--
					newLeft--
				default:
					return files, fmt.Errorf("line %d: unexpected %q in a hunk", i+2, l)
				}
				if oldLeft < 0 || newLeft < 0 {
					return files, fmt.Errorf("line %d: hunk %q has more lines than it counts", i+2, line)
				}
				h.Lines = append(h.Lines, l)
				i++
			}
			if oldLeft > 0 || newLeft > 0 {
				return files, fmt.Errorf("line %d: hunk %q ends early", i+1, line)
			}
			cur.Hunks = append(cur.Hunks, h)
		}
	}
	return files, nil
}

// gitNames splits the names of "diff --git a/x b/x".
func gitNames(s string) (string, string, bool) {
	// Names git quotes, as C strings, when they hold spaces, quotes or
	// characters past ASCII.
	if strings.HasPrefix(s, "\"") {
		q, err := strconv.QuotedPrefix(s)
		if err != nil {
			return "", "", false
		}
		a := unquoteName(q)
		b := unquoteName(strings.TrimPrefix(s[len(q):], " "))
		return strings.TrimPrefix(a, "a/"), strings.TrimPrefix(b, "b/"), true
	}
	if i := strings.Index(s, " \"b/"); i >= 0 {
		return strings.TrimPrefix(s[:i], "a/"), strings.TrimPrefix(unquoteName(s[i+1:]), "b/"), true
	}
	// The names are equal but for their prefixes unless renamed, which
	// rename lines then tell.
	if n := len(s); n%2 == 1 {
		a, b := s[:n/2], s[n/2+1:]
		if strings.TrimPrefix(a, "a/") == strings.TrimPrefix(b, "b/") {
			return strings.TrimPrefix(a, "a/"), strings.TrimPrefix(b, "b/"), true
		}
	}
	if i := strings.Index(s, " b/"); i >= 0 {
		return strings.TrimPrefix(s[:i], "a/"), s[i+3:], true
	}
	return "", "", false
}

// unquoteName unquotes a name git quoted, as "a/\346\226\207.txt".
func unquoteName(s string) string {
	if strings.HasPrefix(s, "\"") {
		if u, err := strconv.Unquote(s); err == nil {
			return u
		}
	}
	return s
}

// patchName reads the name of a --- or +++ line, "" for /dev/null.
func patchName(s, prefix string) string {
	if i := strings.IndexByte(s, '\t'); i >= 0 {
		s = s[:i]
	}
	s = strings.TrimSpace(s)
	if s == "/dev/null" {
		return ""
	}
	if uq, err := strconv.Unquote(s); err == nil {
		s = uq
	}
	return strings.TrimPrefix(s, prefix)
}

func parseHunkHeader(line string) (PatchHunk, error) {
	var h PatchHunk
	end := strings.Index(line[3:], " @@")
	if end < 0 {
		return h, errors.New("malformed hunk header")
	}
	ranges := strings.Fields(line[3 : 3+end])
	if len(ranges) != 2 || !strings.HasPrefix(ranges[0], "-") || !strings.HasPrefix(ranges[1], "+") {
		return h, errors.New("malformed hunk header")
	}
	var err error
	if h.OldStart, h.OldLines, err = hunkRange(ranges[0][1:]); err != nil {
		return h, err
	}
	if h.NewStart, h.NewLines, err = hunkRange(ranges[1][1:]); err != nil {
		return h, err
	}
	h.Section = strings.TrimSpace(line[3+end+3:])
	return h, nil
}

func hunkRange(s string) (int, int, error) {
	start, count, found := strings.Cut(s, ",")
	a, err := strconv.Atoi(start)
	if err != nil {
		return 0, 0, err
	}
	n := 1
	if found {
		if n, err = strconv.Atoi(count); err != nil {
			return 0, 0, err
		}
	}
	return a, n, nil
}

// Apply applies the patch to the old file's text and returns the new
// file's, checking that the lines of both and those deleted are there.
func (p FilePatch) Apply(old string) (string, error) {
	if p.Binary {
		return "", errors.New("ui: binary patch")
	}
	var lines []string
	eol := true
	if old != "" {
		lines = strings.Split(old, "\n")
		if lines[len(lines)-1] == "" {
			lines = lines[:len(lines)-1]
		} else {
			eol = false
		}
	}
	var out []string
	at := 0
	// Whether the new text ends with a newline: as the old does, unless a
	// hunk reaching the end says.
	newEOL, touchedEnd := eol, false
	for _, h := range p.Hunks {
		start := h.OldStart - 1
		if h.OldLines == 0 {
			start = h.OldStart
		}
		if start < at || start > len(lines) {
			return "", fmt.Errorf("ui: hunk at line %d out of place", h.OldStart)
		}
		out = append(out, lines[at:start]...)
		at = start
		lastNew := -1 // the index in h.Lines of the last line of the new text
		for k, l := range h.Lines {
			if l == "" || l[0] == '\\' {
				continue
			}
			if l[0] != '+' {
				if at >= len(lines) || lines[at] != l[1:] {
					return "", fmt.Errorf("ui: hunk at line %d does not match line %d", h.OldStart, at+1)
				}
				at++
			}
			if l[0] != '-' {
				out = append(out, l[1:])
				lastNew = k
			}
		}
		if at == len(lines) {
			touchedEnd = true
			newEOL = !(lastNew >= 0 && lastNew+1 < len(h.Lines) && strings.HasPrefix(h.Lines[lastNew+1], "\\"))
			if lastNew < 0 {
				newEOL = true
			}
		}
	}
	out = append(out, lines[at:]...)
	if !touchedEnd {
		newEOL = eol
	}
	s := strings.Join(out, "\n")
	if newEOL && len(out) > 0 {
		s += "\n"
	}
	return s, nil
}

// diffPartial maps the lines of the texts a patch shows, its hunks'
// lines only, to the lines of the files.
type diffPartial struct {
	oldStarts, newStarts []int // the line of the texts each hunk starts at
	oldFile, newFile     []int // and of the files, from 0
	headers              []string
	old, new             *Buffer // the texts made of the hunks' lines
}

// SetPatch shows a file's patch without the files: its hunks' lines,
// numbered as in the files, each hunk under its header, as Pierre's
// diffs show partial patches. Old and New become texts of those lines,
// not editable, and unchanged lines do not collapse.
func (d *DiffState) SetPatch(p FilePatch) {
	var old, nw strings.Builder
	pt := &diffPartial{}
	oldLine, newLine := 0, 0
	// The changes are the patch's runs of - and + lines, as it says them,
	// not what comparing the texts would find.
	var hunks []diff.Hunk
	run := -1
	for _, h := range p.Hunks {
		run = -1
		pt.oldStarts = append(pt.oldStarts, oldLine)
		pt.newStarts = append(pt.newStarts, newLine)
		pt.oldFile = append(pt.oldFile, max(0, h.OldStart-1))
		pt.newFile = append(pt.newFile, max(0, h.NewStart-1))
		header := fmt.Sprintf("@@ -%d,%d +%d,%d @@", h.OldStart, h.OldLines, h.NewStart, h.NewLines)
		if h.Section != "" {
			header += " " + h.Section
		}
		pt.headers = append(pt.headers, header)
		for _, l := range h.Lines {
			if l == "" || l[0] == '\\' {
				continue
			}
			switch {
			case l[0] != '-' && l[0] != '+':
				run = -1
			case run < 0:
				hunks = append(hunks, diff.Hunk{A0: oldLine, A1: oldLine, B0: newLine, B1: newLine})
				run = len(hunks) - 1
			}
			if run >= 0 {
				if l[0] == '-' {
					hunks[run].A1++
				} else {
					hunks[run].B1++
				}
			}
			text := strings.TrimSuffix(l[1:], "\r")
			if l[0] != '+' {
				if oldLine > 0 {
					old.WriteByte('\n')
				}
				old.WriteString(text)
				oldLine++
			}
			if l[0] != '-' {
				if newLine > 0 {
					nw.WriteByte('\n')
				}
				nw.WriteString(text)
				newLine++
			}
		}
	}
	d.Old, d.New = NewBuffer(old.String()), NewBuffer(nw.String())
	name := p.NewName
	if name == "" {
		name = p.OldName
	}
	if l := LanguageFor(name); l != nil {
		d.Old.SetLanguage(l)
		d.New.SetLanguage(l)
	}
	pt.old, pt.new = d.Old, d.New
	d.partial = pt
	d.Editable = false
	d.prepared = diffPrepared{}
	m := &d.model
	m.set(d.Old, d.New)
	m.oldText, m.oldVer = d.Old.snapshot()
	m.newText, m.newVer = d.New.snapshot()
	m.hunks, m.words = hunks, nil
	m.version++
}

// fileLine maps a line of a side's text to the file's.
func (pt *diffPartial) fileLine(old bool, line int) int {
	starts, file := pt.newStarts, pt.newFile
	if old {
		starts, file = pt.oldStarts, pt.oldFile
	}
	k := 0
	for k+1 < len(starts) && starts[k+1] <= line {
		k++
	}
	if k >= len(starts) {
		return line
	}
	return file[k] + line - starts[k]
}
