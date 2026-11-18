package terminal

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"

	"github.com/egoist/mygo/ui"
)

// GhosttyTheme returns a theme of Ghostty's by its name, as Ghostty's
// theme option takes it ("Catppuccin Mocha", "Dracula", "Rose Pine
// Dawn"…): a theme file of the user's in Ghostty's themes directory
// ($XDG_CONFIG_HOME/ghostty/themes, ~/.config/ghostty/themes by
// default), else one of the themes Ghostty ships, which the plugin
// carries, of any case; or the theme file a path names. Each call returns
// a new Theme.
func GhosttyTheme(name string) (*Theme, error) {
	if filepath.IsAbs(name) {
		return readGhosttyTheme(name)
	}
	if name == "" || strings.ContainsAny(name, `/\`) || name == "." || name == ".." {
		return nil, fmt.Errorf("terminal: %q names no theme of Ghostty's", name)
	}
	if dir := ghosttyThemesDir(); dir != "" {
		th, err := readGhosttyTheme(filepath.Join(dir, name))
		if !errors.Is(err, fs.ErrNotExist) {
			return th, err
		}
	}
	i := slices.IndexFunc(ghosttyThemes[:], func(t ghosttyTheme) bool { return t.name == name })
	if i < 0 {
		i = slices.IndexFunc(ghosttyThemes[:], func(t ghosttyTheme) bool { return strings.EqualFold(t.name, name) })
	}
	if i < 0 {
		return nil, fmt.Errorf("terminal: Ghostty has no theme %q", name)
	}
	return ghosttyThemes[i].theme(), nil
}

// GhosttyThemes returns the names GhosttyTheme takes, sorted: those of the
// user's theme files and of the themes Ghostty ships.
func GhosttyThemes() []string {
	names := make([]string, 0, len(ghosttyThemes))
	for _, t := range ghosttyThemes {
		names = append(names, t.name)
	}
	if dir := ghosttyThemesDir(); dir != "" {
		entries, _ := os.ReadDir(dir)
		for _, e := range entries {
			if !e.IsDir() && !strings.HasPrefix(e.Name(), ".") {
				names = append(names, e.Name())
			}
		}
	}
	slices.SortFunc(names, func(a, b string) int {
		if c := strings.Compare(strings.ToLower(a), strings.ToLower(b)); c != 0 {
			return c
		}
		return strings.Compare(a, b)
	})
	return slices.Compact(names)
}

// ghosttyThemesDir returns the directory of the user's themes of Ghostty,
// as Ghostty finds it.
func ghosttyThemesDir() string {
	dir := os.Getenv("XDG_CONFIG_HOME")
	if dir == "" && runtime.GOOS == "windows" {
		dir = os.Getenv("LOCALAPPDATA")
	}
	if dir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return ""
		}
		dir = filepath.Join(home, ".config")
	}
	return filepath.Join(dir, "ghostty", "themes")
}

func readGhosttyTheme(path string) (*Theme, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("terminal: %w", err)
	}
	th, err := ParseGhosttyTheme(data)
	if err != nil {
		return nil, fmt.Errorf("%w in %s", err, path)
	}
	return th, nil
}

// ParseGhosttyTheme reads a theme in Ghostty's format, Ghostty's
// configuration's lines that set colors:
//
//	palette = 0=#45475a
//	…
//	palette = 15=#a6adc8
//	background = #1e1e2e
//	foreground = #cdd6f4
//	cursor-color = #f5e0dc
//	cursor-text = #1e1e2e
//	selection-background = #f5e0dc
//	selection-foreground = #1e1e2e
//
// Colors are in hex (#rgb, #rrggbb, rrggbb…). Other lines of Ghostty's
// configuration are ignored, as are the palette's colors past the 16th,
// and the colors set to cell-foreground or cell-background, which the
// terminal then picks itself (see Theme). The colors a theme leaves out
// are those of DarkTheme, or LightTheme when its background is light.
func ParseGhosttyTheme(data []byte) (*Theme, error) {
	var (
		th  Theme
		set = map[string]bool{}
	)
	sc := bufio.NewScanner(bytes.NewReader(data))
	for n := 1; sc.Scan(); n++ {
		line := strings.TrimSpace(sc.Text())
		if line == "" || line[0] == '#' {
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			return nil, fmt.Errorf("terminal: line %d of the theme sets nothing", n)
		}
		key, value = strings.TrimSpace(key), strings.TrimSpace(value)
		if len(value) >= 2 && value[0] == '"' && value[len(value)-1] == '"' {
			value = value[1 : len(value)-1]
		}
		var dst *ui.Color
		switch key {
		case "background":
			dst = &th.Background
		case "foreground":
			dst = &th.Foreground
		case "cursor-color":
			dst = &th.Cursor
		case "cursor-text":
			dst = &th.CursorText
		case "selection-background":
			dst = &th.Selection
		case "selection-foreground":
			dst = &th.SelectionText
		case "palette":
			i, c, ok := strings.Cut(value, "=")
			index, err := strconv.ParseUint(strings.TrimSpace(i), 0, 8)
			if !ok || err != nil {
				return nil, fmt.Errorf("terminal: line %d of the theme: %q is no palette entry, as 0=#000000", n, value)
			}
			if index >= 16 {
				continue
			}
			dst, value, key = &th.Palette[index], strings.TrimSpace(c), "palette"+strconv.Itoa(int(index))
		default:
			continue
		}
		if value == "" || value == "cell-foreground" || value == "cell-background" {
			*dst, set[key] = ui.Color{}, false
			continue
		}
		c, ok := parseHexColor(value)
		if !ok {
			return nil, fmt.Errorf("terminal: line %d of the theme: %q is no color in hex", n, value)
		}
		*dst, set[key] = c, true
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	def := DarkTheme()
	if set["background"] && !th.dark() {
		def = LightTheme()
	}
	if !set["background"] {
		th.Background = def.Background
	}
	if !set["foreground"] {
		th.Foreground = def.Foreground
	}
	for i := range th.Palette {
		if !set["palette"+strconv.Itoa(i)] {
			th.Palette[i] = def.Palette[i]
		}
	}
	return &th, nil
}

// parseHexColor parses a color as Ghostty does in hex: #rgb, #rrggbb,
// #rrrgggbbb or #rrrrggggbbbb, or the first two without #.
func parseHexColor(s string) (ui.Color, bool) {
	hex, ok := strings.CutPrefix(s, "#")
	if !ok && len(hex) != 3 && len(hex) != 6 {
		return ui.Color{}, false
	}
	if len(hex) == 0 || len(hex)%3 != 0 || len(hex) > 12 {
		return ui.Color{}, false
	}
	w := len(hex) / 3
	var rgb [3]uint8
	for i := range rgb {
		v, err := strconv.ParseUint(hex[i*w:(i+1)*w], 16, 16)
		if err != nil {
			return ui.Color{}, false
		}
		rgb[i] = uint8(v * 255 / (1<<(4*w) - 1))
	}
	return ui.RGB(rgb[0], rgb[1], rgb[2]), true
}

// ghosttyTheme is a theme Ghostty ships: the palette's 16 colors, then the
// background, the foreground, the cursor's color and text, and the
// selection's background and text, as 0xRRGGBB.
type ghosttyTheme struct {
	name   string
	colors [22]uint32
}

func (t *ghosttyTheme) theme() *Theme {
	c := func(i int) ui.Color {
		v := t.colors[i]
		return ui.RGB(uint8(v>>16), uint8(v>>8), uint8(v))
	}
	th := &Theme{
		Background: c(16), Foreground: c(17),
		Cursor: c(18), CursorText: c(19),
		Selection: c(20), SelectionText: c(21),
	}
	for i := range th.Palette {
		th.Palette[i] = c(i)
	}
	return th
}
