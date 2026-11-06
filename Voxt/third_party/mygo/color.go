package mygo

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/egoist/mygo/internal/platform"
)

// parseColor parses a CSS color: #rgb, #rgba, #rrggbb, #rrggbbaa,
// rgb(r g b), rgba(r, g, b, a) or "transparent".
func parseColor(s string) (platform.Color, error) {
	s = strings.TrimSpace(strings.ToLower(s))
	bad := fmt.Errorf("mygo: invalid color %q", s)
	switch {
	case s == "transparent":
		return platform.Color{}, nil
	case strings.HasPrefix(s, "#"):
		hex := s[1:]
		if len(hex) == 3 || len(hex) == 4 {
			var b strings.Builder
			for _, c := range hex {
				b.WriteRune(c)
				b.WriteRune(c)
			}
			hex = b.String()
		}
		if len(hex) == 6 {
			hex += "ff"
		}
		if len(hex) != 8 {
			return platform.Color{}, bad
		}
		v, err := strconv.ParseUint(hex, 16, 32)
		if err != nil {
			return platform.Color{}, bad
		}
		return platform.Color{R: uint8(v >> 24), G: uint8(v >> 16), B: uint8(v >> 8), A: uint8(v)}, nil
	case strings.HasPrefix(s, "rgb"):
		open, end := strings.IndexByte(s, '('), strings.LastIndexByte(s, ')')
		if open < 0 || end < open {
			return platform.Color{}, bad
		}
		parts := strings.FieldsFunc(s[open+1:end], func(r rune) bool { return r == ',' || r == ' ' || r == '/' })
		if len(parts) != 3 && len(parts) != 4 {
			return platform.Color{}, bad
		}
		c := platform.Color{A: 255}
		for i, p := range parts {
			v, err := colorComponent(p, i == 3)
			if err != nil {
				return platform.Color{}, bad
			}
			switch i {
			case 0:
				c.R = v
			case 1:
				c.G = v
			case 2:
				c.B = v
			case 3:
				c.A = v
			}
		}
		return c, nil
	}
	return platform.Color{}, bad
}

// background is a window background: one color, or a light and a dark one
// that follow the app's appearance.
type background struct{ light, dark platform.Color }

// parseBackground parses a CSS color or light-dark(<light>, <dark>).
func parseBackground(s string) (background, error) {
	args, ok := strings.CutPrefix(strings.ToLower(strings.TrimSpace(s)), "light-dark(")
	if !ok {
		c, err := parseColor(s)
		return background{c, c}, err
	}
	if args, ok = strings.CutSuffix(args, ")"); ok {
		// Split at the comma outside of rgb(…).
		depth := 0
		for i, r := range args {
			switch r {
			case '(':
				depth++
			case ')':
				depth--
			case ',':
				if depth == 0 {
					light, err1 := parseColor(args[:i])
					dark, err2 := parseColor(args[i+1:])
					if err1 == nil && err2 == nil {
						return background{light, dark}, nil
					}
				}
			}
		}
	}
	return background{}, fmt.Errorf("mygo: invalid color %q", s)
}

func colorComponent(s string, alpha bool) (uint8, error) {
	pct := strings.HasSuffix(s, "%")
	v, err := strconv.ParseFloat(strings.TrimSuffix(s, "%"), 64)
	if err != nil {
		return 0, err
	}
	switch {
	case pct:
		v = v / 100 * 255
	case alpha:
		v *= 255
	}
	return uint8(math.Round(min(max(v, 0), 255))), nil
}
