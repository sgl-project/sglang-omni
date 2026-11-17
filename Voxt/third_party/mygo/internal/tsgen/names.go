package tsgen

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// reserved lists JavaScript/TypeScript words that cannot be used as
// parameter or binding names.
var reserved = map[string]bool{
	"arguments": true, "await": true, "break": true, "case": true, "catch": true,
	"class": true, "const": true, "continue": true, "debugger": true, "default": true,
	"delete": true, "do": true, "else": true, "enum": true, "eval": true,
	"export": true, "extends": true, "false": true, "finally": true, "for": true,
	"function": true, "if": true, "implements": true, "import": true, "in": true,
	"instanceof": true, "interface": true, "let": true, "new": true, "null": true,
	"package": true, "private": true, "protected": true, "public": true,
	"return": true, "static": true, "super": true, "switch": true, "this": true,
	"throw": true, "true": true, "try": true, "typeof": true, "var": true,
	"void": true, "while": true, "with": true, "yield": true,
}

// IsIdentifier reports whether s is a valid JavaScript identifier made of
// ASCII letters, digits, '_' and '$'.
func IsIdentifier(s string) bool {
	if s == "" {
		return false
	}
	for i, r := range s {
		switch {
		case r == '_' || r == '$' || ('a' <= r && r <= 'z') || ('A' <= r && r <= 'Z'):
		case i > 0 && '0' <= r && r <= '9':
		default:
			return false
		}
	}
	return true
}

// safeName makes s usable as a parameter name.
func safeName(s string) string {
	if !IsIdentifier(s) {
		s = strings.Map(func(r rune) rune {
			if r == '_' || r == '$' || unicode.IsLetter(r) || unicode.IsDigit(r) {
				return r
			}
			return '_'
		}, s)
		if s == "" || unicode.IsDigit([]rune(s)[0]) {
			s = "_" + s
		}
	}
	if reserved[s] {
		s += "_"
	}
	return s
}

// propertyName renders an object property name, quoting it when needed.
func propertyName(s string) string {
	if IsIdentifier(s) {
		return s
	}
	return quote(s)
}

// camel converts a Go identifier to lower camel case: "GetUserByID" becomes
// "getUserByID", "URLFor" becomes "urlFor" and "ID" becomes "id".
func camel(s string) string {
	runes := []rune(s)
	n := 0
	for n < len(runes) && unicode.IsUpper(runes[n]) {
		n++
	}
	switch {
	case n == 0:
		return s
	case n == 1 || n == len(runes):
		// "Greet" -> "greet", "ID" -> "id"
		for i := 0; i < n; i++ {
			runes[i] = unicode.ToLower(runes[i])
		}
	default:
		// "URLFor" -> "urlFor": keep the last upper case letter, which
		// starts the next word.
		for i := 0; i < n-1; i++ {
			runes[i] = unicode.ToLower(runes[i])
		}
	}
	return string(runes)
}

// eventKey converts an event name such as "file-changed" or "app:ready" to
// a property name like "fileChanged" / "appReady".
func eventKey(name string) string {
	var b strings.Builder
	upper := false
	for i, r := range name {
		if r == '-' || r == '_' || r == ':' || r == '.' || r == ' ' || r == '/' {
			upper = b.Len() > 0
			continue
		}
		if upper {
			r = unicode.ToUpper(r)
			upper = false
		} else if i == 0 {
			r = unicode.ToLower(r)
		}
		b.WriteRune(r)
	}
	key := b.String()
	if !IsIdentifier(key) {
		return quote(name)
	}
	return key
}

// typeName turns a reflect type name, possibly with type arguments such as
// "Page[github.com/x/app.User]", into a TypeScript identifier ("PageUser").
func typeName(name string) string {
	var b strings.Builder
	var part strings.Builder
	flush := func() {
		p := part.String()
		part.Reset()
		if i := strings.LastIndexAny(p, "./"); i >= 0 {
			p = p[i+1:]
		}
		if p == "" {
			return
		}
		r, size := utf8.DecodeRuneInString(p)
		b.WriteRune(unicode.ToUpper(r))
		b.WriteString(p[size:])
	}
	for _, r := range name {
		switch r {
		case '[', ']', ',', ' ', '*':
			flush()
		default:
			part.WriteRune(r)
		}
	}
	flush()
	return safeName(b.String())
}

func quote(s string) string {
	var b strings.Builder
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"', '\\':
			b.WriteByte('\\')
			b.WriteRune(r)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		case ' ':
			b.WriteString(` `)
		case ' ':
			b.WriteString(` `)
		default:
			if r < 0x20 {
				b.WriteString(`\u00`)
				b.WriteByte("0123456789abcdef"[r>>4])
				b.WriteByte("0123456789abcdef"[r&0xf])
			} else {
				b.WriteRune(r)
			}
		}
	}
	b.WriteByte('"')
	return b.String()
}
