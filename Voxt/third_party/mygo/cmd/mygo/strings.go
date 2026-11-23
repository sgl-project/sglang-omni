package main

import (
	"bytes"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf16"
)

// parseStrings reads a .strings file, as an app ships an InfoPlist.strings among its resources:
// the text format Xcode writes, "key" = "value"; in UTF-8 or UTF-16 with a byte order mark, or a
// property list of strings in XML.
func parseStrings(b []byte) (map[string]string, error) {
	switch {
	case bytes.HasPrefix(b, []byte{0xFF, 0xFE}), bytes.HasPrefix(b, []byte{0xFE, 0xFF}):
		big := b[0] == 0xFE
		b = b[2:]
		u := make([]uint16, len(b)/2)
		for i := range u {
			if big {
				u[i] = uint16(b[2*i])<<8 | uint16(b[2*i+1])
			} else {
				u[i] = uint16(b[2*i+1])<<8 | uint16(b[2*i])
			}
		}
		b = []byte(string(utf16.Decode(u)))
	case bytes.HasPrefix(b, []byte("bplist")):
		return nil, errors.New("a binary property list: write it as text")
	}
	b = bytes.TrimPrefix(b, []byte("\xEF\xBB\xBF"))
	if t := bytes.TrimSpace(b); bytes.HasPrefix(t, []byte("<?xml")) || bytes.HasPrefix(t, []byte("<plist")) {
		return parseStringsXML(t)
	}
	p := stringsParser{s: []rune(string(b)), line: 1}
	return p.parse()
}

// parseStringsXML reads a property list of a dictionary of strings.
func parseStringsXML(b []byte) (map[string]string, error) {
	var plist struct {
		Dict struct {
			Items []struct {
				XMLName xml.Name
				Text    string `xml:",chardata"`
			} `xml:",any"`
		} `xml:"dict"`
	}
	d := xml.NewDecoder(bytes.NewReader(b))
	// The text is UTF-8 by now, whatever encoding its declaration names (UTF-16 is decoded above).
	d.CharsetReader = func(_ string, r io.Reader) (io.Reader, error) { return r, nil }
	if err := d.Decode(&plist); err != nil {
		return nil, err
	}
	out := map[string]string{}
	items := plist.Dict.Items
	for i := 0; i+1 < len(items); i += 2 {
		if items[i].XMLName.Local != "key" || items[i+1].XMLName.Local != "string" {
			return nil, fmt.Errorf("%s after a key: only strings", items[i+1].XMLName.Local)
		}
		out[items[i].Text] = items[i+1].Text
	}
	return out, nil
}

// stringsParser reads the text format: "key" = "value"; or "key"; for a value of the key itself,
// with /* */ and // comments, and keys and values quoted or of letters, digits and _$+/:.-.
type stringsParser struct {
	s    []rune
	i    int
	line int
}

func (p *stringsParser) parse() (map[string]string, error) {
	out := map[string]string{}
	for {
		if err := p.space(); err != nil {
			return nil, err
		}
		if p.i == len(p.s) {
			return out, nil
		}
		key, err := p.word()
		if err != nil {
			return nil, err
		}
		value := key
		if err := p.space(); err != nil {
			return nil, err
		}
		if p.next('=') {
			if err := p.space(); err != nil {
				return nil, err
			}
			if value, err = p.word(); err != nil {
				return nil, err
			}
			if err := p.space(); err != nil {
				return nil, err
			}
		}
		if !p.next(';') {
			return nil, p.errorf("a ; after %q", key)
		}
		out[key] = value
	}
}

func (p *stringsParser) errorf(format string, args ...any) error {
	return fmt.Errorf("line %d: want %s", p.line, fmt.Sprintf(format, args...))
}

func (p *stringsParser) next(r rune) bool {
	if p.i < len(p.s) && p.s[p.i] == r {
		p.i++
		return true
	}
	return false
}

// space skips spaces and comments.
func (p *stringsParser) space() error {
	for p.i < len(p.s) {
		switch {
		case p.s[p.i] == '\n':
			p.line++
			p.i++
		case unicode.IsSpace(p.s[p.i]):
			p.i++
		case p.i+1 < len(p.s) && p.s[p.i] == '/' && p.s[p.i+1] == '/':
			for p.i < len(p.s) && p.s[p.i] != '\n' {
				p.i++
			}
		case p.i+1 < len(p.s) && p.s[p.i] == '/' && p.s[p.i+1] == '*':
			j := p.i + 2
			for ; j+1 < len(p.s) && !(p.s[j] == '*' && p.s[j+1] == '/'); j++ {
				if p.s[j] == '\n' {
					p.line++
				}
			}
			if j+1 >= len(p.s) {
				return p.errorf("the end of a comment")
			}
			p.i = j + 2
		default:
			return nil
		}
	}
	return nil
}

// word reads a key or a value: quoted, or a run of letters, digits and _$+/:.-.
func (p *stringsParser) word() (string, error) {
	if !p.next('"') {
		start := p.i
		for p.i < len(p.s) && (unicode.IsLetter(p.s[p.i]) || unicode.IsDigit(p.s[p.i]) || strings.ContainsRune("_$+/:.-", p.s[p.i])) {
			p.i++
		}
		if p.i == start {
			return "", p.errorf("a quoted string")
		}
		return string(p.s[start:p.i]), nil
	}
	var b strings.Builder
	for {
		if p.i == len(p.s) {
			return "", p.errorf("the end of a string")
		}
		r := p.s[p.i]
		p.i++
		switch r {
		case '"':
			return b.String(), nil
		case '\n':
			p.line++
		case '\\':
			if p.i == len(p.s) {
				return "", p.errorf("the end of a string")
			}
			e := p.s[p.i]
			p.i++
			switch e {
			case 'n':
				r = '\n'
			case 't':
				r = '\t'
			case 'r':
				r = '\r'
			case 'a':
				r = '\a'
			case 'b':
				r = '\b'
			case 'f':
				r = '\f'
			case 'v':
				r = '\v'
			case 'U', 'u':
				if p.i+4 > len(p.s) {
					return "", p.errorf("four hexadecimal digits after \\%c", e)
				}
				n, err := strconv.ParseUint(string(p.s[p.i:p.i+4]), 16, 16)
				if err != nil {
					return "", p.errorf("four hexadecimal digits after \\%c", e)
				}
				p.i += 4
				r = rune(n)
				// A character beyond the BMP is two escapes of its UTF-16 surrogates.
				if utf16.IsSurrogate(r) && p.i+6 <= len(p.s) && p.s[p.i] == '\\' && (p.s[p.i+1] == 'U' || p.s[p.i+1] == 'u') {
					if lo, err := strconv.ParseUint(string(p.s[p.i+2:p.i+6]), 16, 16); err == nil {
						if c := utf16.DecodeRune(r, rune(lo)); c != unicode.ReplacementChar {
							r = c
							p.i += 6
						}
					}
				}
			case '0', '1', '2', '3', '4', '5', '6', '7':
				n := int(e - '0')
				for k := 0; k < 2 && p.i < len(p.s) && p.s[p.i] >= '0' && p.s[p.i] <= '7'; k++ {
					n = n*8 + int(p.s[p.i]-'0')
					p.i++
				}
				r = rune(n)
			default:
				r = e // \" \\ \' and any other character stand for themselves
			}
		}
		b.WriteRune(r)
	}
}
