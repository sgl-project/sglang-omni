package parser

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// GitHub's extended autolinks, as cmark-gfm finds them (autolink.c):
// www. and http(s):// links as the inline parser comes to their w and :,
// email addresses in the text once parsed.

func isHostChar(s string) bool {
	r, n := utf8.DecodeRuneInString(s)
	if n == 0 || r == utf8.RuneError && n == 1 {
		return false
	}
	if isUnicodeSpace(r) {
		return false
	}
	if r < utf8.RuneSelf {
		return !isEscapable(byte(r))
	}
	return !unicode.IsPunct(r)
}

// autolinkDelim returns where a link ends once trailing punctuation, an
// entity-like ending and unbalanced parentheses are left out.
func autolinkDelim(data string, end int) int {
	open, closing := 0, 0
	for i := 0; i < end; i++ {
		switch data[i] {
		case '<':
			end = i
		case '(':
			open++
		case ')':
			closing++
		}
	}
	for end > 0 {
		switch data[end-1] {
		case ')':
			if closing <= open {
				return end
			}
			closing--
			end--
		case '?', '!', '.', ',', ':', '*', '_', '~', '\'', '"':
			end--
		case ';':
			k := end - 2
			for k > 0 && isASCIILetter(data[k]) {
				k--
			}
			if k < end-2 && data[k] == '&' {
				end = k
			} else {
				end--
			}
		default:
			return end
		}
	}
	return end
}

// checkDomain returns the length of the domain at the start of data, 0
// if it has an underscore in its last two segments, or, unless short,
// no dot.
func checkDomain(data string, short bool) int {
	np, us1, us2 := 0, 0, 0
	i := 1
	for ; i < len(data)-1; i++ {
		if data[i] == '\\' && i < len(data)-2 {
			i++
		}
		switch {
		case data[i] == '_':
			us2++
		case data[i] == '.':
			us1, us2 = us2, 0
			np++
		case !isHostChar(data[i:]) && data[i] != '-':
			goto done
		}
	}
done:
	if (us1 > 0 || us2 > 0) && np <= 10 {
		return 0
	}
	if short || np > 0 {
		return i
	}
	return 0
}

// inBracket reports whether a link's or an image's text is open.
func (ip *inlineParser) inBracket() bool { return ip.lastB >= 0 }

// wwwLink reads a link starting with www. at pos.
func (ip *inlineParser) wwwLink() bool {
	s := ip.s
	pos := ip.pos
	if ip.inBracket() || !strings.HasPrefix(s[pos:], "www.") {
		return false
	}
	if pos > 0 && strings.IndexByte("*_~(", s[pos-1]) < 0 && !isUnicodeSpaceByte(s[pos-1]) {
		return false
	}
	data := s[pos:]
	end := checkDomain(data, false)
	if end == 0 {
		return false
	}
	for end < len(data) && !isUnicodeSpaceByte(data[end]) && data[end] != '<' {
		end++
	}
	ip.d.steps += end
	end = autolinkDelim(data, end)
	if end == 0 {
		return false
	}
	ip.addAutolink(pos, pos+end, "http://"+data[:end])
	ip.pos = pos + end
	return true
}

func isUnicodeSpaceByte(c byte) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'
}

// urlLink reads a link of http, https or ftp at the colon at pos, taking
// back the scheme from the text before it.
func (ip *inlineParser) urlLink() bool {
	s := ip.s
	pos := ip.pos
	if ip.inBracket() || !strings.HasPrefix(s[pos:], "://") {
		return false
	}
	// The scheme, from the text just added.
	last := ip.nodes[0].last
	if last < 0 {
		return false
	}
	ln := &ip.nodes[last]
	if ln.kind != Text || !ln.plain || ln.end != pos {
		return false
	}
	rewind := 0
	for rewind < len(ln.text) && isASCIILetter(ln.text[len(ln.text)-1-rewind]) {
		rewind++
	}
	ip.d.steps += rewind
	start := pos - rewind
	scheme := strings.ToLower(s[start:pos])
	if scheme != "http" && scheme != "https" && scheme != "ftp" {
		return false
	}
	if len(s) <= pos+3 || !isHostChar(s[pos+3:]) {
		return false
	}
	data := s[pos:]
	n := checkDomain(data[3:], true)
	if n == 0 {
		return false
	}
	end := 3 + n
	for end < len(data) && !isUnicodeSpaceByte(data[end]) && data[end] != '<' {
		end++
	}
	ip.d.steps += end
	end = autolinkDelim(data, end)
	if end == 0 {
		return false
	}
	// Take the scheme off the text.
	ln.end = start
	ln.text = s[ln.start:start]
	ip.addAutolink(start, pos+end, s[start:pos+end])
	ip.pos = pos + end
	return true
}

func (ip *inlineParser) addAutolink(a, b int, dest string) {
	l := ip.newNode(Link, "", a, b)
	ip.nodes[l].dest = dest
	ip.nodes[l].auto = true
	ip.appendChild(0, l)
	t := ip.newNode(Text, ip.s[a:b], a, b)
	ip.appendChild(l, t)
}

// emailLinks finds email addresses in the text below node n, outside
// links, after joining text that follows text.
func (ip *inlineParser) emailLinks(n int32, inLink bool) {
	for c := ip.nodes[n].first; c >= 0; c = ip.nodes[c].next {
		nd := &ip.nodes[c]
		switch nd.kind {
		case Link, Image:
			continue
		case Text:
			if strings.IndexByte(ip.nodes[c].text, '@') >= 0 {
				ip.splitEmails(c)
			}
		default:
			ip.emailLinks(c, inLink)
		}
	}
}

// splitEmails turns the email addresses in text node c into links.
func (ip *inlineParser) splitEmails(c int32) {
	for {
		nd := &ip.nodes[c]
		data := nd.text
		a, b, mailto, ok := findEmail(data)
		if !ok {
			ip.d.steps += len(data)
			return
		}
		ip.d.steps += b
		source := ip.s[nd.start:min(nd.end, len(ip.s))]
		off := func(i int) int { return nd.start + literalToSource(source, data, i) }
		start, end := nd.start, nd.end
		la, lb := off(a), off(b)
		dest := data[a:b]
		if mailto {
			dest = "mailto:" + dest
		}
		l := ip.newNode(Link, "", la, lb)
		ip.nodes[l].dest = dest
		ip.nodes[l].auto = true
		t := ip.newNode(Text, data[a:b], la, lb)
		ip.appendChild(l, t)
		post := ip.newNode(Text, data[b:], lb, end)
		nd = &ip.nodes[c]
		nd.text, nd.start, nd.end = data[:a], start, la
		ip.insertAfter(c, l)
		ip.insertAfter(l, post)
		c = post
	}
}

// findEmail finds the first email address in data, as cmark-gfm does:
// its bytes, and whether its link needs mailto: (not when written with
// mailto: or xmpp:).
func findEmail(data string) (a, b int, mailto, ok bool) {
	offset := 0
	for offset < len(data) {
		at := strings.IndexByte(data[offset:], '@')
		if at < 0 {
			return
		}
		maxRewind := at
		auto, xmpp := true, false
	foundAt:
		base := offset + maxRewind // the @
		rewind := 0
		for ; rewind < maxRewind; rewind++ {
			ch := data[base-rewind-1]
			if isASCIILetter(ch) || isASCIIDigit(ch) || strings.IndexByte(".+-_", ch) >= 0 {
				continue
			}
			if ch == ':' {
				if validProtocol("mailto:", data, base, rewind) {
					auto = false
					continue
				}
				if validProtocol("xmpp:", data, base, rewind) {
					auto, xmpp = false, true
					continue
				}
			}
			break
		}
		if rewind == 0 {
			offset += maxRewind + 1
			continue
		}
		np := 0
		linkEnd := 1
		for ; linkEnd < len(data)-base; linkEnd++ {
			ch := data[base+linkEnd]
			switch {
			case isASCIILetter(ch) || isASCIIDigit(ch):
			case ch == '@':
				offset += maxRewind + 1
				maxRewind = linkEnd - 1
				goto foundAt
			case ch == '.' && linkEnd < len(data)-base-1 && (isASCIILetter(data[base+linkEnd+1]) || isASCIIDigit(data[base+linkEnd+1])):
				np++
			case ch == '/' && xmpp:
			case ch != '-' && ch != '_':
				goto ended
			}
		}
	ended:
		if last := data[base+linkEnd-1]; linkEnd < 2 || np == 0 || !isASCIILetter(last) && last != '.' {
			offset += maxRewind + linkEnd
			continue
		}
		linkEnd = autolinkDelim(data[base:], linkEnd)
		if linkEnd == 0 {
			offset += maxRewind + 1
			continue
		}
		return base - rewind, base + linkEnd, auto, true
	}
	return
}

// validProtocol reports whether protocol ends rewind bytes before the @
// at base, after something other than a letter or a digit.
func validProtocol(protocol, data string, base, rewind int) bool {
	end := base - rewind
	start := end - len(protocol)
	if start < 0 || data[start:end] != protocol {
		return false
	}
	if start == 0 {
		return true
	}
	c := data[start-1]
	return !isASCIILetter(c) && !isASCIIDigit(c)
}
