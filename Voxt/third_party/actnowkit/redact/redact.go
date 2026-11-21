// Package redact removes secrets from text an app logs, shows or exports
// (spec §13, v1.0.9): API keys and tokens in URL queries, signatures of
// presigned URLs, passwords in proxy and other URLs, and bearer tokens.
// Apps pass every log line, error detail and diagnostics file through
// Text (or wrap errors with Error) before writing it.
//
//	log.Print(redact.Text(err.Error()))
//	u := redact.URL("https://x.googleapis.com/v2?key=AIza…&q=hi") // …?key=REDACTED&q=hi
package redact

import (
	"net/url"
	"regexp"
	"strings"
)

// Mask replaces a secret.
const Mask = "REDACTED"

// SecretParams are the query parameters (lower case) whose values are
// secrets. Matching ignores case.
var SecretParams = []string{
	"key", "api_key", "apikey", "api-key", "token", "access_token", "refresh_token", "id_token",
	"auth", "authorization", "password", "passwd", "pwd", "secret", "client_secret",
	"sig", "signature", "x-amz-signature", "x-amz-credential", "x-amz-security-token",
	"x-goog-signature", "x-goog-credential", "pin", "code",
}

func secretParam(name string) bool {
	n := strings.ToLower(name)
	for _, p := range SecretParams {
		if n == p {
			return true
		}
	}
	return false
}

// URL redacts one URL: the password of its user info (user:pass@ →
// user:REDACTED@; a lone user, which proxies use as a token, too) and the
// values of SecretParams in its query. Text that does not parse as a URL
// goes through Text.
func URL(raw string) string {
	u, err := url.Parse(raw)
	if err != nil || u.Scheme == "" {
		return Text(raw)
	}
	changed := false
	if u.User != nil {
		if _, has := u.User.Password(); has {
			u.User = url.UserPassword(u.User.Username(), Mask)
		} else {
			u.User = url.User(Mask)
		}
		changed = true
	}
	if u.RawQuery != "" {
		if q, ok := redactQuery(u.RawQuery); ok {
			u.RawQuery, changed = q, true
		}
	}
	if !changed {
		return raw
	}
	return strings.Replace(u.String(), url.QueryEscape(Mask), Mask, -1)
}

// redactQuery masks secret values in a raw query, keeping its order.
func redactQuery(q string) (string, bool) {
	parts := strings.Split(q, "&")
	changed := false
	for i, p := range parts {
		name, _, has := strings.Cut(p, "=")
		if dn, err := url.QueryUnescape(name); err == nil {
			name = dn
		}
		if has && secretParam(name) {
			parts[i] = p[:strings.IndexByte(p, '=')+1] + Mask
			changed = true
		}
	}
	return strings.Join(parts, "&"), changed
}

var (
	urlRE    = regexp.MustCompile(`[A-Za-z][A-Za-z0-9+.-]*://[^\s"'<>]+`)
	paramRE  = regexp.MustCompile(`(?i)([?&;\s"]|^)(` + paramAlt() + `)=([^&\s"';]+)`)
	bearerRE = regexp.MustCompile(`(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}`)
	headerRE = regexp.MustCompile(`(?i)\b(authorization|x-api-key|x-goog-api-key|api-key|xi-api-key)(\s*[:=]\s*)("?)[^\s",]+`)
	// Well-known key shapes, wherever they appear.
	keyRE = regexp.MustCompile(`\b(AIza[0-9A-Za-z_-]{30,}|sk-[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|xox[abpr]-[A-Za-z0-9-]{10,})`)
)

func paramAlt() string {
	var alt []string
	for _, p := range SecretParams {
		if p == "code" || p == "pin" || p == "auth" || p == "key" {
			continue // too common as words; handled inside URLs
		}
		alt = append(alt, regexp.QuoteMeta(p))
	}
	return strings.Join(alt, "|")
}

// Text redacts every URL in s (URL), secret parameters outside URLs
// (token=…), Authorization headers and bearer tokens, and well-known key
// shapes (Google AIza…, OpenAI sk-…, Hugging Face hf_…, GitHub gh*_…,
// Slack xox*-…).
func Text(s string) string {
	if s == "" {
		return s
	}
	s = urlRE.ReplaceAllStringFunc(s, func(m string) string {
		// Trailing punctuation is not part of the URL.
		trail := ""
		for len(m) > 0 && strings.ContainsRune(".,;:)]}", rune(m[len(m)-1])) {
			trail = m[len(m)-1:] + trail
			m = m[:len(m)-1]
		}
		return URL(m) + trail
	})
	s = paramRE.ReplaceAllString(s, "${1}${2}="+Mask)
	s = headerRE.ReplaceAllString(s, "${1}${2}${3}"+Mask)
	s = bearerRE.ReplaceAllString(s, "${1} "+Mask)
	s = keyRE.ReplaceAllString(s, Mask)
	return s
}

// Error wraps err so that its message is redacted (Text); errors.Is and
// errors.As still see the original.
func Error(err error) error {
	if err == nil {
		return nil
	}
	msg := err.Error()
	if r := Text(msg); r != msg {
		return &redacted{msg: r, err: err}
	}
	return err
}

type redacted struct {
	msg string
	err error
}

func (r *redacted) Error() string { return r.msg }
func (r *redacted) Unwrap() error { return r.err }
