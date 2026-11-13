package credentials

import (
	"bytes"
	"encoding/hex"
	"errors"
	"fmt"
	"os/exec"
	"runtime"
	"strings"
)

// Service is the Keychain service of the family's shared secrets.
const Service = "ActNow"

// Account is a vendor's account name: "vendor.deepseek".
func Account(vendor string) string { return "vendor." + vendor }

// Errors of the Keychain.
var (
	ErrNotFound   = errors.New("no such Keychain item")
	ErrNoKeychain = errors.New("no Keychain on this system")
)

// Keychain is generic-password items of the login keychain, read and
// written by /usr/bin/security as the apps already do (spec §8.2): no -T,
// so any process of the user running security can read them — the same
// protection the apps' own items have today. Every method may block:
// call them off the main thread.
type Keychain struct {
	// Service is the items' service; "" = Service ("ActNow"). Apps also
	// use a Keychain with their own service to read legacy items.
	Service string
	// File is a keychain file to use instead of the login keychain
	// (tests: the login keychain is locked in SSH sessions).
	File string
	// Run runs security with args and stdin and returns its standard
	// output and error; nil = /usr/bin/security (tests replace it). A
	// missing item is ErrNotFound.
	Run func(stdin string, args ...string) (stdout, stderr []byte, err error)
}

func (k Keychain) service() string {
	if k.Service == "" {
		return Service
	}
	return k.Service
}

func (k Keychain) run(stdin string, args ...string) ([]byte, []byte, error) {
	if k.Run != nil {
		return k.Run(stdin, args...)
	}
	return RunSecurity(stdin, args...)
}

// RunSecurity runs /usr/bin/security (the default Keychain.Run).
func RunSecurity(stdin string, args ...string) ([]byte, []byte, error) {
	if runtime.GOOS != "darwin" {
		return nil, nil, ErrNoKeychain
	}
	cmd := exec.Command("/usr/bin/security", args...)
	cmd.Stdin = strings.NewReader(stdin)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		var ee *exec.ExitError
		if errors.As(err, &ee) && ee.ExitCode() == 44 || strings.Contains(stderr.String(), "could not be found") {
			return out, stderr.Bytes(), ErrNotFound
		}
		if msg := strings.TrimSpace(stderr.String()); msg != "" {
			return out, stderr.Bytes(), fmt.Errorf("security: %s", msg)
		}
		return out, stderr.Bytes(), err
	}
	return out, stderr.Bytes(), nil
}

func (k Keychain) with(args ...string) []string {
	if k.File != "" {
		args = append(args, k.File)
	}
	return args
}

// safeWord: names passed through security's interactive mode must not
// need quoting beyond plain double quotes.
func safeWord(s string) bool { return s != "" && !strings.ContainsAny(s, "\"\\\n\r\x00") }

// Get reads a secret; ErrNotFound when there is no item. It uses -g
// rather than -w: -w prints secrets with non-printable characters (and
// quotes, backslashes, non-ASCII) as bare hex, which cannot be told apart
// from a secret made of hex digits; -g marks hex with "0x".
func (k Keychain) Get(account string) (string, error) {
	_, stderr, err := k.run("", k.with("find-generic-password", "-s", k.service(), "-a", account, "-g")...)
	if err != nil {
		return "", err
	}
	return parsePassword(string(stderr))
}

// parsePassword reads security -g's password line:
//
//	password: "printable"
//	password: 0xE5AF86…  "\345\257…"
//	password:            (empty)
func parsePassword(out string) (string, error) {
	for _, line := range strings.Split(out, "\n") {
		rest, ok := strings.CutPrefix(line, "password:")
		if !ok {
			continue
		}
		rest = strings.TrimSpace(rest)
		switch {
		case rest == "":
			return "", nil
		case strings.HasPrefix(rest, "0x"):
			h, _, _ := strings.Cut(rest[2:], " ")
			b, err := hex.DecodeString(h)
			if err != nil {
				return "", fmt.Errorf("security: bad password line: %w", err)
			}
			return string(b), nil
		case len(rest) >= 2 && rest[0] == '"' && rest[len(rest)-1] == '"':
			return rest[1 : len(rest)-1], nil
		}
		return "", errors.New("security: unreadable password line")
	}
	return "", errors.New("security: no password in the output")
}

// Set saves a secret, replacing an item with the same service and
// account. The secret goes to security's standard input (interactive
// mode), hex-encoded (-X), never on a command line other processes can
// list, and needs no escaping.
func (k Keychain) Set(account, label, secret string) error {
	if !safeWord(k.service()) || !safeWord(account) || !safeWord(label) || (k.File != "" && !safeWord(k.File)) {
		return errors.New("credentials: service, account, label or keychain name has quotes, backslashes or line breaks")
	}
	if secret == "" {
		return errors.New("credentials: empty secret")
	}
	cmd := fmt.Sprintf(`add-generic-password -U -s "%s" -a "%s" -l "%s" -X %s`, k.service(), account, label, hex.EncodeToString([]byte(secret)))
	if k.File != "" {
		cmd += ` "` + k.File + `"`
	}
	_, _, err := k.run(cmd+"\n", "-i")
	return err
}

// Delete removes an item; none is not an error.
func (k Keychain) Delete(account string) error {
	_, _, err := k.run("", k.with("delete-generic-password", "-s", k.service(), "-a", account)...)
	if errors.Is(err, ErrNotFound) {
		return nil
	}
	return err
}
