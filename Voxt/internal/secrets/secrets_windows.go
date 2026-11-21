//go:build windows

package secrets

import (
	"errors"
	"fmt"
	"strings"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

// Windows keeps the secrets in the current user's Credential Manager
// (generic credentials "VoxtNew/<account>"), through advapi32's
// CredRead/CredWrite/CredDelete. The commands are the keychain's, so the
// rest of the package (Get, Set, Delete, their callers and tests) works
// unchanged.
var (
	advapi32  = windows.NewLazySystemDLL("advapi32.dll")
	credRead  = advapi32.NewProc("CredReadW")
	credWrite = advapi32.NewProc("CredWriteW")
	credFree  = advapi32.NewProc("CredFree")
	credDel   = advapi32.NewProc("CredDeleteW")
)

const (
	credTypeGeneric                 = 1
	credPersistLocalMachine         = 2
	errNotFound             uintptr = 1168 // ERROR_NOT_FOUND
	errNoLogonSession       uintptr = 1312 // ERROR_NO_SUCH_LOGON_SESSION
)

// notFound classifies a missing credential (a bare machine or a service
// context may answer either).
func notFound(e error) bool {
	errno, ok := e.(syscall.Errno)
	return ok && (uintptr(errno) == errNotFound || uintptr(errno) == errNoLogonSession)
}

type wincredCredential struct {
	Flags              uint32
	Type               uint32
	TargetName         *uint16
	Comment            *uint16
	LastWritten        windows.Filetime
	CredentialBlobSize uint32
	CredentialBlob     *byte
	Persist            uint32
	AttributeCount     uint32
	Attributes         *byte
	TargetAlias        *uint16
	UserName           *uint16
}

// target is the credential's name: the service and the account, so the
// keychain's "-s VoxtNew -a llm.x" becomes "VoxtNew/llm.x".
func target(service, account string) (*uint16, error) {
	return windows.UTF16PtrFromString(service + "/" + account)
}

func credGet(service, account string) (string, error) {
	t, err := target(service, account)
	if err != nil {
		return "", err
	}
	var cred *wincredCredential
	r, _, e := credRead.Call(uintptr(unsafe.Pointer(t)), credTypeGeneric, 0, uintptr(unsafe.Pointer(&cred)))
	if r == 0 {
		if notFound(e) {
			return "", errors.New("could not be found")
		}
		return "", fmt.Errorf("CredRead: %w", e)
	}
	defer credFree.Call(uintptr(unsafe.Pointer(cred)))
	return windows.UTF16ToString((*[1 << 20]uint16)(unsafe.Pointer(cred.CredentialBlob))[:cred.CredentialBlobSize/2]), nil
}

func credSet(service, account, label, value string) error {
	t, err := target(service, account)
	if err != nil {
		return err
	}
	blob := windows.StringToUTF16(value)
	c := &wincredCredential{
		Type:               credTypeGeneric,
		TargetName:         t,
		CredentialBlobSize: uint32(len(blob) * 2),
		CredentialBlob:     (*byte)(unsafe.Pointer(&blob[0])),
		Persist:            credPersistLocalMachine,
	}
	r, _, e := credWrite.Call(uintptr(unsafe.Pointer(c)), 0)
	if r == 0 {
		return fmt.Errorf("CredWrite: %w", e)
	}
	return nil
}

func credDelete(service, account string) error {
	t, err := target(service, account)
	if err != nil {
		return err
	}
	r, _, e := credDel.Call(uintptr(unsafe.Pointer(t)), credTypeGeneric, 0)
	if r == 0 {
		if notFound(e) {
			return errors.New("could not be found")
		}
		return fmt.Errorf("CredDelete: %w", e)
	}
	return nil
}

// platformSecurity answers the keychain commands the package issues
// (find/add/delete-generic-password) from the Credential Manager. The add
// command comes on stdin (interactive mode), the others as arguments.
func platformSecurity(stdin string, args ...string) ([]byte, error) {
	if len(args) > 0 && args[0] == "-i" {
		return nil, credSetCommand(stdin)
	}
	if len(args) == 0 {
		return nil, errors.New("secrets: no command")
	}
	service, account := serviceAccount(args)
	switch args[0] {
	case "find-generic-password":
		value, err := credGet(service, account)
		if err != nil {
			return nil, err
		}
		return []byte(value + "\n"), nil
	case "delete-generic-password":
		return nil, credDelete(service, account)
	}
	return nil, fmt.Errorf("secrets: no Windows equivalent of %q", args[0])
}

// serviceAccount reads the -s and -a flags of a keychain command line.
func serviceAccount(args []string) (service, account string) {
	for i, a := range args {
		if a == "-s" && i+1 < len(args) {
			service = args[i+1]
		}
		if a == "-a" && i+1 < len(args) {
			account = args[i+1]
		}
	}
	return service, account
}

// credSetCommand runs the interactive add-generic-password command:
// add-generic-password -U -s "svc" -a "acct" -l "label" -w "<value>".
func credSetCommand(cmd string) error {
	var service, account, label, value string
	fields := quotedFields(cmd)
	for i := 0; i+1 < len(fields); i++ {
		switch fields[i] {
		case "-s":
			service = fields[i+1]
		case "-a":
			account = fields[i+1]
		case "-l":
			label = fields[i+1]
		case "-w":
			value = fields[i+1]
		}
	}
	return credSet(service, account, label, value)
}

// quotedFields splits a keychain command line on spaces and quotes.
func quotedFields(cmd string) []string {
	var out []string
	var cur strings.Builder
	quoted := false
	for _, r := range cmd {
		switch {
		case r == '"':
			quoted = !quoted
		case (r == ' ' || r == '\n' || r == '\t') && !quoted:
			if cur.Len() > 0 {
				out = append(out, cur.String())
				cur.Reset()
			}
		default:
			cur.WriteRune(r)
		}
	}
	if cur.Len() > 0 {
		out = append(out, cur.String())
	}
	return out
}
