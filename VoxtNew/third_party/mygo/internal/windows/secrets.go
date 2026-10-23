//go:build windows && (amd64 || arm64)

package windows

import (
	"errors"
	"fmt"
	"syscall"
	"unsafe"

	"github.com/egoist/mygo/internal/platform"
)

// secrets keeps secrets as generic credentials of the Credential Manager,
// as keytar does: the target is "service/account", the user name the
// account, and they persist like keytar's, roaming with the user's profile
// on a domain. The system encrypts them for the user (DPAPI). The
// functions may be called from any thread.
type secrets struct{}

var (
	procCredWriteW  = advapi32.NewProc("CredWriteW")
	procCredReadW   = advapi32.NewProc("CredReadW")
	procCredDeleteW = advapi32.NewProc("CredDeleteW")
	procCredFree    = advapi32.NewProc("CredFree")
)

const (
	credTypeGeneric       = 1
	credPersistEnterprise = 3
	errorNotFound         = syscall.Errno(1168)
)

// credential is CREDENTIALW.
type credential struct {
	Flags              uint32
	Type               uint32
	TargetName         *uint16
	Comment            *uint16
	LastWritten        [2]uint32 // FILETIME
	CredentialBlobSize uint32
	CredentialBlob     *byte
	Persist            uint32
	AttributeCount     uint32
	Attributes         uintptr
	TargetAlias        *uint16
	UserName           *uint16
}

func credTarget(service, account string) *uint16 { return u16(service + "/" + account) }

func credErr(op string, err error) error {
	if errors.Is(err, errorNotFound) {
		return platform.ErrSecretNotFound
	}
	return fmt.Errorf("credential manager: %s: %w", op, err)
}

func (secrets) SetSecret(service, account string, secret []byte) error {
	c := credential{
		Type:               credTypeGeneric,
		TargetName:         credTarget(service, account),
		Persist:            credPersistEnterprise,
		UserName:           u16(account),
		CredentialBlobSize: uint32(len(secret)),
	}
	if len(secret) > 0 {
		c.CredentialBlob = &secret[0]
	}
	if r, _, err := procCredWriteW.Call(uintptr(unsafe.Pointer(&c)), 0); r == 0 {
		return credErr("write", err)
	}
	return nil
}

func (secrets) Secret(service, account string) ([]byte, error) {
	var p uintptr
	if r, _, err := procCredReadW.Call(uintptr(unsafe.Pointer(credTarget(service, account))), credTypeGeneric, 0, uintptr(unsafe.Pointer(&p))); r == 0 {
		return nil, credErr("read", err)
	}
	defer procCredFree.Call(p)
	c := (*credential)(native(p))
	if c.CredentialBlobSize == 0 || c.CredentialBlob == nil {
		return []byte{}, nil
	}
	return append([]byte(nil), unsafe.Slice(c.CredentialBlob, c.CredentialBlobSize)...), nil
}

func (secrets) DeleteSecret(service, account string) error {
	if r, _, err := procCredDeleteW.Call(uintptr(unsafe.Pointer(credTarget(service, account))), credTypeGeneric, 0); r == 0 {
		if err := credErr("delete", err); !errors.Is(err, platform.ErrSecretNotFound) {
			return err
		}
	}
	return nil
}
