package secrets

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
)

func TestItems(t *testing.T) {
	saved := security
	defer func() { security = saved }()
	var stdin string
	var args []string
	security = func(in string, a ...string) ([]byte, error) {
		stdin, args = in, a
		if a[0] == "find-generic-password" {
			return []byte("sk-fromkeychain\n"), nil
		}
		return nil, nil
	}
	k := LLM("deepseek", "Voxt DeepSeek API key")
	if k.Service != "Voxt" || k.Account != "llm.deepseek" {
		t.Fatalf("items: %+v", k)
	}
	if err := k.Set(" sk-abc "); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(args, []string{"-i"}) || !strings.Contains(stdin, `-s "Voxt" -a "llm.deepseek" -l "Voxt DeepSeek API key" -w "sk-abc"`) ||
		strings.Contains(strings.Join(args, " "), "sk-abc") {
		t.Errorf("set: args %q stdin %q", args, stdin)
	}
	if err := k.Set(`bad"key`); err == nil {
		t.Error("accepted a quote")
	}
	if got, err := k.Get(); got != "sk-fromkeychain" || err != nil {
		t.Errorf("get: %q %v", got, err)
	}
	k.Set("")
	if args[0] != "delete-generic-password" {
		t.Errorf("empty value deletes: %q", args)
	}
	security = func(string, ...string) ([]byte, error) { return nil, ErrNoKeychain }
	if got, err := k.Get(); got != "" || err != nil {
		t.Errorf("no keychain: %q %v", got, err)
	}
	if err := k.Delete(); err != nil {
		t.Errorf("delete without a keychain: %v", err)
	}
	if Mask("sk-abcdefghijkl") != "sk-…ijkl" || Mask("short") != "•••••" {
		t.Error(Mask("sk-abcdefghijkl"))
	}
}

// TestMacKeychain saves, reads, replaces and removes a secret through
// /usr/bin/security. It runs with VOXTNEW_MAC=1 on macOS:
//
//	VOXTNEW_MAC=1 go test -run TestMacKeychain -v ./internal/secrets
func TestMacKeychain(t *testing.T) {
	if os.Getenv("VOXTNEW_MAC") == "" || runtime.GOOS != "darwin" {
		t.Skip("set VOXTNEW_MAC=1 on a Mac")
	}
	// The login keychain is locked in an SSH session ("User interaction is
	// not allowed"), so the commands run against a keychain of their own;
	// the app uses the login keychain, unlocked in the user's session.
	k := Item{Service: "Voxt (test)", Account: "llm.test"}
	if _, err := security("", "show-keychain-info"); err != nil && strings.Contains(err.Error(), "User interaction is not allowed") {
		file := filepath.Join(t.TempDir(), "test.keychain-db")
		if out, err := exec.Command("/usr/bin/security", "create-keychain", "-p", "test", file).CombinedOutput(); err != nil {
			t.Fatalf("create-keychain: %v %s", err, out)
		}
		defer exec.Command("/usr/bin/security", "delete-keychain", file).Run()
		if out, err := exec.Command("/usr/bin/security", "unlock-keychain", "-p", "test", file).CombinedOutput(); err != nil {
			t.Fatalf("unlock-keychain: %v %s", err, out)
		}
		k.File = file
		t.Log("login keychain locked (SSH session): testing against a temporary keychain")
	}
	k.Delete()
	const key = "sk-probe-0123456789abcdef"
	if err := k.Set(key); err != nil {
		t.Fatalf("save: %v", err)
	}
	if got, err := k.Get(); err != nil || got != key {
		t.Fatalf("read back %q, %v", Mask(got), err)
	}
	if err := k.Set(key + "x"); err != nil {
		t.Fatalf("replace: %v", err)
	}
	if got, _ := k.Get(); got != key+"x" {
		t.Errorf("after replacing: %q", Mask(got))
	}
	if err := k.Set(""); err != nil {
		t.Fatalf("remove: %v", err)
	}
	if got, err := k.Get(); got != "" {
		t.Errorf("after removing: %q %v", Mask(got), err)
	}
}
