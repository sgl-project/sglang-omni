package credentials

import (
	"context"
	"encoding/hex"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"

	"github.com/hehehai/actnowkit/family"
)

// fakeSecurity imitates /usr/bin/security's generic-password commands.
type fakeSecurity struct {
	mu    sync.Mutex
	items map[string]string // service + "\x00" + account → secret
	reads []string          // accounts read
}

func newFake() *fakeSecurity { return &fakeSecurity{items: map[string]string{}} }

func (f *fakeSecurity) run(stdin string, args ...string) ([]byte, []byte, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(args) == 1 && args[0] == "-i" {
		args = splitQuoted(strings.TrimSpace(stdin))
	}
	opt := map[string]string{}
	for i := 1; i < len(args); i++ {
		if strings.HasPrefix(args[i], "-") && i+1 < len(args) && args[i] != "-U" && args[i] != "-g" {
			opt[args[i]] = args[i+1]
			i++
		}
	}
	key := opt["-s"] + "\x00" + opt["-a"]
	switch args[0] {
	case "add-generic-password":
		b, err := hex.DecodeString(opt["-X"])
		if err != nil {
			return nil, nil, err
		}
		f.items[key] = string(b)
		return nil, nil, nil
	case "find-generic-password":
		f.reads = append(f.reads, opt["-a"])
		v, ok := f.items[key]
		if !ok {
			return nil, nil, ErrNotFound
		}
		return nil, []byte("keychain: x\npassword: 0x" + strings.ToUpper(hex.EncodeToString([]byte(v))) + "  \"...\"\n"), nil
	case "delete-generic-password":
		if _, ok := f.items[key]; !ok {
			return nil, nil, ErrNotFound
		}
		delete(f.items, key)
		return nil, nil, nil
	}
	return nil, nil, errors.New("unknown command " + args[0])
}

func splitQuoted(s string) []string {
	var out []string
	var cur strings.Builder
	in := false
	for _, c := range s {
		switch {
		case c == '"':
			in = !in
		case c == ' ' && !in:
			if cur.Len() > 0 {
				out = append(out, cur.String())
				cur.Reset()
			}
		default:
			cur.WriteRune(c)
		}
	}
	if cur.Len() > 0 {
		out = append(out, cur.String())
	}
	return out
}

func TestVendors(t *testing.T) {
	seen := map[string]bool{}
	for _, v := range Vendors {
		if v.ID == "" || v.Name == "" || v.Auth == "" || seen[v.ID] || strings.ContainsAny(v.ID, " ./") {
			t.Errorf("bad vendor %+v", v)
		}
		seen[v.ID] = true
		for _, l := range v.Legacy {
			if l.Service == "" || l.Account == "" {
				t.Errorf("%s: bad legacy %+v", v.ID, l)
			}
		}
	}
	cases := []struct{ app, account, base, want string }{
		{"auralink", "asr.aliyun", "", "dashscope"},
		{"audiosmith", "llm.deepseek", "", "deepseek"},
		{"audiosmith", "llm.moonshot", "https://api.moonshot.ai/v1", "moonshot-intl"},
		{"audiosmith", "llm.moonshot", "https://api.moonshot.cn/v1", "moonshot"},
		{"audiosmith", "llm.siliconflow", "https://api.siliconflow.com/v1", "siliconflow-intl"},
		{"auralink", "asr.openai", "https://api.openai.com/v1", "openai"},
		{"auralink", "asr.openai", "http://127.0.0.1:8000/v1", ""},
		{"dicta", "huggingface-token", "", "huggingface"},
		{"audiosmith", "api.key", "", ""},
		{"audiosmith", "mt.google-v3", "", ""},
	}
	for _, c := range cases {
		v, ok := VendorForLegacy(c.app, c.account, c.base)
		if v.ID != c.want || ok != (c.want != "") {
			t.Errorf("%+v: %s %v", c, v.ID, ok)
		}
	}
}

func TestParsePassword(t *testing.T) {
	cases := map[string]string{
		"password: \"abcd\"\n": "abcd",
		"keychain: \"x\"\npassword: 0x736B2D61225C62  \"sk-a\"\\134b\"": `sk-a"\b`,
		"password: \n":         "",
		"password: \"a\"b\"\n": `a"b`,
	}
	for in, want := range cases {
		if got, err := parsePassword(in); err != nil || got != want {
			t.Errorf("%q: %q %v", in, got, err)
		}
	}
	if _, err := parsePassword("nothing"); err == nil {
		t.Error("no password line accepted")
	}
}

func TestHintFingerprint(t *testing.T) {
	if h := Hint("sk-0123456789a1b2"); h != "sk-…a1b2" {
		t.Error(h)
	}
	if h := Hint("hf_abcdefghijklwxyz"); h != "hf_…wxyz" {
		t.Error(h)
	}
	if h := Hint("abcdefghijklmnop"); h != "…mnop" {
		t.Error(h)
	}
	if h := Hint("short"); h != "…" {
		t.Error(h)
	}
	if f := Fingerprint("x"); !strings.HasPrefix(f, "sha256:") || len(f) != len("sha256:")+12 {
		t.Error(f)
	}
	if v, ok := ParseRef(Ref("deepseek")); !ok || v != "deepseek" || Ref("deepseek") != "actnow:vendor.deepseek" {
		t.Error(v, ok)
	}
}

func TestShareFlow(t *testing.T) {
	d := family.At(t.TempDir())
	ctx := context.Background()
	fake := newFake()
	kc := Keychain{Run: fake.run}
	auralink := &Store{Family: d, BundleID: "dev.hehehai.auralink", Keychain: kc}
	smith := &Store{Family: d, BundleID: "dev.hehehai.audiosmith", Keychain: kc}

	const key = `sk-deep"seek\key-a1b2`
	if err := auralink.Save(ctx, "deepseek", key, SaveOptions{}); err != nil {
		t.Fatal(err)
	}
	if fake.items["ActNow\x00vendor.deepseek"] != key {
		t.Fatalf("item %q", fake.items)
	}
	// The consumer sees the offer without any Keychain read.
	fake.reads = nil
	e, ok := smith.Offer("deepseek")
	if !ok || e.AddedBy != auralink.BundleID || e.Hint != "sk-…a1b2" || len(e.Users) != 1 {
		t.Fatalf("offer %+v", e)
	}
	if _, err := smith.Secret(ctx, "deepseek"); !errors.Is(err, ErrNotConsented) {
		t.Fatalf("read before consent: %v", err)
	}
	if len(fake.reads) != 0 {
		t.Fatalf("keychain read before consent: %v", fake.reads)
	}
	got, err := smith.Use(ctx, "deepseek")
	if err != nil || got != key {
		t.Fatalf("use: %q %v", got, err)
	}
	if others := smith.OtherUsers("deepseek"); len(others) != 1 || others[0] != auralink.BundleID {
		t.Error(others)
	}
	if shared, same := smith.Compare("deepseek", key); !shared || !same {
		t.Error("compare same")
	}
	if _, same := smith.Compare("deepseek", "sk-other"); same {
		t.Error("compare different")
	}
	// Stop using here: the key stays for Auralink.
	if err := smith.StopUsing(ctx, "deepseek"); err != nil {
		t.Fatal(err)
	}
	if smith.Using("deepseek") || !auralink.Using("deepseek") {
		t.Error("users after stop")
	}
	// Item deleted in Keychain Access: missing, metadata dropped.
	delete(fake.items, "ActNow\x00vendor.deepseek")
	if _, err := auralink.Secret(ctx, "deepseek"); !errors.Is(err, ErrMissing) {
		t.Errorf("missing: %v", err)
	}
	if _, ok := auralink.Offer("deepseek"); ok {
		t.Error("metadata kept")
	}
	// Delete for all.
	auralink.Save(ctx, "volcengine-speech", "token-0123456789", SaveOptions{Auth: AuthAppIDAccessToken, AppID: "123456"})
	if e, _ := auralink.Offer("volcengine-speech"); e.AppID != "123456" || e.Auth != AuthAppIDAccessToken {
		t.Errorf("%+v", e)
	}
	if err := auralink.DeleteForAll(ctx, "volcengine-speech"); err != nil {
		t.Fatal(err)
	}
	if len(fake.items) != 0 {
		t.Errorf("items left %v", fake.items)
	}
	f, _, _ := Read(d)
	if f.Rev < 5 || len(f.Vendors) != 0 || f.Schema != "actnow.credentials/1" {
		t.Errorf("%+v", f)
	}
	if err := auralink.Save(ctx, "nope", "x", SaveOptions{}); !errors.Is(err, ErrUnknownVendor) {
		t.Error(err)
	}
}

func TestMigrateLegacy(t *testing.T) {
	d := family.At(t.TempDir())
	ctx := context.Background()
	fake := newFake()
	fake.items["AudioSmith\x00llm.deepseek"] = "sk-legacy-000000a1b2"
	s := &Store{Family: d, BundleID: "dev.hehehai.audiosmith", Keychain: Keychain{Run: fake.run}}
	v, _ := VendorForLegacy("audiosmith", "llm.deepseek", "")
	if err := s.MigrateLegacy(ctx, v.ID, v.Legacy[0], SaveOptions{}); err != nil {
		t.Fatal(err)
	}
	if fake.items["ActNow\x00vendor.deepseek"] != "sk-legacy-000000a1b2" || fake.items["AudioSmith\x00llm.deepseek"] == "" {
		t.Errorf("%v", fake.items)
	}
	if !s.Using("deepseek") {
		t.Error("not a user")
	}
}

// TestMacKeychain runs the real /usr/bin/security against a temporary
// keychain file (the login keychain is locked in SSH sessions):
//
//	ACTNOWKIT_MAC=1 go test -run TestMacKeychain -v ./credentials
func TestMacKeychain(t *testing.T) {
	if os.Getenv("ACTNOWKIT_MAC") == "" || runtime.GOOS != "darwin" {
		t.Skip("set ACTNOWKIT_MAC=1 on a Mac")
	}
	file := filepath.Join(t.TempDir(), "actnowkit-test.keychain-db")
	if out, err := exec.Command("/usr/bin/security", "create-keychain", "-p", "test", file).CombinedOutput(); err != nil {
		t.Fatalf("create-keychain: %v %s", err, out)
	}
	defer exec.Command("/usr/bin/security", "delete-keychain", file).Run()
	if out, err := exec.Command("/usr/bin/security", "unlock-keychain", "-p", "test", file).CombinedOutput(); err != nil {
		t.Fatalf("unlock-keychain: %v %s", err, out)
	}
	k := Keychain{Service: "ActNow (actnowkit test)", File: file}
	if _, err := k.Get("vendor.deepseek"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("empty: %v", err)
	}
	for _, secret := range []string{`sk-plain-0123456789`, `sk-"quoted"\back slash-a1b2`, "密钥-unicode-🙂"} {
		if err := k.Set("vendor.deepseek", "ActNow DeepSeek", secret); err != nil {
			t.Fatalf("set %q: %v", secret, err)
		}
		got, err := k.Get("vendor.deepseek")
		if err != nil || got != secret {
			t.Fatalf("read back %q, want %q: %v", got, secret, err)
		}
	}
	if err := k.Delete("vendor.deepseek"); err != nil {
		t.Fatal(err)
	}
	if err := k.Delete("vendor.deepseek"); err != nil {
		t.Fatal("second delete:", err)
	}
	// The full flow through the real tool.
	d := family.At(t.TempDir())
	ctx := context.Background()
	a := &Store{Family: d, BundleID: "dev.hehehai.auralink", Keychain: k}
	b := &Store{Family: d, BundleID: "dev.hehehai.audiosmith", Keychain: k}
	if err := a.Save(ctx, "deepseek", "sk-mac-0123456789a1b2", SaveOptions{}); err != nil {
		t.Fatal(err)
	}
	if got, err := b.Use(ctx, "deepseek"); err != nil || got != "sk-mac-0123456789a1b2" {
		t.Fatalf("use: %q %v", got, err)
	}
	if err := a.DeleteForAll(ctx, "deepseek"); err != nil {
		t.Fatal(err)
	}
}
