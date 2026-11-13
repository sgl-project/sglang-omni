package conformance

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/hehehai/actnowkit/credentials"
	"github.com/hehehai/actnowkit/family"
	"github.com/hehehai/actnowkit/models"
	"github.com/hehehai/actnowkit/registry"
	famruntime "github.com/hehehai/actnowkit/runtime"
	"github.com/hehehai/actnowkit/settings"
)

// actnowkit passes its own suite.
func TestKit(t *testing.T) { Run(t, Kit) }

// pointer resolves a JSON pointer in decoded JSON.
func pointer(v any, p string) (any, bool) {
	for _, part := range strings.Split(strings.TrimPrefix(p, "/"), "/") {
		switch x := v.(type) {
		case map[string]any:
			var ok bool
			if v, ok = x[part]; !ok {
				return nil, false
			}
		case []any:
			i, err := strconv.Atoi(part)
			if err != nil || i >= len(x) {
				return nil, false
			}
			v = x[i]
		default:
			return nil, false
		}
	}
	return v, true
}

// target is the actnowkit type of each kind, for the cases.
func target(kind string) (any, int) {
	switch kind {
	case family.KindFamily:
		return &family.Info{}, 1
	case registry.Kind:
		return &registry.Manifest{}, 1
	case models.KindClaims:
		return &models.Claims{}, 1
	case settings.Kind:
		return &settings.Settings{}, 1
	case credentials.Kind:
		return &credentials.File{}, 1
	case models.KindRemoved:
		return &models.Removed{}, 1
	case models.KindLicences:
		return &models.Licences{}, 1
	case famruntime.Kind:
		return &famruntime.File{}, 1
	}
	return nil, 0
}

// The sample files read as cases.json says, and a decode/encode round
// trip through actnowkit's types keeps every member.
func TestCases(t *testing.T) {
	cases, err := Cases()
	if err != nil || len(cases) < 10 {
		t.Fatal(len(cases), err)
	}
	for _, c := range cases {
		t.Run(c.File, func(t *testing.T) {
			data, err := File(c.File)
			if err != nil {
				t.Fatal(err)
			}
			v, major := target(c.Kind)
			if v == nil {
				t.Fatalf("no type for kind %q", c.Kind)
			}
			err = family.Decode(c.File, data, c.Kind, major, v)
			var ce *family.CorruptError
			got := "ok"
			switch {
			case errors.Is(err, family.ErrNewerSchema):
				got = "readOnly"
			case errors.As(err, &ce):
				got = "corrupt"
			case errors.Is(err, family.ErrWrongKind):
				got = "wrongKind"
			case err != nil:
				t.Fatal(err)
			}
			if got != c.Result {
				t.Fatalf("result %s, want %s (%v)", got, c.Result, err)
			}
			if got != "ok" && got != "readOnly" {
				return
			}
			out, err := json.Marshal(v)
			if err != nil {
				t.Fatal(err)
			}
			var orig, again any
			json.Unmarshal(data, &orig)
			json.Unmarshal(out, &again)
			for p, want := range c.Expect {
				have, ok := pointer(again, p)
				if !ok || fmt.Sprint(have) != fmt.Sprint(want) {
					t.Errorf("%s = %v (%v), want %v", p, have, ok, want)
				}
			}
			for _, p := range c.Keep {
				w, _ := pointer(orig, p)
				h, ok := pointer(again, p)
				if !ok || !reflect.DeepEqual(w, h) {
					t.Errorf("%s not kept: %v → %v", p, w, h)
				}
			}
			if got == "ok" && !reflect.DeepEqual(orig, again) {
				t.Errorf("round trip changed the file:\n%s\n%s", data, out)
			}
		})
	}
}

func TestCheck(t *testing.T) {
	fx := New(t)
	ctx := t.Context()
	path := fx.Register(PeerSmith)
	fx.Install(repoASR)
	must(t, fx.Peer(PeerSmith).Claim(ctx, models.Claim{Repo: repoASR.Ref.Repo, Revision: repoASR.Ref.Revision}))
	if p := Check(fx.Family, fx.Hub, FixtureSystem{}); len(p) != 0 {
		t.Fatalf("clean directory: %v", p)
	}
	_ = path
	os.WriteFile(fx.Family.SettingsFile(), []byte("{"), 0o644)
	os.WriteFile(fx.Family.ClaimsFile("dev.hehehai.auralink"), []byte(`{"schema":"actnow.models.claims/1","bundleId":"dev.hehehai.auralink","root":"/elsewhere","claims":[{"repo":"../x","revision":"r"}]}`), 0o600)
	os.Chmod(fx.Family.AppFile(PeerSmith), 0o644)
	os.WriteFile(filepath.Join(fx.Family.Root, "secret.txt"), []byte("sk-0123456789abcdefghijklmnop"), 0o600)
	problems := Check(fx.Family, fx.Hub, FixtureSystem{})
	text := fmt.Sprint(problems)
	for _, want := range []string{"settings.json", "../x", "0644", "secret"} {
		if !strings.Contains(text, want) {
			t.Errorf("no problem mentions %q: %v", want, problems)
		}
	}
}

// naiveApp is a pre-spec app: it claims nothing and deletes whole repo
// directories. The suite must catch it.
type naiveApp struct{ fx *Fixture }

func (n naiveApp) BundleID() string                           { return AppUnderTest }
func (n naiveApp) Reuse(context.Context, ...models.Ref) error { return nil }
func (n naiveApp) Remove(_ context.Context, r models.Ref, _ bool) error {
	return os.RemoveAll(models.RepoDir(n.fx.Hub, r.Repo))
}

// v01App removes a model the way actnowkit v0.1 did: the app's whole
// claim on each repo goes, even when another of its models uses it.
type v01App struct{ *KitApp }

func (a v01App) RemoveModel(ctx context.Context, model string, refs ...models.Ref) error {
	for _, r := range refs {
		if _, err := a.store.Delete(ctx, r, models.ThisApp); err != nil {
			return err
		}
	}
	return nil
}

// recorder runs a scenario and records whether it failed.
type recorder struct {
	testing.TB
	mu     sync.Mutex
	failed bool
	logs   []string
}

func (r *recorder) note(args ...any) {
	r.mu.Lock()
	r.failed = true
	r.logs = append(r.logs, fmt.Sprint(args...))
	r.mu.Unlock()
}
func (r *recorder) Error(args ...any)                 { r.note(args...) }
func (r *recorder) Errorf(format string, args ...any) { r.note(fmt.Sprintf(format, args...)) }
func (r *recorder) Fatal(args ...any)                 { r.note(args...); runtime.Goexit() }
func (r *recorder) Fatalf(format string, args ...any) {
	r.note(fmt.Sprintf(format, args...))
	runtime.Goexit()
}
func (r *recorder) FailNow()     { r.note("FailNow"); runtime.Goexit() }
func (r *recorder) Fail()        { r.note("Fail") }
func (r *recorder) Failed() bool { r.mu.Lock(); defer r.mu.Unlock(); return r.failed }

func TestSuiteCatchesV01SharedAux(t *testing.T) {
	sc := Scenarios[slices.IndexFunc(Scenarios, func(s Scenario) bool { return s.Name == "SharedRepoWithinApp" })]
	rec := &recorder{TB: t}
	done := make(chan struct{})
	go func() {
		defer close(done)
		fx := New(rec)
		sc.Run(rec, fx, func(tb testing.TB, fx *Fixture, bid string) any { return v01App{Kit(tb, fx, bid).(*KitApp)} })
	}()
	<-done
	if !rec.Failed() {
		t.Fatal("the scenario passed v0.1's whole-claim removal")
	}
	t.Logf("caught: %s", rec.logs[0])
}

func TestSuiteCatchesNaiveApp(t *testing.T) {
	mustFail := map[string]bool{"ReuseWithoutDownload": true, "DeleteKeepsSharedModel": true, "DeleteWhileDownloading": true, "DeleteWhileLoaded": true, "DeleteForAllApps": true}
	for _, sc := range Scenarios {
		if !mustFail[sc.Name] {
			continue
		}
		t.Run(sc.Name, func(t *testing.T) {
			rec := &recorder{TB: t}
			done := make(chan struct{})
			go func() {
				defer close(done)
				fx := New(rec)
				sc.Run(rec, fx, func(_ testing.TB, fx *Fixture, _ string) any { return naiveApp{fx} })
			}()
			<-done
			if !rec.Failed() {
				t.Errorf("the scenario passed a naive app")
			} else {
				t.Logf("caught: %s", rec.logs[0])
			}
		})
	}
}
