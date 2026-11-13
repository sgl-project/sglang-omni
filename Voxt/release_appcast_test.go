package main

// scripts/release-appcast.sh says what Voxt 1.x's in-app update (Sparkle
// 2.10) needs to install a Voxt 2 release: an appcast entry whose
// sparkle:version is above the running 1.x's CFBundleVersion, an app whose
// own CFBundleVersion is not below it (else Sparkle refuses a downgrade) and
// whose Info.plist has an EdDSA key, and the files the release uploads; and
// that Voxt 1.x on an Intel Mac, where Voxt 2 does not open, is not offered it.
// Runs on Linux: the script is plain bash, awk and Python 3 for JSON.

import (
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// voxt1Build is the CFBundleVersion of Voxt 1.16.3, the newest Voxt 1.x
// (its release.yml: 1×10⁸ + 16×10⁵ + 3×100 + 99).
const voxt1Build = 101600399

// voxt1PublicKey is the SUPublicEDKey of Voxt 1.x's Info.plist.
const voxt1PublicKey = "iMt1voRdgfUUpKxdY7DliGLaOl17gJFB8Jhlo5ox2eE="

func runReleaseAppcast(t *testing.T, stdin string, args ...string) (string, error) {
	t.Helper()
	cmd := exec.Command("bash", append([]string{"scripts/release-appcast.sh"}, args...)...)
	cmd.Stdin = strings.NewReader(stdin)
	out, err := cmd.Output()
	return string(out), err
}

func releaseAppcastJSON(t *testing.T, args ...string) map[string]string {
	t.Helper()
	out, err := runReleaseAppcast(t, "", args...)
	if err != nil {
		t.Fatalf("release-appcast.sh %v: %v", args, err)
	}
	var fields map[string]string
	if err := json.Unmarshal([]byte(out), &fields); err != nil {
		t.Fatalf("release-appcast.sh %v printed %q: %v", args, out, err)
	}
	return fields
}

func TestReleaseAppcastInfoPlistIsNewerThanVoxt1(t *testing.T) {
	// In the order Sparkle must see them; a nightly comes before its release,
	// and the highest patch and minor before the next minor and major.
	versions := []string{"2.0.1-nightly.20261107", "2.0.1", "2.0.2", "2.0.999", "2.1.0", "2.10.0", "2.999.999", "3.0.0"}
	previous := voxt1Build
	for _, version := range versions {
		plist := releaseAppcastJSON(t, "info-plist", version)
		build, err := strconv.Atoi(plist["CFBundleVersion"])
		if err != nil {
			t.Fatalf("%s: CFBundleVersion %q is not a number: Sparkle compares it with %d", version, plist["CFBundleVersion"], voxt1Build)
		}
		if build <= previous {
			t.Errorf("%s: CFBundleVersion %d is not above %d: Voxt 1.x's Sparkle refuses it as a downgrade", version, build, previous)
		}
		previous = build
		if plist["SUPublicEDKey"] != voxt1PublicKey {
			t.Errorf("%s: SUPublicEDKey %q: Voxt 1.x's Sparkle refuses an update without its EdDSA key", version, plist["SUPublicEDKey"])
		}
	}
	for _, bad := range []string{"2.0", "v2.0.1", "2.1000.0", "2.0.1000"} {
		if out, err := runReleaseAppcast(t, "", "info-plist", bad); err == nil {
			t.Errorf("info-plist %s printed %q: a version Voxt 1.x's build numbers cannot hold must fail", bad, out)
		}
	}
}

func TestReleaseAppcastEntryChannels(t *testing.T) {
	plist := releaseAppcastJSON(t, "info-plist", "2.0.1")
	for _, channel := range []string{"beta", "stable"} {
		entry := releaseAppcastJSON(t, "entry", channel, "2.0.1")
		if entry["channel"] != channel || entry["version"] != "2.0.1" {
			t.Errorf("entry %s 2.0.1: %v", channel, entry)
		}
		// voxt-appcast.yaml checks the ZIP's CFBundleVersion against it.
		if entry["sparkleVersion"] != plist["CFBundleVersion"] {
			t.Errorf("entry %s: sparkleVersion %s, the app's CFBundleVersion %s", channel, entry["sparkleVersion"], plist["CFBundleVersion"])
		}
		// Sparkle 2.10 skips an item that needs arm64 on an Intel Mac, where Voxt 2 does not open.
		if entry["hardwareRequirements"] != "arm64" {
			t.Errorf("entry %s: hardwareRequirements %q: Voxt 1.x on an Intel Mac would get a Voxt it cannot open", channel, entry["hardwareRequirements"])
		}
	}
	for _, channel := range []string{"", "nightly", "Stable"} {
		if out, err := runReleaseAppcast(t, "", "entry", channel, "2.0.1"); err == nil {
			t.Errorf("entry %q printed %q: only beta and stable are appcasts of Voxt 1.x", channel, out)
		}
	}
}

func TestVoxt2FeedChannels(t *testing.T) {
	base, err := runReleaseAppcast(t, "", "feed-base")
	if err != nil || base != "https://voxt.actnow.dev/updates/voxt2\n" {
		t.Fatalf("feed-base: %q, %v", base, err)
	}
	for channel, suffix := range map[string]string{"stable": "", "beta": "/beta"} {
		got, err := runReleaseAppcast(t, "", "feed-url", channel)
		want := base[:len(base)-1] + suffix + "/update-darwin-arm64.json\n"
		if err != nil || got != want {
			t.Errorf("feed-url %s: %q, %v; want %q", channel, got, err, want)
		}
	}
	if out, err := runReleaseAppcast(t, "", "feed-url", "nightly"); err == nil {
		t.Errorf("feed-url nightly printed %q: only beta and stable have feeds", out)
	}
}

func TestVoxt2GitHubManifest(t *testing.T) {
	manifest := `{"version":"2.1.1","url":"https://voxt.actnow.dev/updates/voxt2/voxt-2.1.1-darwin-arm64.tar.gz","size":123,"signature":"signed","deltas":[{"from":"2.1.0","url":"https://voxt.actnow.dev/updates/voxt2/voxt-2.1.0-to-2.1.1-darwin-arm64.delta","size":12,"signature":"delta"}],"previous":[{"version":"2.1.0","url":"https://github.com/sgl-project/sglang-omni/releases/download/voxt-v2.1.0/voxt-2.1.0-darwin-arm64.tar.gz","size":100,"signature":"old"}]}`
	out, err := runReleaseAppcast(t, manifest, "github-manifest", "2.1.1", "sgl-project/sglang-omni")
	if err != nil {
		t.Fatal(err)
	}
	var got struct {
		URL    string
		Deltas []struct{ URL string }
		Prev   []struct{ URL string } `json:"previous"`
	}
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatal(err)
	}
	base := "https://github.com/sgl-project/sglang-omni/releases/download/"
	if got.URL != base+"voxt-v2.1.1/voxt-2.1.1-darwin-arm64.tar.gz" || len(got.Deltas) != 1 || got.Deltas[0].URL != base+"voxt-v2.1.1/voxt-2.1.0-to-2.1.1-darwin-arm64.delta" {
		t.Errorf("current release's assets: %s", out)
	}
	if len(got.Prev) != 1 || got.Prev[0].URL != base+"voxt-v2.1.0/voxt-2.1.0-darwin-arm64.tar.gz" {
		t.Errorf("an earlier release's archive moved: %s", out)
	}
	if out, err := runReleaseAppcast(t, `{"version":"2.1.0"}`, "github-manifest", "2.1.1", "sgl-project/sglang-omni"); err == nil {
		t.Errorf("a manifest of another version printed %q", out)
	}
}

func TestReleaseAppcastAssetNames(t *testing.T) {
	// The disk image is named as on the published voxt-v2.0.0 release; the ZIP as Voxt 1.x's.
	const zip, dmg = "Voxt-2.0.1-macOS.zip", "Voxt-2.0.1-arm64.dmg"
	out, err := runReleaseAppcast(t, "", "assets", "2.0.1")
	if err != nil || out != zip+" "+dmg+"\n" {
		t.Fatalf("assets 2.0.1: %q, %v", out, err)
	}
	entry := releaseAppcastJSON(t, "entry", "beta", "2.0.1")
	if entry["assetFileName"] != zip || entry["dmgFileName"] != dmg {
		t.Errorf("the entry names %s and %s, the release uploads %s and %s", entry["assetFileName"], entry["dmgFileName"], zip, dmg)
	}
	sha := strings.Repeat("ab", 32)
	cask, err := runReleaseAppcast(t, "", "cask", "2.0.1", sha, "sgl-project/sglang-omni")
	if err != nil {
		t.Fatal(err)
	}
	url := regexp.MustCompile(`(?m)^\s*url "([^"]+)"$`).FindStringSubmatch(cask)
	want := "https://github.com/sgl-project/sglang-omni/releases/download/voxt-v2.0.1/" + zip
	if url == nil || strings.ReplaceAll(url[1], "#{version}", "2.0.1") != want {
		t.Errorf("the cask's url is %q, want %s:\n%s", url, want, cask)
	}
	for _, line := range []string{`version "2.0.1"`, `sha256 "` + sha + `"`, "depends_on arch: :arm64", `app "Voxt.app"`} {
		if !strings.Contains(cask, "\n  "+line+"\n") {
			t.Errorf("the cask has no %s:\n%s", line, cask)
		}
	}
	// No auto_updates while brew-installed 2.0.x/2.1.0 cannot update
	// themselves: brew upgrade must still deliver the first self-updating
	// release. It returns once those are widely installed.
	if strings.Contains(cask, "auto_updates") {
		t.Errorf("the cask strands brew-installed 2.0.x/2.1.0:\n%s", cask)
	}
	if out, err := runReleaseAppcast(t, "", "cask", "2.0.1", "not-a-digest", "sgl-project/sglang-omni"); err == nil {
		t.Errorf("a cask with a bad SHA-256 printed %q", out)
	}
}

func TestReleaseAppcastNotesByLocale(t *testing.T) {
	const english = "### Added\n\n- **Remote** <models> & \"more\"\n  - Indented item  \nA paragraph.\n"
	const chinese = "\n### 中文\n\n#### 要求\n\n* **Apple 芯片**\n\n#### 变更\n\n- 远程\n"
	const build = "\n### Build\n\nSigned.\n"
	const generated = "\n## What's Changed\n\n### New Contributors\n\n- Unrelated PR\n"
	const wantEnglish = "<h3>Added</h3>\n<ul>\n<li><strong>Remote</strong> &lt;models&gt; &amp; &quot;more&quot;</li>\n<li>Indented item</li>\n</ul>\n" +
		"<p>A paragraph.</p>\n<h3>Build</h3>\n<p>Signed.</p>"
	const wantChinese = "<h3>要求</h3>\n<ul>\n<li><strong>Apple 芯片</strong></li>\n</ul>\n<h3>变更</h3>\n<ul>\n<li>远程</li>\n</ul>"
	for _, test := range []struct {
		name, notes, chineseHTML string
	}{
		{"bilingual", english + chinese + build + generated, wantChinese},
		{"English only", english + build + generated, ""},
	} {
		t.Run(test.name, func(t *testing.T) {
			out, err := runReleaseAppcast(t, test.notes, "notes-by-locale")
			if err != nil {
				t.Fatalf("notes-by-locale: %v", err)
			}
			var localized map[string]string
			if err := json.Unmarshal([]byte(out), &localized); err != nil {
				t.Fatalf("notes-by-locale printed %q: %v", out, err)
			}
			chineseHTML, hasChinese := localized["zh-Hans"]
			if localized["en"] != wantEnglish || chineseHTML != test.chineseHTML || hasChinese != (test.chineseHTML != "") {
				t.Errorf("unexpected localized notes: %v", localized)
			}
		})
	}
}

// An appcast as the update service writes it: Voxt 2.0.2 with its hardware
// requirement, then Voxt 1.16.3 on one line, without one.
const releaseAppcastXML = `<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0" xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle">
  <channel>
    <title>Voxt</title>
    <item>
      <title>2.0.2</title>
      <sparkle:version>200000299</sparkle:version>
      <sparkle:shortVersionString>2.0.2</sparkle:shortVersionString>
      <sparkle:hardwareRequirements> arm64 </sparkle:hardwareRequirements>
      <description><![CDATA[<h3>Added</h3>]]></description>
      <enclosure url="https://voxt.actnow.dev/Voxt-2.0.2-macOS.zip" length="1" type="application/zip"/>
    </item>
    <item><title>1.16.3</title><sparkle:version>101600399</sparkle:version><enclosure url="https://voxt.actnow.dev/Voxt-1.16.3-macOS.zip"/></item>
  </channel>
</rss>
`

func TestNightlyVulnerabilityScanUsesDarwin(t *testing.T) {
	data, err := os.ReadFile("../.github/workflows/voxt-nightly.yaml")
	if err != nil {
		t.Fatal(err)
	}
	workflow := string(data)
	start := strings.Index(workflow, "  vulnerabilities:\n")
	end := strings.Index(workflow, "\n  nightly:\n")
	if start < 0 || end < 0 || end <= start {
		t.Fatal("vulnerability job not found")
	}
	job := workflow[start:end]
	if !strings.Contains(job, "runs-on: macos-26") {
		t.Errorf("vulnerability job must scan the release's Darwin arm64 packages:\n%s", job)
	}
}

func TestNightlyArtifactLookupUsesPackageResult(t *testing.T) {
	data, err := os.ReadFile("docs/release.md")
	if err != nil {
		t.Fatal(err)
	}
	doc := string(data)
	for _, want := range []string{
		"--branch main",
		"voxt-nightly.yaml --branch main --event schedule --status completed",
		"--event workflow_dispatch",
		`select(.name == "nightly / Voxt release package" and .conclusion == "success")`,
	} {
		if !strings.Contains(doc, want) {
			t.Errorf("release instructions do not contain %q", want)
		}
	}
	if strings.Contains(doc, "--workflow voxt-nightly.yaml --status success") {
		t.Error("release instructions still require the vulnerability job to succeed")
	}
}

func TestReleaseAppcastHardwareRequirementKept(t *testing.T) {
	out, err := runReleaseAppcast(t, releaseAppcastXML, "items")
	if want := "200000299 arm64\n101600399 none\n"; err != nil || out != want {
		t.Errorf("items: %v\n got %q\nwant %q", err, out, want)
	}
	// voxt-appcast.yaml makes the entry the channel's latest only when the
	// appcasts' item has its hardware requirement, or when an ok reply kept it
	// and the appcast already renders it for another item. An API reply alone
	// is no proof that Voxt 1.x's XML carries it.
	const voxt1Reply = `{"ok":true,"data":{"channel":"beta","version":"2.0.2","packageURL":"https://voxt.actnow.dev/p","zipURL":"https://voxt.actnow.dev/z","dmgURL":"https://voxt.actnow.dev/d","sha256":"ab","size":1,"updated":true}}`
	withoutElement := strings.ReplaceAll(releaseAppcastXML, "<sparkle:hardwareRequirements> arm64 </sparkle:hardwareRequirements>", "")
	for _, c := range []struct {
		name, build, reply, appcast string
		kept                        bool
	}{
		{"only in the reply", "200000299", strings.Replace(voxt1Reply, `"updated":true`, `"updated":true,"hardwareRequirements": "arm64"`, 1), withoutElement, false},
		{"in the appcast", "200000299", voxt1Reply, releaseAppcastXML, true},
		{"not in either", "200000299", voxt1Reply, withoutElement, false},
		{"an echoed request, not ok", "200000299", `{"ok":false,"error":"unknown field","received":{"hardwareRequirements":"arm64"}}`, withoutElement, false},
		{"an echoed request, ok", "200000299", `{"ok":true,"data":{"version":"2.0.2"},"received":{"hardwareRequirements":"arm64"}}`, withoutElement, false},
		{"not ok", "200000299", `{"ok":false,"data":{"hardwareRequirements":"arm64"}}`, withoutElement, false},
		{"not JSON", "200000299", `"hardwareRequirements":"arm64"`, withoutElement, false},
		{"another build's", "200000399", voxt1Reply, releaseAppcastXML, false},
		{"a new build, kept in the reply and rendered for another item", "200000399", strings.Replace(voxt1Reply, `"updated":true`, `"updated":true,"hardwareRequirements": "arm64"`, 1), releaseAppcastXML, true},
		{"another requirement", "200000299", `{"ok":true,"data":{"hardwareRequirements":"x86_64"}}`, withoutElement, false},
	} {
		_, err := runReleaseAppcast(t, c.appcast, "hardware-kept", c.build, c.reply)
		var exit *exec.ExitError
		switch {
		case c.kept && err != nil:
			t.Errorf("%s: hardware-kept refused: %v", c.name, err)
		case !c.kept && err == nil:
			t.Errorf("%s: hardware-kept passed: Voxt 1.x on an Intel Mac would get a Voxt it cannot open", c.name)
		case !c.kept && errors.As(err, &exit) && !strings.Contains(string(exit.Stderr), "does not output hardwareRequirements yet"):
			t.Errorf("%s: hardware-kept says %q, not why it stops", c.name, exit.Stderr)
		}
	}
}

// A beta's version exists only in CI, and MyGo signs its update only with
// notes under that exact heading (third_party/mygo/cmd/mygo/updates.go,
// releaseNotes): beta-changelog gives it Unreleased's notes, else a line.
func TestReleaseAppcastBetaChangelog(t *testing.T) {
	const version = "2.1.2-beta.20261118.74"
	const released = "## [2.1.1] - 2026-11-17\n\n- Old.\n"
	section := func(changelog string) string {
		_, rest, ok := strings.Cut(changelog, "\n## ["+version+"]\n")
		if !ok {
			t.Fatalf("no %s heading in:\n%s", version, changelog)
		}
		notes, _, _ := strings.Cut(rest, "\n## ")
		return strings.TrimSpace(notes)
	}
	for _, test := range []struct{ name, changelog, want string }{
		{"no Unreleased", "# Changelog\n\nIntro.\n\n" + released, "A beta of Voxt built from main at abc123."},
		{"Unreleased", "# Changelog\n\n## [Unreleased]\n\n- New \\d.\n\n" + released, "- New \\d."},
		{"no section yet", "# Changelog\n", "A beta of Voxt built from main at abc123."},
	} {
		t.Run(test.name, func(t *testing.T) {
			out, err := runReleaseAppcast(t, test.changelog, "beta-changelog", version, "abc123")
			if err != nil {
				t.Fatalf("beta-changelog: %v", err)
			}
			if got := section(out); got != test.want {
				t.Errorf("the beta's notes are %q, want %q", got, test.want)
			}
			if !strings.HasPrefix(out, "# Changelog\n") || strings.Contains(test.changelog, released) != strings.Contains(out, released) {
				t.Errorf("the other sections changed:\n%s", out)
			}
		})
	}
	if out, err := runReleaseAppcast(t, "", "beta-changelog", "2.1.2", "abc123"); err == nil {
		t.Errorf("a release version got a beta section:\n%s", out)
	}
}
