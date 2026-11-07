package main

// scripts/release-appcast.sh says what Voxt 1.x's in-app update (Sparkle
// 2.10) needs to install a Voxt 2 release: an appcast entry whose
// sparkle:version is above the running 1.x's CFBundleVersion, an app whose
// own CFBundleVersion is not below it (else Sparkle refuses a downgrade) and
// whose Info.plist has an EdDSA key, and the files the release uploads; and
// that Voxt 1.x on an Intel Mac, where Voxt 2 does not open, is not offered it.
// Runs on Linux: the script is plain bash and awk.

import (
	"encoding/json"
	"errors"
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
	// Voxt 2 does not update itself: with auto_updates, brew upgrade would skip it.
	if strings.Contains(cask, "auto_updates") {
		t.Errorf("the cask says auto_updates:\n%s", cask)
	}
	if out, err := runReleaseAppcast(t, "", "cask", "2.0.1", "not-a-digest", "sgl-project/sglang-omni"); err == nil {
		t.Errorf("a cask with a bad SHA-256 printed %q", out)
	}
}

func TestReleaseAppcastNotes(t *testing.T) {
	notes := "### Added\n\n- **Remote** <models> & \"more\"\n  - Indented item  \nA paragraph.\n\n### 中文\n\n#### 新增\n\n- 远程\n"
	want := "<h3>Added</h3>\n<ul>\n<li>**Remote** &lt;models&gt; &amp; &quot;more&quot;</li>\n<li>Indented item</li>\n</ul>\n" +
		"<p>A paragraph.</p>\n<h3>中文</h3>\n<h4>新增</h4>\n<ul>\n<li>远程</li>\n</ul>\n"
	out, err := runReleaseAppcast(t, notes, "notes")
	if err != nil || out != want {
		t.Errorf("notes: %v\n got %q\nwant %q", err, out, want)
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

func TestReleaseAppcastHardwareRequirementKept(t *testing.T) {
	out, err := runReleaseAppcast(t, releaseAppcastXML, "items")
	if want := "200000299 arm64\n101600399 none\n"; err != nil || out != want {
		t.Errorf("items: %v\n got %q\nwant %q", err, out, want)
	}
	// voxt-appcast.yaml makes the entry the channel's latest only when the
	// update service kept its hardware requirement: in its reply to the
	// entry, or in the appcasts' item of the build.
	for _, c := range []struct {
		name, build, reply, appcast string
		kept                        bool
	}{
		{"in the reply", "200000299", `{"ok":true,"data":{"entry":{"hardwareRequirements": "arm64"}}}`, "", true},
		{"in the appcast", "200000299", `{"ok":true}`, releaseAppcastXML, true},
		{"not in either", "200000299", `{"ok":true,"data":{"version":"2.0.2"}}`, strings.ReplaceAll(releaseAppcastXML, "<sparkle:hardwareRequirements> arm64 </sparkle:hardwareRequirements>", ""), false},
		{"another build's", "200000399", `{"ok":true}`, releaseAppcastXML, false},
		{"another requirement", "200000299", `{"hardwareRequirements":"x86_64"}`, "", false},
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
