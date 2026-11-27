package main

// scripts/homebrew-cask.sh writes the Homebrew cask of a stable release
// (Casks/voxt.rb of the zhaochenyang20/voxt tap), and
// the tap's Bump Voxt workflow (release/homebrew-tap) runs it for each stable release.
// The cask must offer the release's own disk image, follow only stable
// voxt-v tags and leave updates to the app. Runs on Linux: the script is bash.

import (
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

const caskSHA = "df760280097fbe1b4a61d4366f49cff296d91089fa572230de9806a153aa4ec6"

func runHomebrewCask(t *testing.T, version, sha256File, digest string) (string, error) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "Voxt.dmg.sha256")
	if err := os.WriteFile(path, []byte(sha256File), 0o644); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command("bash", "scripts/homebrew-cask.sh", version, path, digest).Output()
	return string(out), err
}

func TestHomebrewCaskOffersTheReleaseDiskImage(t *testing.T) {
	cask, err := runHomebrewCask(t, "2.1.4", caskSHA+"  Voxt-2.1.4-arm64.dmg\n", "sha256:"+caskSHA)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		`cask "voxt" do`,
		`version "2.1.4"`,
		`sha256 "` + caskSHA + `"`,
		`url "https://github.com/sgl-project/sglang-omni/releases/download/voxt-v#{version}/Voxt-#{version}-arm64.dmg"`,
		"auto_updates true",
		"depends_on arch: :arm64",
		"depends_on macos: :sequoia",
		`app "Voxt.app"`,
		"strategy :github_releases",
		`"~/Library/Application Support/VoxtNew",`,
	} {
		if !strings.Contains(cask, want) {
			t.Errorf("the cask has no %q:\n%s", want, cask)
		}
	}
	if strings.Contains(cask, "SGLang Omni") {
		t.Error("zap removes the speech models other apps share")
	}

	m := regexp.MustCompile(`regex\(/(.+)/i\)`).FindStringSubmatch(cask)
	if m == nil {
		t.Fatalf("the cask has no livecheck regex:\n%s", cask)
	}
	tags := regexp.MustCompile("(?i)" + m[1])
	for tag, stable := range map[string]bool{
		"voxt-v2.1.4":                   true,
		"voxt-v10.0.0":                  true,
		"voxt-v2.1.5-beta.20261126.144": false,
		"voxt-v2.1.5-nightly.20261127":  false,
		"v0.2.0":                        false,
		"omni-v2.2.0":                   false,
	} {
		if tags.MatchString(tag) != stable {
			t.Errorf("livecheck takes %s: %v, want %v", tag, !stable, stable)
		}
	}
}

func TestHomebrewCaskRefusesWhatIsNotTheRelease(t *testing.T) {
	other := strings.Repeat("0", 64)
	for name, c := range map[string]struct{ version, file, digest string }{
		"a beta":                     {"2.1.5-beta.20261126.144", caskSHA + "  Voxt-2.1.5-beta.20261126.144-arm64.dmg\n", "sha256:" + caskSHA},
		"another digest on GitHub":   {"2.1.4", caskSHA + "  Voxt-2.1.4-arm64.dmg\n", "sha256:" + other},
		"no digest on GitHub":        {"2.1.4", caskSHA + "  Voxt-2.1.4-arm64.dmg\n", ""},
		"the sum of another file":    {"2.1.4", caskSHA + "  Voxt-2.1.4-macOS.zip\n", "sha256:" + caskSHA},
		"the sum of another version": {"2.1.4", caskSHA + "  Voxt-2.1.3-arm64.dmg\n", "sha256:" + caskSHA},
		"an empty sum file":          {"2.1.4", "", "sha256:" + caskSHA},
	} {
		if out, err := runHomebrewCask(t, c.version, c.file, c.digest); err == nil {
			t.Errorf("%s: wrote a cask:\n%s", name, out)
		}
	}
}

func TestHomebrewTapBumpsToStableReleases(t *testing.T) {
	data, err := os.ReadFile("release/homebrew-tap/.github/workflows/bump.yml")
	if err != nil {
		t.Fatal(err)
	}
	workflow := string(data)
	for _, want := range []string{
		"schedule:",
		`ln -s "$GITHUB_WORKSPACE" "$tap"`,
		"select(.draft == false and .prerelease == false",
		`test("^voxt-v[0-9]+\\.[0-9]+\\.[0-9]+$")`,
		"contents/Voxt/scripts/homebrew-cask.sh?ref=$TAG",
		`bash "$script" "$VERSION" "$RUNNER_TEMP/$dmg.sha256" "$digest" > Casks/voxt.rb`,
		"brew audit --cask --strict --online zhaochenyang20/voxt/voxt",
		`gh pr create --head "voxt-$VERSION"`,
	} {
		if !strings.Contains(workflow, want) {
			t.Errorf("the tap's bump.yml has no %q", want)
		}
	}
	if strings.Index(workflow, "brew audit") > strings.Index(workflow, "gh pr create") {
		t.Error("the tap's bump.yml opens the pull request before brew audit")
	}
	for _, banned := range []string{"secrets.", "git push --force", "git push -f", "git push origin main"} {
		if strings.Contains(workflow, banned) {
			t.Errorf("the tap's bump.yml has %q", banned)
		}
	}
}
