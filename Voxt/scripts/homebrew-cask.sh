#!/usr/bin/env bash
# The Homebrew cask of a stable Voxt release, Casks/voxt.rb of the
# sgl-project/homebrew-voxt tap (brew install --cask sgl-project/voxt/voxt).
# .github/workflows/voxt-homebrew.yaml runs it when a release is published
# and opens a pull request on the tap with the result (docs/release.md).
#
#   scripts/homebrew-cask.sh VERSION SHA256_FILE DIGEST
#
# VERSION is a stable X.Y.Z (a beta gets no cask), SHA256_FILE the release's
# Voxt-X.Y.Z-arm64.dmg.sha256 and DIGEST the digest GitHub gives for the disk
# image asset (sha256:…). It fails unless the file names that disk image and
# both give the same SHA-256, so the cask never offers a file that is not the
# one on the release.
set -euo pipefail

fail() {
  echo "homebrew-cask.sh: $*" >&2
  exit 1
}

(($# == 3)) || fail "usage: scripts/homebrew-cask.sh VERSION SHA256_FILE DIGEST"
version="$1" file="$2" digest="${3#sha256:}"
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || fail "'$version' is not a stable version X.Y.Z"
dmg="Voxt-$version-arm64.dmg"
sha256="" name=""
read -r sha256 name < "$file" || true
name="${name#\*}"
[[ "$sha256" =~ ^[0-9a-f]{64}$ ]] || fail "$file has no SHA-256"
[[ "$name" == "$dmg" ]] || fail "$file is for '$name', not $dmg"
[[ "$sha256" == "$digest" ]] || fail "$file says $sha256 but GitHub has '$digest' for $dmg"

# auto_updates: Voxt installs its own updates (update-darwin-arm64.json), so
# brew upgrade leaves an installed Voxt alone unless asked with --greedy.
# The zap list is Voxt's own data; the speech models in Application
# Support/SGLang Omni are shared with other apps and stay.
cat <<CASK
cask "voxt" do
  version "$version"
  sha256 "$sha256"

  url "https://github.com/sgl-project/sglang-omni/releases/download/voxt-v#{version}/Voxt-#{version}-arm64.dmg"
  name "Voxt"
  desc "Voice input, translation and rewriting with local speech models"
  homepage "https://github.com/sgl-project/sglang-omni/tree/main/Voxt"

  livecheck do
    url :url
    regex(/^voxt-v(\d+(?:\.\d+)+)$/i)
    strategy :github_releases
  end

  auto_updates true
  depends_on arch: :arm64
  depends_on macos: :sequoia

  app "Voxt.app"

  zap trash: [
    "~/Library/Application Support/VoxtNew",
    "~/Library/Caches/VoxtNew",
    "~/Library/HTTPStorages/dev.hehehai.voxtnew",
    "~/Library/Logs/VoxtNew",
    "~/Library/Preferences/dev.hehehai.voxtnew.plist",
    "~/Library/WebKit/dev.hehehai.voxtnew",
  ]
end
CASK
