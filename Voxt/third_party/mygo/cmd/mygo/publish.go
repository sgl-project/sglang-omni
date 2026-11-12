package main

import (
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
)

// checkUpload checks, before a build, that -upload has somewhere to upload
// to.
func checkUpload(c *Config) error {
	switch u := c.Updates; {
	case u != nil && u.S3 != nil:
		_, err := newS3Client(u.S3)
		return err
	case u == nil || u.GitHub == "":
		return fmt.Errorf("-upload needs updates.github or updates.s3 in %s", c.configName())
	}
	return nil
}

// publish uploads the artifacts of a build where updates point to: the
// bucket of updates.s3, else the GitHub release of this version.
func publish(c *Config, artifacts []string) error {
	if c.Updates != nil && c.Updates.S3 != nil {
		return publishS3(c, artifacts)
	}
	return publishGitHub(c, artifacts)
}

// uploads returns the artifacts of a build to publish: the installers and
// update files, and apart the update manifests, which go last so that they
// never point at a file that is not there.
func uploads(artifacts []string) (files, manifests []string, err error) {
	for _, a := range artifacts {
		info, err := os.Stat(a)
		if err != nil || info.IsDir() {
			continue // the macOS bundle ships in the disk image and the update
		}
		name := filepath.Base(a)
		switch {
		case strings.HasPrefix(name, "update-") && strings.HasSuffix(name, ".json"):
			manifests = append(manifests, a)
		case name == installScriptName:
			// The same script for every Linux target.
			if !slices.ContainsFunc(files, func(f string) bool { return filepath.Base(f) == name }) {
				files = append(files, a)
			}
		case strings.HasSuffix(name, ".dmg"), strings.HasSuffix(name, ".exe") && strings.Contains(name, " Setup "),
			strings.HasSuffix(name, ".deb"), strings.HasSuffix(name, ".tar.gz"), strings.HasSuffix(name, ".delta"):
			files = append(files, a)
		}
	}
	if len(files)+len(manifests) == 0 {
		return nil, nil, errors.New("nothing to upload")
	}
	return files, manifests, nil
}

// publishGitHub uploads the artifacts of a build to the GitHub release of
// this version, a draft created if needed, with the gh CLI. Installed apps
// see the update once the release is published.
func publishGitHub(c *Config, artifacts []string) error {
	if c.Updates == nil || c.Updates.GitHub == "" {
		return fmt.Errorf("-upload needs updates.github in %s", c.configName())
	}
	gh, err := exec.LookPath("gh")
	if err != nil {
		return errors.New("-upload needs the GitHub CLI (https://cli.github.com), signed in with gh auth login or GH_TOKEN")
	}
	repo, tag := c.Updates.GitHub, c.Updates.TagPrefix+c.Version
	files, manifests, err := uploads(artifacts)
	if err != nil {
		return err
	}
	// With a tag prefix, the repository's latest release is another's, which
	// the app's releases leave alone.
	latest := ""
	if c.Updates.tagged() {
		latest = " --latest=false"
	}
	if exec.Command(gh, "release", "view", tag, "--repo", repo).Run() != nil {
		notes, _ := c.releaseNotes()
		t := con.start("Creating the draft release " + tag + " of " + repo)
		args := []string{"release", "create", tag, "--repo", repo, "--draft", "--title", tag, "--notes", notes}
		if latest != "" {
			args = append(args, "--latest=false")
		}
		if out, err := exec.Command(gh, args...).CombinedOutput(); err != nil {
			t.fail()
			return fmt.Errorf("gh release create: %v\n%s", err, out)
		}
		t.done("Created the draft release " + tag + " of " + repo)
	}
	for _, batch := range [][]string{files, manifests} {
		if len(batch) == 0 {
			continue
		}
		t := con.start("Uploading " + uploadNames(batch))
		args := append([]string{"release", "upload", tag, "--repo", repo, "--clobber"}, batch...)
		if out, err := exec.Command(gh, args...).CombinedOutput(); err != nil {
			t.fail()
			return fmt.Errorf("gh release upload: %v\n%s", err, out)
		}
		t.done("Uploaded " + uploadNames(batch))
	}
	logf("Uploaded to the release %s of %s; publish it when every platform is there:\n  gh release edit %s --repo %s --draft=false%s", tag, repo, tag, repo, latest)
	return nil
}

// publishS3 uploads the artifacts of a build to the bucket of updates.s3,
// which updates.url serves. Installed apps see the update of a platform
// once its manifest is there.
func publishS3(c *Config, artifacts []string) error {
	bucket, err := newS3Client(c.Updates.S3)
	if err != nil {
		return err
	}
	files, manifests, err := uploads(artifacts)
	if err != nil {
		return err
	}
	for _, f := range slices.Concat(files, manifests) {
		name := filepath.Base(f)
		t := con.start("Uploading " + name)
		t.set(sizeOf(f))
		if err := t.end(bucket.put(bucket.key(name), f, uploadHeader(name)), "Uploaded "+name); err != nil {
			return err
		}
	}
	logf("Uploaded to s3://%s/%s, which %s serves", bucket.bucket, bucket.prefix, c.Updates.URL)
	return nil
}

// uploadHeader returns the headers of an uploaded file: its type and, for
// the files whose names stay from version to version, that caches must
// check them again.
func uploadHeader(name string) http.Header {
	typ, ok := contentTypes[filepath.Ext(name)]
	if !ok {
		typ = "application/octet-stream"
	}
	h := http.Header{"Content-Type": {typ}}
	if name == installScriptName || strings.HasPrefix(name, "update-") && strings.HasSuffix(name, ".json") {
		h.Set("Cache-Control", "no-cache")
	}
	return h
}

var contentTypes = map[string]string{
	".dmg":  "application/x-apple-diskimage",
	".exe":  "application/vnd.microsoft.portable-executable",
	".deb":  "application/vnd.debian.binary-package",
	".gz":   "application/gzip",
	".json": "application/json",
	".sh":   "text/plain; charset=utf-8", // readable in a browser before piping it to sh
}

// uploadNames names the files of an upload.
func uploadNames(files []string) string {
	if len(files) == 1 {
		return filepath.Base(files[0])
	}
	return fmt.Sprintf("%d files", len(files))
}
