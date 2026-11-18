package scripts

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestVendorMygoCLIExecutable(t *testing.T) {
	root := t.TempDir()
	write := func(name, content string) {
		t.Helper()
		if err := os.MkdirAll(filepath.Dir(name), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(name, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	script, err := os.ReadFile("vendor-mygo.sh")
	if err != nil {
		t.Fatal(err)
	}
	scriptPath := filepath.Join(root, "scripts", "vendor-mygo.sh")
	if err := os.MkdirAll(filepath.Dir(scriptPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(scriptPath, script, 0o755); err != nil {
		t.Fatal(err)
	}
	write(filepath.Join(root, "go.mod"), "module vendorcheck\n\ngo 1.23\n\nrequire (\n\tgithub.com/egoist/mygo v0.0.0\n\texample.com/other v0.0.0\n)\n")
	if err := os.MkdirAll(filepath.Join(root, "third_party", "mygo-patches"), 0o755); err != nil {
		t.Fatal(err)
	}

	moduleCache := filepath.Join(root, "modcache")
	source := filepath.Join(moduleCache, "github.com", "egoist", "mygo@v0.0.0")
	write(filepath.Join(source, "go.mod"), "module github.com/egoist/mygo\n\ngo 1.23\n")
	write(filepath.Join(source, "internal", "scene", "scene.go"), "package scene\n")
	write(filepath.Join(source, "packages", "cli", "bin", "mygo.js"), "#!/usr/bin/env node\n")

	run := func(args ...string) (string, error) {
		command := exec.Command(scriptPath, args...)
		command.Dir = root
		command.Env = append(os.Environ(), "GOMODCACHE="+moduleCache, "GOPROXY=off", "GOTOOLCHAIN=local", "GOFLAGS=")
		output, err := command.CombinedOutput()
		return string(output), err
	}
	if output, err := run(); err != nil {
		t.Fatalf("vendor: %v\n%s", err, output)
	}
	cli := filepath.Join(root, "third_party", "mygo", "packages", "cli", "bin", "mygo.js")
	info, err := os.Stat(cli)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm()&0o111 == 0 {
		t.Fatalf("generated CLI mode is %04o", info.Mode().Perm())
	}
	if output, err := run("--check"); err != nil {
		t.Fatalf("check generated tree: %v\n%s", err, output)
	}
	if err := os.Chmod(cli, 0o644); err != nil {
		t.Fatal(err)
	}
	if output, err := run("--check"); err == nil {
		t.Fatalf("check accepted a non-executable CLI:\n%s", output)
	}
}
