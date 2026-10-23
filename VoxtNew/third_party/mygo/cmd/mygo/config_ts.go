package main

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
)

// mygo.config.ts is evaluated by a JavaScript runtime that runs TypeScript:
// Bun, or Node.js 22.6 and later, which strip the types. A loader imports
// the file, awaits its default export, calls it with the command when it is
// a function, and writes the configuration as JSON, which then goes through
// the same checks as mygo.json. The runtime reads the loader from its
// standard input, not from a command line argument: on Windows, npm
// installs commands as batch files (bun.cmd), and cmd.exe ends their
// command line at the loader's first line break.

// running is the mygo command running, which a configuration exporting a
// function gets.
var running string

// configRuntimes are the runtimes that may evaluate mygo.config.ts, in order
// of preference.
var configRuntimes = []string{"bun", "node"}

const configLoader = `
import { writeFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const mod = await import(pathToFileURL(process.env.MYGO_CONFIG_FILE).href);
let config = await mod.default;
if (typeof config === "function") config = await config({ command: process.env.MYGO_CONFIG_COMMAND });
if (typeof config !== "object" || config === null || Array.isArray(config)) {
  throw new Error("the default export is not a configuration object, or a function returning one");
}
writeFileSync(process.env.MYGO_CONFIG_OUT, JSON.stringify(config));
`

// evalTSConfig evaluates the configuration in file and returns it as JSON.
func evalTSConfig(root, file string) ([]byte, error) {
	cmd, err := configRuntime()
	if err != nil {
		return nil, err
	}
	out, err := os.CreateTemp("", "mygo-config-*.json")
	if err != nil {
		return nil, err
	}
	out.Close()
	defer os.Remove(out.Name())
	var stderr bytes.Buffer
	cmd.Dir = root
	cmd.Env = append(os.Environ(), "MYGO_CONFIG_FILE="+file, "MYGO_CONFIG_OUT="+out.Name(), "MYGO_CONFIG_COMMAND="+running)
	// What the configuration prints must not mix with mygo's own output.
	printed := con.output("config", os.Stderr)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = strings.NewReader(configLoader), printed, &stderr
	if err := cmd.Run(); err != nil {
		if msg := strings.TrimSpace(stderr.String()); msg != "" {
			return nil, errors.New(msg)
		}
		return nil, fmt.Errorf("%s: %w", cmd.Args[0], err)
	}
	_, _ = printed.Write(stderr.Bytes())
	printed.flush()
	data, err := os.ReadFile(out.Name())
	if err == nil && len(data) == 0 {
		err = fmt.Errorf("%s exited without writing the configuration", cmd.Args[0])
	}
	return data, err
}

// configRuntime returns the command that runs configLoader, given on its
// standard input.
func configRuntime() (*exec.Cmd, error) {
	for _, name := range configRuntimes {
		path, err := exec.LookPath(name)
		if err != nil {
			continue
		}
		if name == "bun" {
			return exec.Command(path, "run", "-"), nil
		}
		if flags, ok := nodeTypeScriptFlags(path); ok {
			return exec.Command(path, append(flags, "--input-type=module", "-")...), nil
		}
	}
	return nil, errors.New("evaluating " + tsConfig + " needs Bun (https://bun.sh) or Node.js 22.6 or later")
}

var nodeVersions sync.Map // path → the output of node --version

// nodeTypeScriptFlags returns the flags that make the Node.js at path run
// TypeScript, and false when it is too old to.
func nodeTypeScriptFlags(path string) ([]string, bool) {
	v, ok := nodeVersions.Load(path)
	if !ok {
		out, err := exec.Command(path, "--version").Output()
		if err != nil {
			return nil, false
		}
		v, _ = nodeVersions.LoadOrStore(path, strings.TrimSpace(string(out)))
	}
	parts := strings.SplitN(strings.TrimPrefix(v.(string), "v"), ".", 3)
	if len(parts) < 2 {
		return nil, false
	}
	major, _ := strconv.Atoi(parts[0])
	minor, _ := strconv.Atoi(parts[1])
	flags := []string{"--no-warnings"}
	switch {
	case major >= 24, major == 23 && minor >= 6, major == 22 && minor >= 18:
		// Types are stripped by default.
		return flags, true
	case major == 23, major == 22 && minor >= 6:
		return append(flags, "--experimental-strip-types"), true
	}
	return nil, false
}
