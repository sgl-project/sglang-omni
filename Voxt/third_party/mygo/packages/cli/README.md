# mygo-cli

The `mygo` command line tool of [MyGo](https://github.com/egoist/mygo), the
desktop app framework for Go whose windows show web pages, in the system
webview, or native UI written in Go, as prebuilt binaries for macOS, Linux
and Windows (arm64 and x64).

Create a project, which depends on mygo-cli and runs it from its scripts:

```sh
bunx mygo-cli init my-app    # or: npx mygo-cli init my-app
cd my-app
bun run dev                  # mygo dev: the app with live reload
bun run build                # mygo build: the packaged app
```

In a project that depends on mygo-cli, `bunx mygo <command>` runs its
`mygo`. Or install it globally with `npm install -g mygo-cli`. The commands
are those of the CLI installed with Go
(`go install github.com/egoist/mygo/cmd/mygo@latest`): `init`, `dev`,
`build`, `generate`, `install-skills`, `keygen` and `doctor`. Building apps
still needs [Go](https://go.dev/dl/), which compiles them.

Outside such a project, run `bunx mygo-cli` or `npx mygo-cli` rather than
`bunx mygo`: npm has an unrelated package named mygo.

## How it works

Each platform's binary is in a package of its own, such as
`@egoist/mygo-cli-darwin-arm64`, which mygo-cli lists as an optional dependency:
package managers install only the one that matches the machine. The `mygo`
command then runs that binary. Set `MYGO_CLI_BINARY` to the path of another
build of the CLI to use it instead.

## mygo.config.ts

`defineConfig` types the configuration of a project, `mygo.config.ts`:

```ts
import { defineConfig } from "mygo-cli";

export default defineConfig({
  name: "My App",
  identifier: "com.example.myapp",
  devUrl: "http://localhost:5173",
});
```

## The binary

From JavaScript, `binaryPath()` returns the path of the binary:

```ts
import { binaryPath } from "mygo-cli";

Bun.spawnSync([binaryPath(), "generate"]);
```
