// Sets or checks the version MyGo is released with, which the Go module
// (mygo.Version), the CLI and the npm packages share:
//
//   bun scripts/version.ts 0.2.0            # set it everywhere, bun.lock too
//   bun scripts/version.ts --check v0.2.0   # check that everything has it
import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { platformDir, platformName, writeManifests } from "../packages/cli/build.ts";
import { platforms } from "../packages/cli/index.js";

const root = join(import.meta.dir, "..");

/** Go files declaring the version. */
const goFiles = [
  { path: "mygo.go", pattern: /^const Version = "([^"]*)"$/m, decl: "const Version" },
  { path: "cmd/mygo/main.go", pattern: /^const version = "([^"]*)"$/m, decl: "const version" },
];

/** The official plugins' JavaScript packages, whose peer mygo-runtime is. */
export const pluginPackages = ["plugins/fetch", "plugins/websocket", "plugins/sqlite", "plugins/watch"];

/** package.json files of the npm packages. */
const packages = [
  "packages/runtime",
  ...pluginPackages,
  "packages/cli",
  ...platforms.map((p) => platformDir(p).slice(root.length + 1)),
];

/**
 * Matches the version that mygo-cli's optionalDependencies give a platform
 * package in bun.lock, the only entry of its name with a string value.
 */
function lockedPlatform(platform: string): RegExp {
  return new RegExp(`("${platformName(platform)}": )"([^"]*)"`);
}

/**
 * Updates bun.lock to the versions of the packages. bun install keeps the
 * versions of mygo-cli's optionalDependencies, which are workspaces, so
 * they are set here.
 */
async function updateLockfile(v: string): Promise<void> {
  const install = Bun.spawnSync([process.execPath, "install", "--lockfile-only"], {
    cwd: root,
    stdout: "inherit",
    stderr: "inherit",
  });
  if (!install.success) throw new Error("bun install --lockfile-only failed");
  const file = join(root, "bun.lock");
  let lock = await readFile(file, "utf8");
  for (const p of platforms) {
    if (!lockedPlatform(p).test(lock)) throw new Error(`bun.lock has no version of ${platformName(p)}`);
    lock = lock.replace(lockedPlatform(p), `$1"${v}"`);
  }
  await writeFile(file, lock);
}

async function versions(): Promise<Map<string, string>> {
  const found = new Map<string, string>();
  for (const { path, pattern } of goFiles) {
    found.set(path, (await readFile(join(root, path), "utf8")).match(pattern)?.[1] ?? "(none)");
  }
  for (const dir of packages) {
    const pkg = JSON.parse(await readFile(join(root, dir, "package.json"), "utf8"));
    found.set(`${dir}/package.json`, pkg.version);
    for (const [name, v] of Object.entries<string>(pkg.optionalDependencies ?? {})) {
      found.set(`${dir}/package.json: ${name}`, v);
    }
    for (const deps of ["peerDependencies", "devDependencies"]) {
      const runtime = pkg[deps]?.["mygo-runtime"];
      if (runtime) found.set(`${dir}/package.json: ${deps} mygo-runtime`, runtime.replace(/^\^/, ""));
    }
  }
  const lock = await readFile(join(root, "bun.lock"), "utf8");
  for (const p of platforms) {
    found.set(`bun.lock: ${platformName(p)}`, lock.match(lockedPlatform(p))?.[2] ?? "(none)");
  }
  return found;
}

if (import.meta.main) {
  const args = process.argv.slice(2);
  if (args[0] === "--check" && args[1]) {
    const want = args[1].replace(/^v/, "");
    const wrong = [...(await versions())].filter(([, v]) => v !== want);
    for (const [where, v] of wrong) console.error(`${where} has ${v}, not ${want}`);
    if (wrong.length > 0) process.exit(1);
    console.log(`everything is at ${want}`);
  } else if (args.length === 1 && /^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$/.test(args[0]!)) {
    const v = args[0]!;
    for (const { path, pattern, decl } of goFiles) {
      const file = join(root, path);
      await writeFile(file, (await readFile(file, "utf8")).replace(pattern, `${decl} = "${v}"`));
    }
    for (const dir of ["packages/runtime", ...pluginPackages]) {
      const file = join(root, dir, "package.json");
      const pkg = JSON.parse(await readFile(file, "utf8"));
      pkg.version = v;
      for (const deps of [pkg.peerDependencies, pkg.devDependencies]) {
        if (deps?.["mygo-runtime"]) deps["mygo-runtime"] = `^${v}`;
      }
      await writeFile(file, JSON.stringify(pkg, null, 2) + "\n");
    }
    await writeManifests(v);
    await updateLockfile(v);
    console.log(`set the version to ${v}: commit, then push the tag v${v} to release`);
  } else {
    console.error("usage: bun scripts/version.ts <version> | --check <tag>");
    process.exit(2);
  }
}
