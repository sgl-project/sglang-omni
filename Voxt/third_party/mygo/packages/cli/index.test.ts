import { expect, test } from "bun:test";
import { chmodSync, cpSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, sep } from "node:path";
import { build, dir, goTargets, platformDir, platformName, version } from "./build.ts";
import { defineConfig, platformPackage, platforms } from "./index.js";

const host = `${process.platform}-${process.arch}`;
const exe = process.platform === "win32" ? "mygo.exe" : "mygo";
const readJSON = (path: string) => JSON.parse(readFileSync(path, "utf8"));

test("platformPackage names the package of each platform", () => {
  expect(platformPackage("darwin", "arm64")).toBe("@egoist/mygo-cli-darwin-arm64");
  expect(platformPackage("win32", "x64")).toBe("@egoist/mygo-cli-win32-x64");
  expect(platformName("linux-x64")).toBe(platformPackage("linux", "x64"));
  expect(() => platformPackage("freebsd", "x64")).toThrow(/go install github.com\/egoist\/mygo\/cmd\/mygo@v/);
  expect(Object.keys(goTargets).sort()).toEqual([...platforms].sort());
});

test("the packages have the version of the Go module", async () => {
  const v = await version();
  const main = readJSON(join(dir, "package.json"));
  expect(main.version).toBe(v);
  expect(main.optionalDependencies).toEqual(Object.fromEntries(platforms.map((p) => [platformName(p), v])));
  for (const platform of platforms) {
    const pkg = readJSON(join(platformDir(platform), "package.json"));
    const [os, cpu] = platform.split("-");
    expect(pkg).toMatchObject({ name: platformName(platform), version: v, os: [os], cpu: [cpu], license: "MIT" });
  }
});

test("the published packages carry the license", () => {
  const license = readFileSync(join(dir, "..", "..", "LICENSE"), "utf8");
  for (const pkg of [join(dir, "..", "runtime"), dir, ...platforms.map(platformDir)]) {
    expect(readJSON(join(pkg, "package.json")).license).toBe("MIT");
    expect(readFileSync(join(pkg, "LICENSE"), "utf8")).toBe(license);
  }
});

// install lays out mygo-cli in node_modules as a package manager would, with
// or without the package of this platform, and returns the binary's path.
function install(root: string, withPlatform: boolean) {
  const cli = join(root, "node_modules", "mygo-cli");
  mkdirSync(join(cli, "bin"), { recursive: true });
  for (const file of ["package.json", "index.js", "bin/mygo.js"]) cpSync(join(dir, file), join(cli, file));
  if (withPlatform) {
    const pkg = join(root, "node_modules", platformName(host));
    cpSync(platformDir(host), pkg, { recursive: true });
    return join(pkg, "bin", exe);
  }
}

function run(runtime: string, root: string, args: string[], env: Record<string, string> = {}) {
  const r = Bun.spawnSync([runtime, join(root, "node_modules", "mygo-cli", "bin", "mygo.js"), ...args], {
    env: { ...process.env, MYGO_CLI_BINARY: "", ...env },
  });
  return { code: r.exitCode, out: r.stdout.toString(), err: r.stderr.toString() };
}

const runtimes = ["node", "bun"].filter((r) => Bun.which(r));

test.skipIf(!platforms.includes(host))(
  "mygo runs the binary of the platform's package",
  async () => {
    await build([host]);
    const v = await version();
    const root = mkdtempSync(join(tmpdir(), "mygo-cli-"));
    try {
      const bin = install(root, true)!;
      if (process.platform !== "win32") chmodSync(bin, 0o644); // a package manager dropped the executable bit
      for (const runtime of runtimes) {
        expect(run(runtime, root, ["version"])).toMatchObject({ code: 0, out: `mygo ${v}\n` });
        expect(run(runtime, root, ["no-such-command"]).code).toBe(2);
      }
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  },
  120_000,
);

// bunLock returns a bun.lock of a project depending on mygo-cli, which
// resolved the package of this platform or, when bun could not fetch it,
// did not.
function bunLock(withPlatform: boolean) {
  const pkg = platformName(host);
  const resolved = withPlatform ? `\n    "${pkg}": ["${pkg}@0.1.1", "", {}, "sha512-"],` : "";
  return `{
  "lockfileVersion": 2,
  "workspaces": { "": { "dependencies": { "mygo-cli": "^0.1.1" } } },
  "packages": {
    "mygo-cli": ["mygo-cli@0.1.1", "", { "optionalDependencies": { "${pkg}": "0.1.1" } }, "sha512-"],${resolved}
  }
}
`;
}

test("mygo explains a missing platform package", () => {
  const base = mkdtempSync(join(tmpdir(), "mygo-cli-"));
  const missing = `${platformName(host)}, the package with the mygo binary of this platform, is not installed`;
  const omitted = "Reinstall mygo-cli without omitting optional dependencies";
  try {
    const project = join(base, "project");
    install(project, false);
    const bunx = join(base, "bunx-501-mygo-cli@latest");
    install(bunx, false);
    const npx = join(base, "_npx", "0123456789abcdef");
    install(npx, false);
    for (const runtime of runtimes) {
      rmSync(join(project, "bun.lock"), { force: true });
      let r = run(runtime, project, ["version"]);
      expect(r.code).toBe(1);
      expect(r.err).toContain(missing);
      expect(r.err).toContain(omitted);
      // bun keeps out what its lockfile lacks. The messages name the real
      // paths, which can differ from ours, as in 8.3 names on Windows.
      writeFileSync(join(project, "bun.lock"), bunLock(false));
      r = run(runtime, project, ["version"]);
      expect(r.err).toContain(missing);
      expect(r.err).toContain(`${sep}project${sep}bun.lock lacks it, even with --force: remove that file`);
      writeFileSync(join(project, "bun.lock"), bunLock(true));
      expect(run(runtime, project, ["version"]).err).toContain(omitted);
      // bunx and npx run the install they made again.
      r = run(runtime, bunx, ["version"]);
      expect(r.err).toContain(missing);
      expect(r.err).toContain("bunx installed mygo-cli without it in ");
      expect(r.err).toContain(`${sep}bunx-501-mygo-cli@latest: remove that directory, then run bunx again`);
      r = run(runtime, npx, ["version"]);
      expect(r.err).toContain("npx installed mygo-cli without it in ");
      expect(r.err).toContain(`${sep}_npx${sep}0123456789abcdef: remove that directory, then run npx again`);
      // MYGO_CLI_BINARY points at a binary of one's own instead.
      const own = run(runtime, project, ["--version"], { MYGO_CLI_BINARY: process.execPath });
      expect(own.code).toBe(0);
    }
  } finally {
    rmSync(base, { recursive: true, force: true });
  }
});

test("defineConfig returns the configuration", async () => {
  const config = defineConfig({
    name: "My App",
    fileAssociations: [{ ext: ["md"], role: "Viewer" }],
    macos: { infoPlist: { NSCameraUsageDescription: "Scan documents." } },
  });
  expect(config.name).toBe("My App");
  const fn = defineConfig(({ command }) => ({ devUrl: command === "dev" ? "http://localhost:5173" : undefined }));
  expect(await fn({ command: "dev" })).toEqual({ devUrl: "http://localhost:5173" });
});
