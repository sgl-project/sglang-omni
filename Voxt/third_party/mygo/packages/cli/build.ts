// Builds the mygo binaries of the platform packages in npm/, for every
// platform or for those given as arguments (e.g. `bun run build.ts
// darwin-arm64`), and writes the manifests of mygo-cli and its platform
// packages with the version of the Go module (mygo.Version), and their
// license.
import { copyFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { platforms } from "./index.js";

export const dir = import.meta.dir;
const root = join(dir, "..", "..");

/** GOOS and GOARCH of each platform. */
export const goTargets: Record<string, [goos: string, goarch: string]> = {
  "darwin-arm64": ["darwin", "arm64"],
  "darwin-x64": ["darwin", "amd64"],
  "linux-arm64": ["linux", "arm64"],
  "linux-x64": ["linux", "amd64"],
  "win32-arm64": ["windows", "arm64"],
  "win32-x64": ["windows", "amd64"],
};

const osNames: Record<string, string> = { darwin: "macOS", linux: "Linux", win32: "Windows" };

const repository = { type: "git", url: "git+https://github.com/egoist/mygo.git" };

/** Returns mygo.Version, which every package is released with. */
export async function version(): Promise<string> {
  const source = await readFile(join(root, "mygo.go"), "utf8");
  const match = source.match(/^const Version = "([^"]+)"$/m);
  if (!match?.[1]) throw new Error("mygo.go declares no Version");
  return match[1];
}

/**
 * Returns the name of a platform package. Its scope keeps npm from taking
 * names such as mygo-cli-win32-x64 for spam.
 */
export function platformName(platform: string): string {
  return `@egoist/mygo-cli-${platform}`;
}

/** Returns the directory of a platform package. */
export function platformDir(platform: string): string {
  return join(dir, "npm", platform);
}

/** Writes the manifests of mygo-cli and its platform packages. */
export async function writeManifests(v: string): Promise<void> {
  const main = JSON.parse(await readFile(join(dir, "package.json"), "utf8"));
  main.version = v;
  main.optionalDependencies = Object.fromEntries(platforms.map((p) => [platformName(p), v]));
  await writeJSON(join(dir, "package.json"), main);
  for (const platform of platforms) {
    const [os, cpu] = platform.split("-") as [string, string];
    const name = platformName(platform);
    const description = `The ${osNames[os]} ${cpu} binary of mygo-cli`;
    await mkdir(platformDir(platform), { recursive: true });
    await writeJSON(join(platformDir(platform), "package.json"), {
      name,
      version: v,
      description,
      repository: { ...repository, directory: `packages/cli/npm/${platform}` },
      license: "MIT",
      os: [os],
      cpu: [cpu],
      files: ["bin"],
      preferUnplugged: true,
    });
    await copyFile(join(root, "LICENSE"), join(platformDir(platform), "LICENSE"));
    await writeFile(
      join(platformDir(platform), "README.md"),
      `# ${name}\n\n${description}, the command line tool of MyGo. Install\n` +
        `[mygo-cli](https://www.npmjs.com/package/mygo-cli) instead: package managers\n` +
        `install the package of the platform they run on.\n`,
    );
  }
}

/** Builds the binaries of platforms into their packages. */
export async function build(targets: readonly string[] = platforms): Promise<void> {
  await writeManifests(await version());
  for (const platform of targets) await buildBinary(platform);
}

/**
 * Builds the binary of a platform into its package, whose manifest
 * writeManifests wrote. It doesn't block the event loop, so that
 * scripts/publish.ts publishes other packages meanwhile.
 */
export async function buildBinary(platform: string): Promise<void> {
  const target = goTargets[platform];
  if (!target) throw new Error(`unknown platform ${platform}, want one of ${platforms.join(", ")}`);
  const [goos, goarch] = target;
  const bin = join(platformDir(platform), "bin", goos === "windows" ? "mygo.exe" : "mygo");
  console.log(`building ${platform}`);
  const go = Bun.spawn(["go", "build", "-trimpath", "-ldflags=-s -w", "-o", bin, "./cmd/mygo"], {
    cwd: root,
    env: { ...process.env, GOOS: goos, GOARCH: goarch, CGO_ENABLED: "0" },
    stdout: "inherit",
    stderr: "inherit",
  });
  if ((await go.exited) !== 0) throw new Error(`go build failed for ${platform}`);
}

async function writeJSON(path: string, value: unknown): Promise<void> {
  await writeFile(path, JSON.stringify(value, null, 2) + "\n");
}

if (import.meta.main) {
  const args = process.argv.slice(2);
  await build(args.length > 0 ? args : platforms);
}
