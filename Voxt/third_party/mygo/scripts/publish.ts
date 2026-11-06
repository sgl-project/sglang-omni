// Publishes the npm packages of a release: mygo-runtime, the official
// plugins' packages (@mygo-plugins/*), the platform
// packages of mygo-cli with freshly built binaries, then mygo-cli, which
// depends on them, once npm serves them. Versions already on npm are
// skipped, so a release that failed halfway can run again, and prereleases
// get the next dist-tag. Arguments go to npm publish, e.g. --dry-run or
// --provenance.
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { buildBinary, dir as cliDir, platformDir, version, writeManifests } from "../packages/cli/build.ts";
import { platforms } from "../packages/cli/index.js";
import { pluginPackages } from "./version.ts";

const root = join(import.meta.dir, "..");
const flags = process.argv.slice(2);

// npm serves a package minutes after accepting it (one to five for 0.2.11),
// so each package goes out as soon as it can, alongside the others: the
// runtime and the plugins while the binaries build, and each platform
// package once its binary is built.
await writeManifests(await version());
const libraries = [join(root, "packages", "runtime"), ...pluginPackages.map((p) => join(root, p))];
const publishing = libraries.map(publish);
for (const platform of platforms) {
  await buildBinary(platform);
  publishing.push(publish(platformDir(platform)));
}
if (!(await Promise.all(publishing)).every(Boolean)) process.exit(1);
// A package manager installs mygo-cli without the platform packages it
// cannot fetch: bun then keeps them out of later installs while its lockfile
// lacks them. The projects mygo init creates depend on mygo-runtime too.
if (!flags.includes("--dry-run")) {
  for (const pkg of [...libraries, ...platforms.map(platformDir)]) await waitUntilServed(await manifest(pkg));
}
if (!(await publish(cliDir))) process.exit(1);

async function manifest(pkg: string): Promise<{ name: string; version: string }> {
  return JSON.parse(await readFile(join(pkg, "package.json"), "utf8"));
}

/**
 * Publishes a package unless npm has its version, and reports whether that
 * went well. It prints npm's output once done, so that packages published
 * together don't mix theirs.
 */
async function publish(pkg: string): Promise<boolean> {
  const { name, version } = await manifest(pkg);
  const view = Bun.spawn(["npm", "view", `${name}@${version}`, "version"], { stderr: "ignore" });
  if ((await new Response(view.stdout).text()).trim() === version) {
    console.log(`${name}@${version} is already published`);
    return true;
  }
  const tag = version.includes("-") ? ["--tag", "next"] : [];
  const npm = Bun.spawn(["npm", "publish", "--access", "public", ...tag, ...flags], {
    cwd: pkg,
    stdout: "pipe",
    stderr: "pipe",
  });
  const [stdout, stderr, code] = await Promise.all([
    new Response(npm.stdout).text(),
    new Response(npm.stderr).text(),
    npm.exited,
  ]);
  process.stdout.write(stderr + stdout);
  if (code !== 0) console.error(`publishing ${name}@${version} failed`);
  return code === 0;
}

/** Waits until npm serves name@version to package managers. */
async function waitUntilServed({ name, version }: { name: string; version: string }): Promise<void> {
  const deadline = Date.now() + 10 * 60_000;
  for (;;) {
    try {
      const res = await fetch(`https://registry.npmjs.org/${name.replace("/", "%2f")}`, {
        // The abbreviated metadata package managers install from.
        headers: { accept: "application/vnd.npm.install-v1+json; q=1.0, application/json; q=0.8, */*" },
      });
      const doc = res.ok ? ((await res.json()) as { versions?: Record<string, unknown> }) : {};
      if (doc.versions?.[version]) return;
    } catch {}
    if (Date.now() > deadline) {
      console.error(`npm does not serve ${name}@${version}: run the release again later`);
      process.exit(1);
    }
    console.log(`waiting for npm to serve ${name}@${version}`);
    await Bun.sleep(10_000);
  }
}
