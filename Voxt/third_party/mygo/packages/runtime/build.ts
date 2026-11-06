// Builds dist/ of the mygo-runtime package: an ES module and its type
// declarations. dist/ is not committed: `bun run build` at the root builds
// it, as CI and releases do.
import { rm } from "node:fs/promises";
import { join } from "node:path";

const dir = import.meta.dir;
await rm(join(dir, "dist"), { recursive: true, force: true });
const result = await Bun.build({
  entrypoints: [join(dir, "src/index.ts"), join(dir, "src/i18n.ts")],
  outdir: join(dir, "dist"),
  format: "esm",
  target: "browser",
});
if (!result.success) {
  for (const log of result.logs) console.error(log);
  process.exit(1);
}
const tsc = Bun.spawnSync(["tsc", "-p", join(dir, "tsconfig.json")], { stdout: "inherit", stderr: "inherit" });
if (tsc.exitCode !== 0) process.exit(1);
console.log(`wrote ${join(dir, "dist")}`);
