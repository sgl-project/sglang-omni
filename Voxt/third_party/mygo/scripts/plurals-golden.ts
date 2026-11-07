// Writes i18n/testdata/plurals_icu.json: the plural categories ICU
// (Intl.PluralRules) gives numbers in every language of CLDR's plural rules,
// cardinal and ordinal, which TestPluralsMatchICU checks package i18n's
// compiled rules against. Run it again after updating CLDR, with a Node.js
// whose ICU has the same CLDR (`node -p process.versions.cldr`):
//
//   node scripts/plurals-golden.ts
//
// Node.js, not Bun: Bun on macOS uses the system's ICU, whose data Apple
// changes (it gives bg, af and es ordinal categories CLDR does not have).
//
// The languages are those of i18n/testdata/plural_samples.txt, which
// `go generate ./i18n` writes; the numbers are integers and decimals
// written with zero to three fraction digits, whose trailing zeros count.
import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";

const dir = join(import.meta.dirname, "..", "i18n", "testdata");

/** The numbers, as written: their fraction digits are visible ones. */
export const numbers: string[] = [];
for (let i = 0; i <= 120; i++) numbers.push(String(i));
for (const n of [
  200, 201, 202, 203, 205, 211, 212, 1000, 1001, 1002, 1011, 1100, 10000, 20000, 40000, 100000, 100001,
  1000000, 1000001, 2000000, 10000000, 100000000, 1000000000,
]) {
  numbers.push(String(n));
}
for (let i = 0; i <= 30; i++) numbers.push((i / 10).toFixed(1));
for (const s of ["0.00", "0.01", "0.05", "0.10", "0.11", "0.25", "1.00", "1.01", "1.10", "1.25", "1.50", "2.00",
  "2.05", "3.00", "5.30", "10.10", "11.00", "21.00", "101.01", "0.000", "1.000", "1.005", "2.500", "1000000.0"]) {
  numbers.push(s);
}

const initials: Record<string, string> = { zero: "z", one: "o", two: "t", few: "f", many: "m", other: "x" };

const samples = await readFile(join(dir, "plural_samples.txt"), "utf8");
const langs = { cardinal: new Set<string>(), ordinal: new Set<string>() };
for (const line of samples.split("\n")) {
  if (line === "" || line.startsWith("#")) continue;
  const [kind, list] = line.split(" ") as [keyof typeof langs, string];
  for (const l of list.split(",")) langs[kind].add(l);
}

const out: Record<string, Record<string, string>> = { cardinal: {}, ordinal: {} };
for (const kind of ["cardinal", "ordinal"] as const) {
  for (const lang of [...langs[kind]].sort()) {
    // Languages ICU does not have would get the root's rules.
    if (Intl.PluralRules.supportedLocalesOf([lang]).length === 0) continue;
    const rules = new Map<number, Intl.PluralRules>();
    let cats = "";
    for (const n of numbers) {
      const v = n.includes(".") ? n.split(".")[1]!.length : 0;
      let pr = rules.get(v);
      if (!pr) {
        pr = new Intl.PluralRules(lang, {
          type: kind,
          minimumFractionDigits: v,
          maximumFractionDigits: v,
        });
        rules.set(v, pr);
      }
      cats += initials[pr.select(Number(n))];
    }
    out[kind]![lang] = cats;
  }
}

const file = join(dir, "plurals_icu.json");
const versions = process.versions as Record<string, string | undefined>;
const icu = `${versions.icu ?? "unknown"} (CLDR ${versions.cldr ?? "unknown"})`;
// A language a line, the numbers on one.
const map = (m: Record<string, string>) =>
  "{\n" + Object.entries(m).map(([k, v]) => `  ${JSON.stringify(k)}: ${JSON.stringify(v)}`).join(",\n") + "\n }";
await writeFile(
  file,
  `{\n "icu": ${JSON.stringify(icu)},\n "numbers": ${JSON.stringify(numbers)},\n "cardinal": ${map(out.cardinal!)},\n "ordinal": ${map(out.ordinal!)}\n}\n`,
);
console.log(`wrote ${file}: ICU ${icu}, ${Object.keys(out.cardinal!).length} cardinal and ${Object.keys(out.ordinal!).length} ordinal languages`);
