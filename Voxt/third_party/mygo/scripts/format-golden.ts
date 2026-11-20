// Writes i18n/testdata/format_icu.json: how ICU (Intl.NumberFormat,
// Intl.DateTimeFormat, Intl.DisplayNames, Intl.Locale) writes numbers,
// dates and times, the first day of the week and the hour cycle, and
// the names of languages in themselves, in every locale package i18n has formatting
// data of, which TestFormatMatchesICU checks package i18n against. Run it
// again after updating CLDR, with a Node.js whose ICU has the same CLDR
// (`node -p process.versions.cldr`):
//
//   node scripts/format-golden.ts
//
// Node.js, not Bun: Bun on macOS uses the system's ICU, whose data Apple
// changes.
//
// Dates are in the Gregorian calendar (-u-ca-gregory), which package i18n
// formats in, and in UTC unless a case names a zone. The locales are those
// of i18n/testdata/format_locales.json, which `go generate ./i18n` writes.
import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";

const dir = join(import.meta.dirname, "..", "i18n", "testdata");
const { locales, autonyms } = JSON.parse(await readFile(join(dir, "format_locales.json"), "utf8")) as {
  locales: string[];
  autonyms: string[];
};

type NumberCase = { kind: "number"; value: string; options: Intl.NumberFormatOptions & { roundingMode?: string } };
type DateCase = { kind: "date"; time: string; zone?: string; options: Intl.DateTimeFormatOptions };
type NamesCase = { kind: "names"; times: string[]; options: Intl.DateTimeFormatOptions };
type Case = NumberCase | DateCase | NamesCase | { kind: "firstDay" | "hourCycle" };

const cases: Case[] = [];

// Numbers.
const decimals = ["0", "1", "-1", "1.5", "0.001234", "1234.5", "-1234567.891", "12345", "123456789", "1000000",
  "0.5", "2.5", "1.0005", "99999.9999", "123456789012345680000", "-0.0001"];
for (const value of decimals) cases.push({ kind: "number", value, options: {} });
for (const value of ["0", "0.256", "-0.5", "1.5", "12.345", "1234.56"]) cases.push({ kind: "number", value, options: { style: "percent" } });
for (const value of ["1", "1234.5", "-0.125"]) {
  cases.push({ kind: "number", value, options: { minimumFractionDigits: 2, maximumFractionDigits: 2 } });
}
for (const value of ["2.5", "3.5", "-2.5"]) {
  cases.push({ kind: "number", value, options: { maximumFractionDigits: 0, roundingMode: "halfEven" } });
}
cases.push({ kind: "number", value: "1234567.891", options: { useGrouping: false } });
cases.push({ kind: "number", value: "1234.5", options: { minimumFractionDigits: 3 } });

// Dates and times.
const times = ["2026-01-05T13:07:09Z", "2025-12-29T00:07:09Z", "2024-07-14T12:00:00Z", "1999-10-31T06:30:05Z"];
const styles = ["full", "long", "medium", "short"] as const;
for (const time of times.slice(0, 2)) {
  for (const s of styles) cases.push({ kind: "date", time, options: { dateStyle: s } });
}
for (const time of times) {
  for (const s of styles) cases.push({ kind: "date", time, options: { timeStyle: s } });
}
for (const s of styles) cases.push({ kind: "date", time: times[0]!, options: { dateStyle: s, timeStyle: s } });
cases.push({ kind: "date", time: times[0]!, options: { dateStyle: "full", timeStyle: "short" } });
cases.push({ kind: "date", time: times[0]!, options: { dateStyle: "short", timeStyle: "medium" } });
for (const hourCycle of ["h12", "h23"] as const) {
  for (const time of [times[0]!, times[1]!]) {
    for (const s of ["short", "medium"] as const) cases.push({ kind: "date", time, options: { timeStyle: s, hourCycle } });
    cases.push({ kind: "date", time, options: { hour: "numeric", minute: "2-digit", hourCycle } });
    cases.push({ kind: "date", time, options: { hour: "numeric", minute: "2-digit", second: "2-digit", hourCycle } });
  }
}
// The skeletons the controls and messages use, as Intl's options.
const skeletons: Intl.DateTimeFormatOptions[] = [
  { year: "numeric", month: "long" }, // yMMMM, a calendar's title
  { year: "numeric", month: "short" }, // yMMM
  { month: "short", day: "numeric" }, // MMMd
  { month: "long", day: "numeric" }, // MMMMd
  { year: "numeric", month: "numeric", day: "numeric" }, // yMd
  { year: "numeric", month: "short", day: "numeric" }, // yMMMd
  { year: "numeric", month: "long", day: "numeric", weekday: "long" }, // yMMMMEEEEd
  { year: "numeric", month: "short", day: "numeric", weekday: "short" }, // yMMMEEEd
  { year: "numeric", month: "2-digit", day: "2-digit" }, // yMMdd
  { year: "numeric", month: "numeric" }, // yM
  { month: "short", day: "numeric", weekday: "short" }, // MMMEEEd
  { day: "numeric", weekday: "short" }, // EEEd
  { weekday: "long" }, // EEEE
  { month: "long" }, // MMMM
  { year: "numeric" }, // y
  { day: "numeric" }, // d
  { hour: "numeric", minute: "2-digit" }, // jmm
  { hour: "numeric", minute: "2-digit", second: "2-digit" }, // jmmss
  { hour: "numeric" }, // j
  { minute: "2-digit", second: "2-digit" }, // mmss
  { year: "numeric", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }, // yMMMdjmm
  { year: "numeric", month: "long", day: "numeric", hour: "numeric", minute: "2-digit" }, // yMMMMdjmm
  { hour: "numeric", minute: "2-digit", timeZoneName: "short" }, // jmmz
];
for (const options of skeletons) {
  for (const time of times) cases.push({ kind: "date", time, options });
}
// Offsets from GMT.
for (const zone of ["Etc/GMT-8", "Etc/GMT+5"]) {
  for (const s of ["full", "long"] as const) cases.push({ kind: "date", time: times[0]!, zone, options: { timeStyle: s } });
}

// The names of months and weekdays, as dates of every month and weekday
// write them: Intl gives no list of them but its patterns'.
const monthTimes = Array.from({ length: 12 }, (_, m) => new Date(Date.UTC(2023, m, 15)).toISOString());
const dayTimes = Array.from({ length: 7 }, (_, d) => new Date(Date.UTC(2023, 0, 1 + d)).toISOString());
for (const month of ["long", "short", "narrow"] as const) {
  cases.push({ kind: "names", times: monthTimes, options: { month } });
  cases.push({ kind: "names", times: monthTimes, options: { month, day: "numeric" } });
}
for (const weekday of ["long", "short", "narrow"] as const) {
  cases.push({ kind: "names", times: dayTimes, options: { weekday } });
}
cases.push({ kind: "names", times: dayTimes, options: { weekday: "long", year: "numeric", month: "long", day: "numeric" } });
cases.push({ kind: "firstDay" }, { kind: "hourCycle" });

const hourCycles: Record<string, number> = { h11: 12, h12: 12, h23: 24, h24: 24 };

const out: Record<string, string[]> = {};
for (const locale of locales) {
  const row: string[] = [];
  for (const c of cases) {
    switch (c.kind) {
      case "number":
        row.push(new Intl.NumberFormat(locale, c.options).format(c.value as unknown as number));
        break;
      case "date":
        row.push(new Intl.DateTimeFormat(`${locale}-u-ca-gregory`, { timeZone: c.zone ?? "UTC", ...c.options }).format(new Date(c.time)));
        break;
      case "names": {
        const f = new Intl.DateTimeFormat(`${locale}-u-ca-gregory`, { timeZone: "UTC", ...c.options });
        row.push(c.times.map((t) => f.format(new Date(t))).join("|"));
        break;
      }
      case "firstDay": {
        const info = (new Intl.Locale(locale) as Intl.Locale & { getWeekInfo(): { firstDay: number } }).getWeekInfo();
        row.push(String(info.firstDay % 7));
        break;
      }
      case "hourCycle":
        row.push(String(hourCycles[new Intl.DateTimeFormat(locale, { hour: "numeric" }).resolvedOptions().hourCycle!]));
        break;
    }
  }
  out[locale] = row;
}
const names_: Record<string, string> = {};
for (const tag of autonyms) names_[tag] = new Intl.DisplayNames([tag], { type: "language" }).of(tag) ?? "";

const versions = process.versions as Record<string, string | undefined>;
const icu = `${versions.icu ?? "unknown"} (CLDR ${versions.cldr ?? "unknown"})`;
const lines = (m: Record<string, unknown>) =>
  "{\n" + Object.entries(m).map(([k, v]) => `  ${JSON.stringify(k)}: ${JSON.stringify(v)}`).join(",\n") + "\n }";
const file = join(dir, "format_icu.json");
await writeFile(
  file,
  `{\n "icu": ${JSON.stringify(icu)},\n "cases": [\n${cases.map((c) => "  " + JSON.stringify(c)).join(",\n")}\n ],\n "locales": ${lines(out)},\n "autonyms": ${lines(names_)}\n}\n`,
);
console.log(`wrote ${file}: ICU ${icu}, ${locales.length} locales, ${cases.length} cases`);
