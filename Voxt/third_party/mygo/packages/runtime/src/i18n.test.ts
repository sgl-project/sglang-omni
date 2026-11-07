import { afterAll, afterEach, describe, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  createCatalog,
  currentLocale,
  formatMessage,
  isPseudoLocale,
  MessageSyntaxError,
  parseMessage,
  skeletonOptions,
  type CatalogData,
} from "./i18n";

// The cases package i18n checks too (i18n/testdata/messages.json), so that
// Go and pages say the same.
const fixtures = JSON.parse(readFileSync(join(import.meta.dir, "../../../i18n/testdata/messages.json"), "utf8")) as {
  format: { message: string; locale?: string; args?: Record<string, unknown>; want: string }[];
  errors: string[];
  pseudo: { message: string; locale: string; args?: Record<string, unknown>; want: string }[];
};

describe("formatMessage", () => {
  test("has the fixtures", () => {
    expect(fixtures.format.length).toBeGreaterThan(50);
    expect(fixtures.errors.length).toBeGreaterThan(20);
  });
  for (const c of [...fixtures.format, ...fixtures.pseudo]) {
    test(`${c.message} in ${c.locale ?? "en"} with ${JSON.stringify(c.args ?? {})}`, () => {
      expect(formatMessage(c.locale ?? "en", c.message, c.args)).toBe(c.want);
    });
  }
  for (const s of fixtures.errors) {
    test(`${JSON.stringify(s)} does not parse`, () => {
      expect(() => parseMessage(s)).toThrow(MessageSyntaxError);
    });
  }

  test("formats numbers and dates with Intl", () => {
    expect(formatMessage("en", "{n}", { n: 1234.5 })).toBe("1,234.5");
    expect(formatMessage("de", "{n}", { n: 1234.5 })).toBe("1.234,5");
    expect(formatMessage("de", "{n, plural, one {# Datei} other {# Dateien}}", { n: 1000 })).toBe("1.000 Dateien");
    expect(formatMessage("en", "{n, number, ::currency/EUR}", { n: 3 })).toBe("€3.00");
    const day = new Date(Date.UTC(2026, 0, 2, 15, 4));
    expect(formatMessage("en", "{d, date, short}", { d: day })).toBe(
      new Intl.DateTimeFormat("en", { dateStyle: "short" }).format(day),
    );
    expect(formatMessage("en", "{d, time, short}", { d: day })).toBe(
      new Intl.DateTimeFormat("en", { timeStyle: "short" }).format(day),
    );
    expect(formatMessage("en", "{d}", { d: day })).toBe(
      new Intl.DateTimeFormat("en", { dateStyle: "short", timeStyle: "short" }).format(day),
    );
    expect(formatMessage("en", "{d, date}", { d: "tomorrow" })).toBe("tomorrow");
  });

  test("writes other values", () => {
    expect(formatMessage("en", "{e}", { e: new Error("boom") })).toBe("boom");
    expect(formatMessage("en", "{v}", { v: null })).toBe("");
    expect(formatMessage("en", "{n, plural, one {#} other {# x}}", { n: 2n })).toBe("2 x");
  });

  test("reports where a message does not parse", () => {
    try {
      parseMessage("Hello {name");
      throw new Error("parsed");
    } catch (err) {
      expect(err).toBeInstanceOf(MessageSyntaxError);
      expect((err as MessageSyntaxError).offset).toBe(11);
    }
  });
});

const data: CatalogData = {
  source: "en",
  languages: ["en", "de", "en-001", "pt", "pt-PT", "zh-Hans", "zh-Hant"],
  fallbacks: {
    en: ["en"],
    de: ["de", "en"],
    "en-001": ["en-001", "en"],
    pt: ["pt", "en"],
    "pt-PT": ["pt-PT", "pt", "en"],
    "zh-Hans": ["zh-Hans", "en"],
    "zh-Hant": ["zh-Hant", "en"],
  },
  params: { hello: ["name"], color: [], files: ["n"], broken: [] },
  messages: {
    en: {
      hello: "Hello, {name}!",
      color: "Color",
      files: "{n, plural, one {# file} other {# files}}",
      broken: "Fine in English",
    },
    "en-001": { color: "Colour" },
    de: { hello: "Hallo, {name}!", files: "{n, plural, one {# Datei} other {# Dateien}}", broken: "{kaputt" },
    "zh-Hans": { hello: "你好，{name}！" },
    "zh-Hant": { hello: "你好，{name}！", color: "顏色" },
    pt: { color: "Cor" },
    "pt-PT": { hello: "Olá, {name}!" },
  },
};

interface Messages {
  hello(name: string): string;
  color(): string;
  files(n: number): string;
  broken(): string;
}

describe("createCatalog", () => {
  const catalog = createCatalog<Messages>(data);
  const g = globalThis as { mygo?: unknown };
  afterEach(() => {
    delete g.mygo;
  });

  test("falls back along the languages", () => {
    const cases: [string, string, Record<string, unknown>, string][] = [
      ["en", "hello", { name: "Ada" }, "Hello, Ada!"],
      ["de", "hello", { name: "Ada" }, "Hallo, Ada!"],
      ["de-CH", "hello", { name: "Ada" }, "Hallo, Ada!"],
      ["de", "color", {}, "Color"],
      ["de", "broken", {}, "broken"],
      ["en-001", "color", {}, "Colour"],
      ["zh-TW", "color", {}, "顏色"],
      ["zh-Hant-HK", "color", {}, "顏色"],
      ["zh-CN", "color", {}, "Color"],
      ["pt-PT", "color", {}, "Cor"],
      ["fr", "hello", { name: "Ada" }, "Hello, Ada!"],
      ["de", "missing", {}, "missing"],
      // Properties every object has are no messages.
      ["de", "toString", {}, "toString"],
      ["de", "constructor", {}, "constructor"],
      ["de", "files", { n: 1 }, "1 Datei"],
      // English's plural rules for an English message: French says one for 0.
      ["fr", "files", { n: 0 }, "0 files"],
      ["en-XA", "color", {}, "[Çöļöŕ one]"],
      ["ar-XB", "color", {}, "\u200f\u202eColor\u202c\u200f"],
      ["", "color", {}, "Color"],
    ];
    for (const [locale, key, args, want] of cases) {
      expect(catalog.format(locale, key, args), `${locale} ${key}`).toBe(want);
    }
    expect(catalog.resolve("zh-TW")).toBe("zh-Hant");
    expect(catalog.resolve("en-AU")).toBe("en");
    expect(catalog.resolve("xx")).toBe("en");
  });

  test("binds the messages to the language the page shows", () => {
    const messages = catalog.messages();
    g.mygo = { locale: "de" };
    expect(currentLocale()).toBe("de");
    expect(messages.files(3)).toBe("3 Dateien");
    g.mygo = { locale: "zh-Hant" };
    expect(messages.hello("Ada")).toBe("你好，Ada！");
    expect(catalog.messagesIn("en").files(1)).toBe("1 file");
    expect(catalog.languages).toEqual(data.languages);
  });

  test("reads the browser's language outside MyGo", () => {
    delete g.mygo;
    expect(currentLocale()).toBe(navigator.languages?.[0] ?? navigator.language ?? "");
  });
});

test("isPseudoLocale", () => {
  expect(isPseudoLocale("en-XA")).toBe(true);
  expect(isPseudoLocale("ar-xb")).toBe(true);
  expect(isPseudoLocale("en")).toBe(false);
  expect(isPseudoLocale("not a tag")).toBe(false);
});

// The module of internal/i18ngen's golden catalog, as mygo generate writes
// it, importing this package's source in place of mygo-runtime/i18n: it
// type-checks, and says its messages.
describe("the module mygo generate writes", () => {
  const dir = mkdtempSync(join(tmpdir(), "mygo-locales-"));
  const file = join(dir, "locales.ts");
  const source = readFileSync(join(import.meta.dir, "../../../internal/i18ngen/testdata/app/locales.ts"), "utf8");
  expect(source).toContain(`from "mygo-runtime/i18n"`);
  writeFileSync(file, source.replace(`from "mygo-runtime/i18n"`, `from ${JSON.stringify(join(import.meta.dir, "i18n.ts"))}`));
  afterAll(() => rmSync(dir, { recursive: true, force: true }));

  test("type-checks", () => {
    const tsc = Bun.spawnSync(
      [
        "tsc", "--noEmit", "--strict", "--noUncheckedIndexedAccess", "--target", "ES2022", "--module", "ESNext",
        "--moduleResolution", "bundler", "--allowImportingTsExtensions", "--lib", "ES2022,DOM", "--skipLibCheck", file,
      ],
      { cwd: dir },
    );
    expect(tsc.stdout.toString() + tsc.stderr.toString()).toBe("");
    expect(tsc.exitCode).toBe(0);
  });

  test("says the messages", async () => {
    const locales = await import(file);
    expect(locales.languages).toEqual(["en", "de", "en-GB", "zh-Hant"]);
    const de = locales.messagesIn("de");
    expect(de.title()).toBe("Notizen");
    expect(de.filesSelected(0)).toBe("Keine Dateien ausgewählt");
    expect(de.filesSelected(1)).toBe("1 Datei ausgewählt");
    expect(de.progress(0.5)).toBe("50\u00a0% done");
    expect(locales.messagesIn("en-GB").progress(0.5)).toBe("50% complete");
    expect(locales.messagesIn("zh-TW").filesSelected(2)).toBe("已選取 2 個檔案");
    expect(locales.messagesIn("en").guests("Ada", 3)).toBe("Ada and 2 others");
    expect(locales.messagesIn("en").place(22)).toBe("You came 22nd");
    expect(locales.messagesIn("en").positional("a", "b")).toBe("a of b");
    expect(locales.messagesIn("en").keyword("x", "y", "z", "w")).toBe("x y z w");
    expect(locales.messagesIn("en-XA").title()).toBe("[Ñöţéš one]");
    const g = globalThis as { mygo?: unknown };
    g.mygo = { locale: "de" };
    try {
      expect(locales.messages.greeting("Ada")).toBe("Hallo, Ada!");
    } finally {
      delete g.mygo;
    }
  });
});

describe("skeletons and number stems", () => {
  const day = new Date(Date.UTC(2026, 0, 2, 15, 4));
  test("formats a date of a skeleton as Intl's options", () => {
    expect(skeletonOptions("yMMMd")).toEqual({ year: "numeric", month: "short", day: "numeric" });
    expect(skeletonOptions("jm")).toEqual({ hour: "numeric", minute: "numeric" });
    expect(skeletonOptions("Hmm")).toEqual({ hour: "numeric", hourCycle: "h23", minute: "2-digit" });
    expect(skeletonOptions("EEEEd")).toEqual({ weekday: "long", day: "numeric" });
    expect(skeletonOptions("Kms")).toEqual({ hour: "numeric", hourCycle: "h11", minute: "numeric", second: "numeric" });
    expect(formatMessage("en", "{d, date, ::yMMMM}", { d: day })).toBe(
      new Intl.DateTimeFormat("en", { year: "numeric", month: "long" }).format(day),
    );
  });
});
