/**
 * Formats the messages of a MyGo app's catalogs in its pages: messages of
 * ICU MessageFormat, as package i18n formats them in Go, with the plural
 * rules, numbers and dates of the browser's `Intl`. The module
 * `mygo generate` writes next to the TypeScript client (`locales.ts`) is
 * built on it:
 *
 * ```ts
 * import { messages } from "./locales";
 *
 * status.textContent = messages.filesSelected(3); // "3 files selected"
 * ```
 *
 * Use {@link formatMessage} directly to format a message of your own.
 *
 * @module
 */

/** A part of a parsed message. */
export type MessagePart =
  | string
  | { type: "pound" }
  | { type: "arg"; name: string }
  | { type: "number" | "date" | "time"; name: string; style: string }
  | {
      type: "plural" | "selectordinal" | "select";
      name: string;
      offset: number;
      cases: { key: string; parts: MessagePart[] }[];
    };

/** The error of a message that does not parse. */
export class MessageSyntaxError extends Error {
  constructor(
    /** The message. */
    readonly source: string,
    /** Where in it the error is, in UTF-16 code units. */
    readonly offset: number,
    reason: string,
  ) {
    super(`${reason} at offset ${offset} of ${JSON.stringify(source)}`);
    this.name = "MessageSyntaxError";
  }
}

const pluralKeywords = ["zero", "one", "two", "few", "many", "other"];

function isPatternSpace(c: string): boolean {
  return /^[\t\n\v\f\r \u0085\u200e\u200f\u2028\u2029]$/.test(c);
}

function isPatternSyntax(c: string): boolean {
  const r = c.codePointAt(0)!;
  if (r < 0x80) {
    return (r >= 0x21 && r <= 0x2f) || (r >= 0x3a && r <= 0x40) || (r >= 0x5b && r <= 0x5e) || r === 0x60 || (r >= 0x7b && r <= 0x7e);
  }
  return (
    (r >= 0xa1 && r <= 0xa7) || r === 0xa9 || r === 0xab || r === 0xac || r === 0xae || r === 0xb0 || r === 0xb1 ||
    r === 0xb6 || r === 0xbb || r === 0xbf || r === 0xd7 || r === 0xf7 ||
    (r >= 0x2010 && r <= 0x2027) || (r >= 0x2030 && r <= 0x203e) || (r >= 0x2041 && r <= 0x2053) ||
    (r >= 0x2055 && r <= 0x205e) || (r >= 0x2190 && r <= 0x245f) || (r >= 0x2500 && r <= 0x2775) ||
    (r >= 0x2794 && r <= 0x2bff) || (r >= 0x2e00 && r <= 0x2e7f) || (r >= 0x3001 && r <= 0x3003) ||
    (r >= 0x3008 && r <= 0x3020) || r === 0x3030 || r === 0xfd3e || r === 0xfd3f || r === 0xfe45 || r === 0xfe46
  );
}

const decimalRe = /^-?\d+(\.\d+)?$/;

/**
 * Parses a message of ICU MessageFormat, as package i18n's ParseMessage
 * does: the same messages parse, and fail, in Go and in pages.
 */
export function parseMessage(source: string): MessagePart[] {
  let i = 0;
  const s = source;
  const fail = (reason: string): never => {
    throw new MessageSyntaxError(source, i, reason);
  };
  const space = () => {
    while (i < s.length && isPatternSpace(s[i]!)) i++;
  };
  const word = (): string => {
    const start = i;
    while (i < s.length) {
      const c = String.fromCodePoint(s.codePointAt(i)!);
      if (isPatternSpace(c) || isPatternSyntax(c)) break;
      i += c.length;
    }
    return s.slice(start, i);
  };
  const validName = (name: string): boolean =>
    /^\d/.test(name) ? /^(0|[1-9]\d*)$/.test(name) : /^[\p{L}_]/u.test(name);

  const message = (depth: number, inPlural: boolean): MessagePart[] => {
    if (depth > 32) fail("arguments nest too deep");
    const parts: MessagePart[] = [];
    let text = "";
    const flush = () => {
      if (text !== "") parts.push(text);
      text = "";
    };
    while (i < s.length) {
      const c = s[i]!;
      if (c === "'") {
        i++;
        if (s[i] === "'") {
          text += "'";
          i++;
        } else if (i < s.length && (s[i] === "{" || s[i] === "}" || (s[i] === "#" && inPlural))) {
          while (i < s.length) {
            if (s[i] === "'") {
              if (s[i + 1] === "'") {
                text += "'";
                i += 2;
                continue;
              }
              i++;
              break;
            }
            text += s[i];
            i++;
          }
        } else {
          text += "'";
        }
      } else if (c === "{") {
        flush();
        parts.push(argument(depth));
      } else if (c === "}") {
        break;
      } else if (c === "#" && inPlural) {
        flush();
        parts.push({ type: "pound" });
        i++;
      } else {
        text += c;
        i++;
      }
    }
    flush();
    return parts;
  };

  const argument = (depth: number): MessagePart => {
    i++; // {
    space();
    const name = word();
    if (name === "") fail("an argument has no name");
    if (!validName(name)) fail(`${JSON.stringify(name)} is not an argument name`);
    space();
    if (i >= s.length) fail(`unclosed argument {${name}`);
    if (s[i] === "}") {
      i++;
      return { type: "arg", name };
    }
    if (s[i] !== ",") fail(`expected , or } after the argument name ${JSON.stringify(name)}`);
    i++;
    space();
    const type = word();
    space();
    if (type === "number" || type === "date" || type === "time") {
      let style = "";
      if (s[i] === ",") {
        i++;
        const start = i;
        let nest = 0;
        for (;;) {
          if (i >= s.length) fail("unclosed style");
          const c = s[i];
          if (c === "'") {
            const end = s.indexOf("'", i + 1);
            if (end < 0) fail("unclosed quote in a style");
            i = end + 1;
            continue;
          }
          if (c === "{") nest++;
          if (c === "}") {
            if (nest === 0) break;
            nest--;
          }
          i++;
        }
        style = s.slice(start, i).replace(/^[\s\u0085\u200e\u200f]+|[\s\u0085\u200e\u200f]+$/g, "");
        if (style === "") fail(`the ${type} argument ${JSON.stringify(name)} has an empty style`);
      }
      if (s[i] !== "}") fail(`unclosed argument {${name}, ${type}`);
      i++;
      return { type, name, style };
    }
    if (type === "") fail(`the argument ${JSON.stringify(name)} has no type`);
    if (type !== "plural" && type !== "selectordinal" && type !== "select") {
      fail(`unknown argument type ${JSON.stringify(type)} (want number, date, time, plural, selectordinal or select)`);
    }
    if (s[i] !== ",") fail(`expected , after ${type}`);
    i++;
    const plural = type !== "select";
    let offset = 0;
    space();
    if (plural && s.startsWith("offset:", i)) {
      i += "offset:".length;
      space();
      const start = i;
      while (i < s.length && s[i]! >= "0" && s[i]! <= "9") i++;
      if (start === i || i - start > 15) fail("offset: needs a number");
      offset = Number(s.slice(start, i));
    }
    const cases: { key: string; parts: MessagePart[] }[] = [];
    for (;;) {
      space();
      if (i >= s.length) fail(`unclosed ${type} argument ${JSON.stringify(name)}`);
      if (s[i] === "}") {
        i++;
        break;
      }
      let key: string;
      if (plural && s[i] === "=") {
        i++;
        const start = i;
        while (i < s.length && /[0-9.]/.test(s[i]!)) i++;
        const value = s.slice(start, i);
        if (!/^\d+(\.\d+)?$/.test(value)) fail(`=${value} is not a number`);
        key = "=" + value;
      } else {
        key = word();
        if (key === "") fail(`expected a case of the ${type} argument ${JSON.stringify(name)}`);
        if (plural && !pluralKeywords.includes(key)) {
          fail(`${JSON.stringify(key)} is not a plural category (zero, one, two, few, many, other)`);
        }
      }
      if (cases.some((c) => c.key === key)) fail(`the ${type} argument ${JSON.stringify(name)} has the case ${key} twice`);
      space();
      if (s[i] !== "{") fail(`expected { after the case ${key}`);
      i++;
      const parts = message(depth + 1, plural);
      if (i >= s.length) fail(`unclosed case ${key}`);
      i++; // }
      cases.push({ key, parts });
    }
    if (!cases.some((c) => c.key === "other")) fail(`the ${type} argument ${JSON.stringify(name)} has no other case`);
    return { type: type as "plural" | "selectordinal" | "select", name, offset, cases };
  };

  const parts = message(0, false);
  if (i < s.length) fail("unmatched }");
  return parts;
}

/** A number as a page formats it: its value, and its visible fraction digits. */
interface Num {
  value: number;
  /** The fraction digits a string of a decimal number shows, as 2 for "1.50". */
  digits?: number;
  /** The decimal number as written, for the cases of exact values. */
  text: string;
}

function toNum(v: unknown): Num | undefined {
  if (typeof v === "number") return Number.isFinite(v) ? { value: v, text: String(v) } : undefined;
  if (typeof v === "bigint") return { value: Number(v), text: v.toString() };
  if (typeof v === "string") {
    const t = v.trim();
    if (!decimalRe.test(t)) return undefined;
    return { value: Number(t), digits: t.includes(".") ? t.split(".")[1]!.length : 0, text: t };
  }
  return undefined;
}

/** Writes n less offset, with the fraction digits of n. */
function minus(n: Num, offset: number): Num {
  if (offset === 0) return n;
  const value = n.value - offset;
  if (n.digits !== undefined) return { value, digits: n.digits, text: value.toFixed(n.digits) };
  return { value, text: String(value) };
}

/** Whether a number written as text equals an exact case, "=1" for 1.0. */
function exactEquals(text: string, exact: string): boolean {
  if (text.startsWith("-")) return false;
  const norm = (s: string) => {
    let [i, f = ""] = s.split(".");
    i = i!.replace(/^0+(?=\d)/, "");
    f = f.replace(/0+$/, "");
    return f === "" ? i : `${i}.${f}`;
  };
  return /^\d+(\.\d+)?$/.test(text) && norm(text) === norm(exact);
}

const caches = new Map<string, Intl.PluralRules | Intl.NumberFormat | Intl.DateTimeFormat>();

function cached<T extends Intl.PluralRules | Intl.NumberFormat | Intl.DateTimeFormat>(key: string, make: () => T): T {
  let v = caches.get(key) as T | undefined;
  if (!v) {
    v = make();
    if (caches.size > 256) caches.clear();
    caches.set(key, v);
  }
  return v;
}

function safeLocale(locale: string): string | undefined {
  try {
    return Intl.getCanonicalLocales(locale)[0];
  } catch {
    return undefined;
  }
}

function pluralCategory(locale: string, n: Num, type: "cardinal" | "ordinal"): string {
  const digits = n.digits;
  const pr = cached(`p|${locale}|${type}|${digits ?? ""}`, () =>
    new Intl.PluralRules(safeLocale(locale), digits === undefined ? { type } : { type, minimumFractionDigits: digits, maximumFractionDigits: digits }),
  );
  return (pr as Intl.PluralRules).select(Math.abs(n.value));
}

/** The hooks that write numbers and dates, which tests and apps may replace. */
export const formatters = {
  /**
   * Writes a number in an ICU number style: "" (the default), "integer",
   * "percent", or a skeleton ("::percent .00"), of which this formatter
   * reads percent, integer, precision-integer, the precision of fraction
   * digits (.00, .0#, .##), group-off and currency/, and writes the rest
   * as the default. Package i18n reads the same, but currencies.
   */
  number(locale: string, n: number, style: string, digits?: number): string {
    const options: Intl.NumberFormatOptions & { roundingMode?: string } = {};
    const stems = style.startsWith("::") ? style.slice(2).trim().split(/\s+/) : [style];
    let fraction = false;
    for (const stem of stems) {
      if (stem === "integer" || stem === "precision-integer") {
        options.maximumFractionDigits = 0;
        options.roundingMode = "halfEven";
      } else if (stem === "percent") {
        options.style = "percent";
        options.roundingMode = "halfEven";
      } else if (stem === "group-off") {
        options.useGrouping = false;
      } else if (/^\.[0#]+$/.test(stem)) {
        options.minimumFractionDigits = stem.split("0").length - 1;
        options.maximumFractionDigits = stem.length - 1;
        fraction = true;
      } else if (stem.startsWith("currency/")) {
        options.style = "currency";
        options.currency = stem.slice("currency/".length);
      }
    }
    if (digits !== undefined && !fraction && options.style === undefined && options.maximumFractionDigits === undefined) {
      options.minimumFractionDigits = options.maximumFractionDigits = Math.min(digits, 100);
    }
    const nf = cached(`n|${locale}|${JSON.stringify(options)}`, () => new Intl.NumberFormat(safeLocale(locale), options));
    return (nf as Intl.NumberFormat).format(n);
  },
  /**
   * Writes a time: its date, its time or both ("datetime"), in an ICU
   * style, "short", "medium" (the default), "long" or "full", or a
   * skeleton of CLDR ("::yMMMd"), which Intl finds the locale's pattern
   * of as package i18n does.
   */
  date(locale: string, d: Date, kind: "date" | "time" | "datetime", style: string): string {
    const s = (["short", "medium", "long", "full"].includes(style) ? style : "medium") as "short" | "medium" | "long" | "full";
    const options: Intl.DateTimeFormatOptions = style.startsWith("::")
      ? skeletonOptions(style.slice(2).trim())
      : kind === "date"
        ? { dateStyle: s }
        : kind === "time"
          ? { timeStyle: s }
          : { dateStyle: "short", timeStyle: "short" };
    const df = cached(`d|${locale}|${JSON.stringify(options)}`, () => new Intl.DateTimeFormat(safeLocale(locale), options));
    return (df as Intl.DateTimeFormat).format(d);
  },
};

/**
 * The options of Intl.DateTimeFormat of a skeleton of CLDR, "yMMMd" or
 * "jm": V8 makes the skeleton back of them, and finds its pattern with
 * ICU's DateTimePatternGenerator. An hour letter other than j is an
 * hourCycle, as in package i18n. EEEEEE, CLDR's short weekday ("Mo"),
 * has no option of Intl: it is the abbreviated one ("Mon"), where package
 * i18n writes the short one.
 */
export function skeletonOptions(skeleton: string): Intl.DateTimeFormatOptions {
  const o: Intl.DateTimeFormatOptions = {};
  for (const [field] of skeleton.matchAll(/([A-Za-z])\1*/g)) {
    const n = field.length;
    switch (field[0]) {
      case "y":
        o.year = n === 2 ? "2-digit" : "numeric";
        break;
      case "M":
      case "L":
        o.month = n === 1 ? "numeric" : n === 2 ? "2-digit" : n === 3 ? "short" : n === 4 ? "long" : "narrow";
        break;
      case "d":
        o.day = n === 2 ? "2-digit" : "numeric";
        break;
      case "E":
      case "c":
        o.weekday = n === 4 ? "long" : n === 5 ? "narrow" : "short";
        break;
      case "j":
      case "h":
      case "H":
      case "K":
      case "k":
        o.hour = n === 2 ? "2-digit" : "numeric";
        if (field[0] === "h") o.hourCycle = "h12";
        if (field[0] === "H") o.hourCycle = "h23";
        if (field[0] === "K") o.hourCycle = "h11";
        if (field[0] === "k") o.hourCycle = "h24";
        break;
      case "m":
        o.minute = n === 2 ? "2-digit" : "numeric";
        break;
      case "s":
        o.second = n === 2 ? "2-digit" : "numeric";
        break;
      case "z":
      case "v":
        o.timeZoneName = n >= 4 ? "long" : "short";
        break;
    }
  }
  return o;
}

function valueString(locale: string, v: unknown): string {
  if (typeof v === "string") return v;
  if (v instanceof Date) return formatters.date(locale, v, "datetime", "short");
  if (v === null || v === undefined) return "";
  if (typeof v === "number" || typeof v === "bigint") {
    const n = toNum(v);
    return n ? formatters.number(locale, n.value, "", n.digits) : String(v);
  }
  if (v instanceof Error) return v.message;
  return String(v);
}

/** Transforms the text of a message for a pseudo locale. */
interface Pseudo {
  text(s: string): string;
  placeholder(s: string): string;
  wrap(s: string): string;
}

const accented: Record<string, string> = {
  a: "å", b: "ƀ", c: "ç", d: "ð", e: "é", f: "ƒ", g: "ĝ", h: "ĥ", i: "î", j: "ĵ", k: "ķ", l: "ļ", m: "ɱ",
  n: "ñ", o: "ö", p: "þ", q: "ǫ", r: "ŕ", s: "š", t: "ţ", u: "û", v: "ṽ", w: "ŵ", x: "ẋ", y: "ý", z: "ž",
  A: "Å", B: "Ɓ", C: "Ç", D: "Ð", E: "É", F: "Ƒ", G: "Ĝ", H: "Ĥ", I: "Î", J: "Ĵ", K: "Ķ", L: "Ļ", M: "Ṁ",
  N: "Ñ", O: "Ö", P: "Þ", Q: "Ǫ", R: "Ŕ", S: "Š", T: "Ţ", U: "Û", V: "Ṽ", W: "Ŵ", X: "Ẋ", Y: "Ý", Z: "Ž",
};

const padWords = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split(" ");

const RLM = "\u200f";
const RLO = "\u202e";
const PDF = "\u202c";

/** The pseudo locales, which test layouts without translations. */
export const pseudoLocales = ["en-XA", "ar-XB"] as const;

function canonical(tag: string): string {
  return safeLocale(tag.replace(/_/g, "-").split(/[.@]/)[0]!) ?? "";
}

/** Strips the extensions of a canonical tag, as "-u-ca-buddhist". */
function stripExtensions(tag: string): string {
  const parts = tag.split("-");
  const i = parts.findIndex((p, i) => i > 0 && p.length === 1);
  return i < 0 ? tag : parts.slice(0, i).join("-");
}

/** Reports whether tag is a pseudo locale: en-XA or ar-XB. */
export function isPseudoLocale(tag: string): boolean {
  return (pseudoLocales as readonly string[]).includes(stripExtensions(canonical(tag)));
}

function pseudoFor(locale: string): Pseudo | undefined {
  switch (stripExtensions(canonical(locale))) {
    case "en-XA": {
      let length = 0;
      return {
        text(s) {
          let out = "";
          for (const c of s) {
            out += accented[c] ?? c;
            length++;
          }
          return out;
        },
        placeholder: (s) => s,
        wrap(s) {
          const pad = Math.floor((length * 2 + 4) / 5);
          let out = "[" + s;
          for (let i = 0, n = 0; n < pad; i++) {
            const w = padWords[i % padWords.length]!;
            out += " " + w;
            n += 1 + w.length;
          }
          return out + "]";
        },
      };
    }
    case "ar-XB":
      return {
        text: (s) => s.replace(/[^\s]+/gu, (w) => RLM + RLO + w + PDF + RLM),
        placeholder: (s) => RLM + s + RLM,
        wrap: (s) => s,
      };
  }
  return undefined;
}

function formatParts(
  parts: MessagePart[],
  locale: string,
  pluralLocale: string,
  args: Record<string, unknown>,
  pseudo: Pseudo | undefined,
  pound: { num?: Num; raw?: unknown } | undefined,
): string {
  let out = "";
  const placeholder = (s: string) => (pseudo ? pseudo.placeholder(s) : s);
  for (const part of parts) {
    if (typeof part === "string") {
      out += pseudo ? pseudo.text(part) : part;
      continue;
    }
    if (part.type === "pound") {
      if (!pound) {
        out += "#";
      } else if (pound.num) {
        out += placeholder(formatters.number(locale, pound.num.value, "", pound.num.digits));
      } else {
        out += placeholder(valueString(locale, pound.raw));
      }
      continue;
    }
    if (!Object.prototype.hasOwnProperty.call(args, part.name)) {
      out += `{${part.name}}`;
      continue;
    }
    const v = args[part.name];
    switch (part.type) {
      case "arg":
        out += placeholder(valueString(locale, v));
        break;
      case "number": {
        const n = toNum(v);
        out += placeholder(n ? formatters.number(locale, n.value, part.style, n.digits) : valueString(locale, v));
        break;
      }
      case "date":
      case "time":
        out += placeholder(v instanceof Date ? formatters.date(locale, v, part.type, part.style) : valueString(locale, v));
        break;
      case "select": {
        const key = typeof v === "string" ? v : typeof v === "number" ? (toNum(v)?.text ?? String(v)) : String(v);
        const c = part.cases.find((c) => c.key === key) ?? part.cases.find((c) => c.key === "other")!;
        out += formatParts(c.parts, locale, pluralLocale, args, pseudo, pound);
        break;
      }
      case "plural":
      case "selectordinal": {
        const n = toNum(v);
        const other = part.cases.find((c) => c.key === "other")!;
        if (!n) {
          out += formatParts(other.parts, locale, pluralLocale, args, pseudo, { raw: v });
          break;
        }
        let chosen = part.cases.find((c) => c.key.startsWith("=") && exactEquals(n.text, c.key.slice(1)));
        const shown = minus(n, part.offset);
        if (!chosen) {
          const cat = pluralCategory(pluralLocale, shown, part.type === "plural" ? "cardinal" : "ordinal");
          chosen = part.cases.find((c) => c.key === cat) ?? other;
        }
        out += formatParts(chosen.parts, locale, pluralLocale, args, pseudo, { num: shown });
        break;
      }
    }
  }
  return out;
}

const parsedCache = new Map<string, MessagePart[]>();

function parsed(message: string): MessagePart[] {
  let parts = parsedCache.get(message);
  if (!parts) {
    parts = parseMessage(message);
    parsedCache.set(message, parts);
  }
  return parts;
}

/**
 * Formats a message of ICU MessageFormat with the values of its arguments
 * for locale, as package i18n's Message.Format does: its plural rules
 * pick the cases, its conventions write numbers and dates, and the pseudo
 * locales en-XA and ar-XB transform the message. An argument args lacks is
 * written as it is, `{name}`. Throws a {@link MessageSyntaxError} for a
 * message that does not parse.
 */
export function formatMessage(locale: string, message: string, args: Record<string, unknown> = {}): string {
  const pseudo = pseudoFor(locale);
  return format(parsed(message), pseudo ? "en" : locale, pseudo ? "en" : locale, args, pseudo);
}

function format(parts: MessagePart[], locale: string, pluralLocale: string, args: Record<string, unknown>, pseudo: Pseudo | undefined): string {
  const out = formatParts(parts, locale, pluralLocale, args, pseudo, undefined);
  return pseudo ? pseudo.wrap(out) : out;
}

/** The data of a catalog, as the module `mygo generate` writes has it. */
export interface CatalogData {
  /** The source language, which every message falls back to. */
  source: string;
  /** The languages of the catalog: the source, then the others. */
  languages: string[];
  /** For each language, the languages it falls back along: itself, its parents, the source. */
  fallbacks: Record<string, string[]>;
  /** The names of the arguments of each key, in the order its function takes them. */
  params: Record<string, string[]>;
  /** The messages of each language by key. */
  messages: Record<string, Record<string, string>>;
}

/** A catalog of messages in a page, which {@link createCatalog} returns. */
export interface Catalog<M> {
  /** The languages of the catalog: the source, then the others. */
  readonly languages: readonly string[];
  /** Formats the message of key in locale, or returns key without one. */
  format(locale: string, key: string, args?: Record<string, unknown>): string;
  /** Returns the language of the catalog whose messages locale says. */
  resolve(locale: string): string;
  /** The messages in the language the page shows, read at each call. */
  messages(): M;
  /** The messages in the language of locale. */
  messagesIn(locale: string): M;
}

/**
 * Returns the language the page shows: the app's, which the MyGo runtime
 * holds (`window.mygo.locale`, set before the page's scripts run), else
 * the browser's first preferred language.
 */
export function currentLocale(): string {
  const rt = (globalThis as { mygo?: { locale?: unknown } }).mygo;
  const locale = typeof rt?.locale === "function" ? (rt.locale as () => unknown)() : rt?.locale;
  if (typeof locale === "string" && locale !== "") return locale;
  if (typeof navigator !== "undefined") return navigator.languages?.[0] ?? navigator.language ?? "";
  return "";
}

/** Creates the catalog of a module `mygo generate` wrote. */
export function createCatalog<M>(data: CatalogData): Catalog<M> {
  const resolved = new Map<string, string>();
  const resolve = (locale: string): string => {
    let r = resolved.get(locale);
    if (r !== undefined) return r;
    r = data.source;
    const tag = stripExtensions(canonical(locale));
    if (data.languages.includes(tag)) {
      r = tag;
    } else if (tag !== "" && !isPseudoLocale(tag)) {
      // The tag without its last subtags, then a language written in the
      // same script, as zh-Hant for zh-TW.
      const found = truncations(tag).find((t) => data.languages.includes(t)) ?? sameScript(tag, data.languages);
      if (found) r = found;
    }
    resolved.set(locale, r);
    return r;
  };
  const has = (o: object, k: string): boolean => Object.prototype.hasOwnProperty.call(o, k);
  const lookup = (locale: string, key: string): [string, string] | undefined => {
    const lang = isPseudoLocale(locale) ? data.source : resolve(locale);
    for (const t of data.fallbacks[lang] ?? [data.source]) {
      // Own keys only: a key named as a property of every object, as
      // "toString", is a message only where the catalog has it.
      const messages = has(data.messages, t) ? data.messages[t] : undefined;
      const m = messages && has(messages, key) ? messages[key] : undefined;
      if (m !== undefined) return [m, t];
    }
    return undefined;
  };
  const formatKey = (locale: string, key: string, args: Record<string, unknown> = {}): string => {
    const found = lookup(locale, key);
    if (!found) return key;
    const [message, lang] = found;
    let parts: MessagePart[];
    try {
      parts = parsed(message);
    } catch {
      return key;
    }
    const pseudo = pseudoFor(locale);
    const numbers = pseudo || locale === "" ? lang : locale;
    return format(parts, numbers, lang, args, pseudo);
  };
  const bind = (locale: () => string): M => {
    const out: Record<string, (...values: unknown[]) => string> = {};
    for (const [key, names] of Object.entries(data.params)) {
      out[key] = (...values: unknown[]) => {
        const args: Record<string, unknown> = {};
        names.forEach((name, i) => (args[name] = values[i]));
        return formatKey(locale(), key, args);
      };
    }
    return out as M;
  };
  return {
    languages: data.languages,
    format: formatKey,
    resolve,
    messages: () => bind(currentLocale),
    messagesIn: (locale) => bind(() => locale),
  };
}

function truncations(tag: string): string[] {
  const parts = tag.split("-");
  const out: string[] = [];
  for (let n = parts.length; n > 0; n--) out.push(parts.slice(0, n).join("-"));
  return out;
}

/** The first of langs in the language and script of tag. */
function sameScript(tag: string, langs: string[]): string | undefined {
  try {
    const want = new Intl.Locale(tag).maximize();
    const same = langs.filter((l) => {
      const have = new Intl.Locale(l).maximize();
      return have.language === want.language && have.script === want.script;
    });
    return same.find((l) => new Intl.Locale(l).maximize().region === want.region) ?? same[0];
  } catch {
    return undefined;
  }
}
