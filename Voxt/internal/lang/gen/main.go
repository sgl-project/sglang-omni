//go:build ignore

// Command gen builds internal/lang/languages.tsv and aliases.tsv from:
//
//   - OmniVoice's language list (Apache-2.0):
//     https://raw.githubusercontent.com/k2-fsa/OmniVoice/HEAD/docs/lang_id_name_map.tsv
//   - Whisper's LANGUAGES table (MIT):
//     https://raw.githubusercontent.com/openai/whisper/main/whisper/tokenizer.py
//   - Unicode CLDR 48.2 JSON (Unicode licence), npm packages
//     cldr-localenames-full and cldr-core: native and Chinese names,
//     likely script and region, macrolanguage and legacy aliases, RTL scripts.
//
// Usage (from this directory, with the downloads unpacked under $D):
//
//	go run main.go -omni $D/lang_id_name_map.tsv -whisper $D/tokenizer.py \
//	    -names $D/cldr-localenames-full/package/main -core $D/cldr-core/package -out ..
package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"
)

type row struct {
	code, iso3, name, native, zh, script, region, macro, models, whisper string
	rtl                                                                  bool
}

func main() {
	omni := flag.String("omni", "", "OmniVoice lang_id_name_map.tsv")
	whisper := flag.String("whisper", "", "whisper/tokenizer.py")
	names := flag.String("names", "", "cldr-localenames-full/package/main")
	core := flag.String("core", "", "cldr-core/package")
	out := flag.String("out", "..", "output directory")
	flag.Parse()

	var likely map[string]string
	readJSON(filepath.Join(*core, "supplemental/likelySubtags.json"), &likely, "supplemental", "likelySubtags")
	var aliases map[string]map[string]string
	readJSON(filepath.Join(*core, "supplemental/aliases.json"), &aliases, "supplemental", "metadata", "alias", "languageAlias")
	var scripts map[string]map[string]any
	readJSON(filepath.Join(*core, "scriptMetadata.json"), &scripts, "scriptMetadata")
	rtl := map[string]bool{}
	for k, v := range scripts {
		if v["rtl"] == "YES" {
			rtl[k] = true
		}
	}
	names3 := map[string]string{} // CLDR alias reason "overlong": 3-letter → 2-letter
	for k, v := range aliases {
		if v["_reason"] == "overlong" && len(k) == 3 && !strings.Contains(v["_replacement"], "-") {
			names3[v["_replacement"]] = k
		}
	}
	localName := func(locale, code string) string {
		var m map[string]string
		p := filepath.Join(*names, locale, "languages.json")
		if _, err := os.Stat(p); err != nil {
			return ""
		}
		readJSON(p, &m, "main", locale, "localeDisplayNames", "languages")
		return m[code]
	}
	var zhNames map[string]string
	readJSON(filepath.Join(*names, "zh", "languages.json"), &zhNames, "main", "zh", "localeDisplayNames", "languages")
	var enNames map[string]string
	readJSON(filepath.Join(*names, "en", "languages.json"), &enNames, "main", "en", "localeDisplayNames", "languages")

	byCode := map[string]*row{}
	var rows []*row
	// OmniVoice.
	f, err := os.Open(*omni)
	if err != nil {
		log.Fatal(err)
	}
	r := csv.NewReader(f)
	r.Comma = '\t'
	recs, err := r.ReadAll()
	if err != nil {
		log.Fatal(err)
	}
	for _, rec := range recs[1:] {
		x := &row{code: rec[0], name: rec[1], iso3: rec[2], models: "o"}
		rows = append(rows, x)
		byCode[x.code] = x
	}
	// Whisper.
	src, err := os.ReadFile(*whisper)
	if err != nil {
		log.Fatal(err)
	}
	block := regexp.MustCompile(`(?s)LANGUAGES = \{(.*?)\n\}`).FindSubmatch(src)
	n := 0
	for _, m := range regexp.MustCompile(`"([a-z]+)": "([^"]+)"`).FindAllSubmatch(block[1], -1) {
		wc, wname := string(m[1]), string(m[2])
		n++
		code := wc
		if a, ok := aliases[wc]; ok && (a["_reason"] == "deprecated" || a["_reason"] == "legacy") {
			code = a["_replacement"] // jw → jv, tl → fil
		}
		x := byCode[code]
		if x == nil {
			name := enNames[code]
			if name == "" {
				name = strings.Title(wname)
			}
			iso3 := names3[code]
			if iso3 == "" {
				iso3 = code
			}
			x = &row{code: code, name: name, iso3: iso3}
			rows = append(rows, x)
			byCode[code] = x
		}
		x.models += "w"
		if wc != code {
			x.whisper = wc
		}
	}
	if n != 100 {
		log.Fatalf("whisper: %d languages, want 100 (99 + Cantonese)", n)
	}
	// CLDR data.
	missing := []string{}
	for _, x := range rows {
		if a, ok := aliases[x.code]; ok && a["_reason"] == "macrolanguage" {
			x.macro = a["_replacement"]
		}
		x.native = localName(x.code, x.code)
		if x.native == "" && x.macro != "" {
			x.native = localName(x.macro, x.macro)
		}
		x.native = capFirst(x.native)
		x.zh = zhNames[x.code]
		ls := likely[x.code]
		if ls == "" && x.macro != "" {
			ls = likely[x.macro]
		}
		if ls == "" {
			ls = fallbackLikely[x.code]
		}
		if ls == "" {
			missing = append(missing, x.code)
		}
		parts := strings.Split(ls, "-")
		if len(parts) == 3 {
			x.script, x.region = parts[1], parts[2]
		}
		x.rtl = rtl[x.script]
	}
	if len(missing) > 0 {
		log.Fatalf("no likely script/region for %v: add them to fallbackLikely", missing)
	}
	slices.SortFunc(rows, func(a, b *row) int { return strings.Compare(a.code, b.code) })

	var b strings.Builder
	b.WriteString("# code\tiso639_3\tname\tnative\tzh\tscript\trtl\tregion\tmacro\tmodels\twhisper\n")
	for _, x := range rows {
		rt := "0"
		if x.rtl {
			rt = "1"
		}
		fmt.Fprintf(&b, "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", x.code, x.iso3, x.name, x.native, x.zh,
			x.script, rt, x.region, x.macro, x.models, x.whisper)
	}
	write(filepath.Join(*out, "languages.tsv"), b.String())

	// Aliases: legacy, bibliographic, overlong and macrolanguage codes that
	// point at a code in the table and are not themselves codes in it.
	var al []string
	for k, v := range aliases {
		rep := v["_replacement"]
		if byCode[k] != nil || byCode[rep] == nil || strings.ContainsAny(k, "-_") || len(k) > 3 {
			continue
		}
		switch v["_reason"] {
		case "deprecated", "legacy", "bibliographic", "overlong", "macrolanguage":
			al = append(al, k+"\t"+rep)
		}
	}
	slices.Sort(al)
	write(filepath.Join(*out, "aliases.tsv"), "# alias\tcode\n"+strings.Join(al, "\n")+"\n")
	log.Printf("%d languages, %d aliases", len(rows), len(al))
}

// fallbackLikely covers codes CLDR's likelySubtags lacks.
var fallbackLikely = map[string]string{
	"aju": "aju-Hebr-MA", "dgo": "dgo-Deva-IN", "gya": "gya-Latn-CF", "mlq": "mlq-Latn-SN", "ssi": "ssi-Deva-IN",
	"xpe": "xpe-Latn-LR", "src": "src-Latn-IT", "pnb": "pnb-Arab-PK",
}

func capFirst(s string) string {
	r, n := utf8.DecodeRuneInString(s)
	if n == 0 || !unicode.IsLower(r) {
		return s
	}
	return string(unicode.ToTitle(r)) + s[n:]
}

func readJSON(path string, v any, keys ...string) {
	data, err := os.ReadFile(path)
	if err != nil {
		log.Fatal(err)
	}
	var raw json.RawMessage = data
	for _, k := range keys {
		var m map[string]json.RawMessage
		if err := json.Unmarshal(raw, &m); err != nil {
			log.Fatalf("%s: %v", path, err)
		}
		raw = m[k]
	}
	if err := json.Unmarshal(raw, v); err != nil {
		log.Fatalf("%s: %v", path, err)
	}
}

func write(path, s string) {
	if err := os.WriteFile(path, []byte(s), 0o644); err != nil {
		log.Fatal(err)
	}
}
