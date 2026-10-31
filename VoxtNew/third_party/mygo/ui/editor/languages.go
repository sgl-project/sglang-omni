package editor

import (
	"path"
	"regexp"
	"strings"
	"sync"
)

// LanguageFor returns the language of a file by its name, as
// "main.go" or "Makefile", or nil when MyGo knows none for it.
func LanguageFor(filename string) *Language {
	base := path.Base(strings.ReplaceAll(filename, "\\", "/"))
	ext := strings.ToLower(path.Ext(base))
	for _, l := range Languages() {
		for _, f := range l.Filenames {
			if f == base {
				return l
			}
		}
	}
	for _, l := range Languages() {
		for _, e := range l.Extensions {
			if e == ext {
				return l
			}
		}
	}
	return nil
}

// LanguageNamed returns the language MyGo knows by a name, as "Go",
// "TypeScript" or "tsx", ignoring case: its Name, an extension without
// the dot, or a usual alias, as Markdown's code blocks name them.
func LanguageNamed(name string) *Language {
	name = strings.ToLower(strings.TrimSpace(name))
	if name == "" {
		return nil
	}
	name = languageAlias(name)
	for _, l := range Languages() {
		if strings.ToLower(l.Name) == name {
			return l
		}
		for _, e := range l.Extensions {
			if e[1:] == name {
				return l
			}
		}
	}
	return nil
}

// languageAlias returns the name of the language name is another name
// for.
func languageAlias(name string) string {
	switch name {
	case "golang":
		return "go"
	case "js", "node", "mjs", "cjs":
		return "javascript"
	case "ts":
		return "typescript"
	case "py", "python3":
		return "python"
	case "sh", "bash", "zsh", "console", "shellscript", "fish":
		return "shell"
	case "yml":
		return "yaml"
	case "md":
		return "markdown"
	case "rs":
		return "rust"
	case "cpp", "cc", "cxx", "objc", "objective-c", "objectivec":
		return "c++"
	case "cs", "csharp":
		return "c#"
	case "kt":
		return "kotlin"
	case "rb":
		return "ruby"
	case "patch", "udiff":
		return "diff"
	case "docker":
		return "dockerfile"
	case "make", "mk":
		return "makefile"
	case "svg", "plist":
		return "xml"
	case "jsonc", "json5":
		return "json"
	case "postgres", "postgresql", "mysql", "sqlite", "plsql":
		return "sql"
	case "ini", "cfg", "conf":
		return "toml"
	case "proto":
		return "protobuf"
	case "scss", "less":
		return "css"
	}
	return name
}

// Languages returns the languages MyGo knows how to highlight.
func Languages() []*Language {
	languagesOnce.Do(makeLanguages)
	return languageList
}

var (
	languagesOnce sync.Once
	languageList  []*Language
)

func cAutoClose(quotes ...string) []BracketPair {
	out := []BracketPair{{Open: "(", Close: ")"}, {Open: "[", Close: "]"}, {Open: "{", Close: "}"}}
	for _, q := range quotes {
		out = append(out, BracketPair{Open: q, Close: q, NotIn: []TokenKind{TokenString, TokenComment}})
	}
	return out
}

func cString(q string) lexString {
	return lexString{open: q, close: q, escape: '\\', kind: TokenString}
}

const cOperators = "+-*/%=&|^!<>~?:@#"

func makeLanguages() {
	cBrackets := []BracketPair{{Open: "(", Close: ")"}, {Open: "[", Close: "]"}, {Open: "{", Close: "}"}}
	notInText := []TokenKind{TokenString, TokenComment}
	goLang := &Language{
		Name: "Go", Extensions: []string{".go"}, LineComment: "//", BlockComment: [2]string{"/*", "*/"},
		Brackets: cBrackets, AutoClose: cAutoClose("\"", "'", "`"), TabSize: 4, HardTabs: true,
		Tokenizer: &lexer{
			lineComments: []string{"//"}, blockComments: [][2]string{{"/*", "*/"}},
			strings: []lexString{cString("\""), cString("'"), {open: "`", close: "`", multiline: true, kind: TokenString}},
			words: words(TokenKeyword, "break case chan const continue default defer else fallthrough for func go goto if import interface map package range return select struct switch type var",
				TokenType, "any bool byte comparable complex64 complex128 error float32 float64 int int8 int16 int32 int64 rune string uint uint8 uint16 uint32 uint64 uintptr",
				TokenConstant, "true false nil iota",
				TokenFunction, "append cap clear close complex copy delete imag len make max min new panic print println real recover"),
			calls: true, operators: cOperators,
		},
	}
	jsWords := []any{TokenKeyword, "async await break case catch class const continue debugger default delete do else export extends finally for from function get if import in instanceof let new of return set static super switch this throw try typeof var void while with yield as",
		TokenConstant, "true false null undefined NaN Infinity",
		TokenType, "Array Boolean Date Error Function JSON Map Math Number Object Promise Proxy Reflect RegExp Set String Symbol WeakMap WeakSet BigInt console window document globalThis"}
	jsStrings := []lexString{cString("\""), cString("'"), {open: "`", close: "`", escape: '\\', multiline: true, interpOpen: "${", interpClose: "}", kind: TokenString}}
	javascript := &Language{
		Name: "JavaScript", Extensions: []string{".js", ".mjs", ".cjs", ".jsx"}, LineComment: "//", BlockComment: [2]string{"/*", "*/"},
		Brackets: cBrackets, AutoClose: cAutoClose("\"", "'", "`"), WordChars: "$", TabSize: 2,
		Tokenizer: &lexer{lineComments: []string{"//"}, blockComments: [][2]string{{"/*", "*/"}}, strings: jsStrings,
			words: words(jsWords...), identChars: "$", calls: true, capitalTypes: true, regexps: true, decorator: '@', operators: cOperators},
	}
	tsWords := append(append([]any{}, jsWords...), TokenKeyword, "abstract declare enum implements interface keyof namespace private protected public readonly type infer is satisfies override module",
		TokenType, "any boolean never number object string symbol unknown void bigint")
	typescript := &Language{
		Name: "TypeScript", Extensions: []string{".ts", ".mts", ".cts", ".tsx"}, LineComment: "//", BlockComment: [2]string{"/*", "*/"},
		Brackets: cBrackets, AutoClose: cAutoClose("\"", "'", "`"), WordChars: "$", TabSize: 2,
		Tokenizer: &lexer{lineComments: []string{"//"}, blockComments: [][2]string{{"/*", "*/"}}, strings: jsStrings,
			words: words(tsWords...), identChars: "$", calls: true, capitalTypes: true, regexps: true, decorator: '@', operators: cOperators},
	}
	json := &Language{
		Name: "JSON", Extensions: []string{".json", ".jsonc", ".json5", ".webmanifest"}, Filenames: []string{".babelrc", ".eslintrc", "tsconfig.json"},
		LineComment: "//", BlockComment: [2]string{"/*", "*/"}, Brackets: cBrackets, AutoClose: cAutoClose("\""), TabSize: 2,
		Tokenizer: &jsonLexer{lexer{lineComments: []string{"//"}, blockComments: [][2]string{{"/*", "*/"}}, strings: []lexString{cString("\""), cString("'")},
			words: words(TokenConstant, "true false null"), operators: ":"}},
	}
	pyStrings := []lexString{}
	for _, p := range []string{"", "r", "R", "b", "B", "u", "U", "f", "F", "rb", "br", "Rb", "bR", "fr", "rf", "Fr", "rF"} {
		raw := strings.ContainsAny(p, "rR")
		f := strings.ContainsAny(p, "fF")
		for _, q := range []string{`"""`, `'''`, `"`, `'`} {
			s := lexString{open: q, close: q, prefix: p, multiline: len(q) == 3, kind: TokenString}
			if !raw {
				s.escape = '\\'
			}
			if f {
				s.interpOpen, s.interpClose = "{", "}"
			}
			pyStrings = append(pyStrings, s)
		}
	}
	python := &Language{
		Name: "Python", Extensions: []string{".py", ".pyi", ".pyw"}, Filenames: []string{"SConstruct", "SConscript"}, LineComment: "#",
		Brackets: cBrackets, AutoClose: cAutoClose("\"", "'"), TabSize: 4,
		IncreaseIndent: regexp.MustCompile(`^\s*(class|def|elif|else|except|finally|for|if|try|while|with|async|match|case)\b.*:\s*(#.*)?$|[\[({]\s*$`),
		Tokenizer: &lexer{lineComments: []string{"#"}, strings: pyStrings,
			words: words(TokenKeyword, "and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield match case",
				TokenConstant, "True False None NotImplemented Ellipsis __name__ __file__",
				TokenType, "int float str bytes bool list dict set tuple object type complex frozenset bytearray range Exception",
				TokenFunction, "print len open isinstance super self cls"),
			calls: true, capitalTypes: true, decorator: '@', operators: cOperators},
	}
	rust := &Language{
		Name: "Rust", Extensions: []string{".rs"}, LineComment: "//", BlockComment: [2]string{"/*", "*/"},
		Brackets: cBrackets, AutoClose: cAutoClose("\""), TabSize: 4,
		Tokenizer: &lexer{lineComments: []string{"//"}, blockComments: [][2]string{{"/*", "*/"}}, nestComments: true,
			strings: []lexString{{open: "\"", close: "\"", escape: '\\', multiline: true, kind: TokenString},
				{open: "\"", close: "\"", prefix: "b", escape: '\\', multiline: true, kind: TokenString},
				{open: "#\"", close: "\"#", prefix: "r", multiline: true, kind: TokenString},
				{open: "\"", close: "\"", prefix: "r", multiline: true, kind: TokenString},
				{open: "'", close: "'", escape: '\\', kind: TokenString}},
			words: words(TokenKeyword, "as async await break const continue crate dyn else enum extern fn for if impl in let loop match mod move mut pub ref return self Self static struct super trait type unsafe use where while yield",
				TokenType, "bool char f32 f64 i8 i16 i32 i64 i128 isize str u8 u16 u32 u64 u128 usize String Vec Option Result Box",
				TokenConstant, "true false None Some Ok Err"),
			calls: true, capitalTypes: true, directives: false, decorator: '#', operators: cOperators},
	}
	cWords := []any{TokenKeyword, "auto break case const continue default do else enum extern for goto if inline register restrict return sizeof static struct switch typedef union volatile while _Alignas _Alignof _Atomic _Bool _Generic _Noreturn _Static_assert _Thread_local",
		TokenType, "char double float int long short signed unsigned void size_t ssize_t int8_t int16_t int32_t int64_t uint8_t uint16_t uint32_t uint64_t bool FILE",
		TokenConstant, "NULL true false"}
	c := &Language{
		Name: "C", Extensions: []string{".c", ".h"}, LineComment: "//", BlockComment: [2]string{"/*", "*/"},
		Brackets: cBrackets, AutoClose: cAutoClose("\"", "'"), TabSize: 4,
		Tokenizer: &lexer{lineComments: []string{"//"}, blockComments: [][2]string{{"/*", "*/"}}, strings: []lexString{cString("\""), cString("'")},
			words: words(cWords...), calls: true, directives: true, operators: cOperators},
	}
	cpp := &Language{
		Name: "C++", Extensions: []string{".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx", ".ino", ".mm", ".m"}, LineComment: "//", BlockComment: [2]string{"/*", "*/"},
		Brackets: cBrackets, AutoClose: cAutoClose("\"", "'"), TabSize: 4,
		Tokenizer: &lexer{lineComments: []string{"//"}, blockComments: [][2]string{{"/*", "*/"}}, strings: []lexString{cString("\""), cString("'"), {open: "R\"(", close: ")\"", multiline: true, kind: TokenString}},
			words: words(append(append([]any{}, cWords...), TokenKeyword, "alignas alignof and asm catch class concept consteval constexpr constinit co_await co_return co_yield decltype delete dynamic_cast explicit export friend mutable namespace new noexcept not operator or private protected public reinterpret_cast requires static_assert static_cast template this thread_local throw try typeid typename using virtual xor override final @interface @implementation @end @property self",
				TokenType, "wchar_t char8_t char16_t char32_t std string vector map", TokenConstant, "nullptr nil YES NO")...),
			calls: true, capitalTypes: true, directives: true, operators: cOperators},
	}
	javaWords := []any{TokenKeyword, "abstract assert break case catch class const continue default do else enum extends final finally for goto if implements import instanceof interface native new package private protected public return static strictfp super switch synchronized this throw throws transient try volatile while var record sealed permits yield",
		TokenType, "boolean byte char double float int long short void String Object Integer",
		TokenConstant, "true false null"}
	java := &Language{
		Name: "Java", Extensions: []string{".java"}, LineComment: "//", BlockComment: [2]string{"/*", "*/"},
		Brackets: cBrackets, AutoClose: cAutoClose("\"", "'"), TabSize: 4,
		Tokenizer: &lexer{lineComments: []string{"//"}, blockComments: [][2]string{{"/*", "*/"}}, strings: []lexString{{open: `"""`, close: `"""`, escape: '\\', multiline: true, kind: TokenString}, cString("\""), cString("'")},
			words: words(javaWords...), calls: true, capitalTypes: true, decorator: '@', operators: cOperators},
	}
	kotlin := &Language{
		Name: "Kotlin", Extensions: []string{".kt", ".kts"}, LineComment: "//", BlockComment: [2]string{"/*", "*/"},
		Brackets: cBrackets, AutoClose: cAutoClose("\"", "'"), TabSize: 4,
		Tokenizer: &lexer{lineComments: []string{"//"}, blockComments: [][2]string{{"/*", "*/"}}, nestComments: true,
			strings: []lexString{{open: `"""`, close: `"""`, multiline: true, interpOpen: "${", interpClose: "}", dollar: true, kind: TokenString},
				{open: "\"", close: "\"", escape: '\\', interpOpen: "${", interpClose: "}", dollar: true, kind: TokenString}, cString("'")},
			words: words(TokenKeyword, "as break class continue do else for fun if in interface is object package return super this throw try typealias typeof val var when while by catch constructor delegate dynamic field file finally get import init param property receiver set setparam where actual abstract annotation companion const crossinline data enum expect external final infix inline inner internal lateinit noinline open operator out override private protected public reified sealed suspend tailrec vararg",
				TokenConstant, "true false null", TokenType, "Any Boolean Byte Char Double Float Int Long Nothing Short String Unit"),
			calls: true, capitalTypes: true, decorator: '@', operators: cOperators},
	}
	swift := &Language{
		Name: "Swift", Extensions: []string{".swift"}, LineComment: "//", BlockComment: [2]string{"/*", "*/"},
		Brackets: cBrackets, AutoClose: cAutoClose("\""), TabSize: 4,
		Tokenizer: &lexer{lineComments: []string{"//"}, blockComments: [][2]string{{"/*", "*/"}}, nestComments: true,
			strings: []lexString{{open: `"""`, close: `"""`, escape: '\\', multiline: true, kind: TokenString}, cString("\"")},
			words: words(TokenKeyword, "associatedtype class deinit enum extension fileprivate func import init inout internal let open operator private precedencegroup protocol public rethrows static struct subscript typealias var break case catch continue default defer do else fallthrough for guard if in repeat return throw switch where while as is try await async throws some any self Self super actor nonisolated isolated mutating nonmutating override weak unowned lazy final required convenience",
				TokenConstant, "true false nil", TokenType, "Int Double Float String Bool Character Array Dictionary Set Optional Void"),
			calls: true, capitalTypes: true, decorator: '@', directives: true, operators: cOperators},
	}
	csharp := &Language{
		Name: "C#", Extensions: []string{".cs", ".csx"}, LineComment: "//", BlockComment: [2]string{"/*", "*/"},
		Brackets: cBrackets, AutoClose: cAutoClose("\"", "'"), TabSize: 4,
		Tokenizer: &lexer{lineComments: []string{"//"}, blockComments: [][2]string{{"/*", "*/"}},
			strings: []lexString{{open: "@\"", close: "\"", multiline: true, kind: TokenString}, {open: "$\"", close: "\"", escape: '\\', interpOpen: "{", interpClose: "}", kind: TokenString}, cString("\""), cString("'")},
			words: words(TokenKeyword, "abstract as base break case catch checked class const continue default delegate do else enum event explicit extern finally fixed for foreach goto if implicit in interface internal is lock namespace new operator out override params private protected public readonly ref return sealed sizeof stackalloc static struct switch this throw try typeof unchecked unsafe using virtual volatile while async await var get set init record when where yield",
				TokenType, "bool byte char decimal double float int long object sbyte short string uint ulong ushort void dynamic",
				TokenConstant, "true false null"),
			calls: true, capitalTypes: true, directives: true, operators: cOperators},
	}
	shell := &Language{
		Name: "Shell", Extensions: []string{".sh", ".bash", ".zsh", ".fish", ".ksh", ".command"}, Filenames: []string{".bashrc", ".zshrc", ".profile", ".bash_profile", "PKGBUILD", ".envrc"},
		LineComment: "#", Brackets: cBrackets, AutoClose: cAutoClose("\"", "'", "`"), TabSize: 2,
		IncreaseIndent: regexp.MustCompile(`(\bthen|\bdo|\{|\(|\belse)\s*$`),
		DecreaseIndent: regexp.MustCompile(`^\s*(fi|done|esac|else|elif|\}|\))\b`),
		Tokenizer: &lexer{lineComments: []string{"#"},
			strings: []lexString{{open: "\"", close: "\"", escape: '\\', multiline: true, dollar: true, kind: TokenString}, {open: "'", close: "'", multiline: true, kind: TokenString}, {open: "`", close: "`", escape: '\\', kind: TokenString}},
			words: words(TokenKeyword, "if then else elif fi case esac for select while until do done in function time coproc return exit break continue local export declare readonly unset shift source alias eval exec set trap",
				TokenFunction, "echo printf cd pwd test read mkdir rm cp mv ls cat grep sed awk find xargs sudo chmod chown curl git go bun npm"),
			identChars: "-", variables: true, operators: "|&;<>=!*?"},
	}
	yaml := &Language{
		Name: "YAML", Extensions: []string{".yaml", ".yml"}, Filenames: []string{".clang-format"}, LineComment: "#",
		Brackets: cBrackets, AutoClose: cAutoClose("\"", "'"), TabSize: 2,
		IncreaseIndent: regexp.MustCompile(`:\s*(#.*)?$|^\s*-\s*$`),
		Tokenizer: &lexer{lineComments: []string{"#"}, strings: []lexString{cString("\""), {open: "'", close: "'", kind: TokenString}},
			words: words(TokenConstant, "true false null yes no on off True False Null ~"), keys: true, identChars: "-./", operators: "|>-*&!?:"},
	}
	toml := &Language{
		Name: "TOML", Extensions: []string{".toml", ".ini", ".cfg", ".conf", ".properties", ".editorconfig"}, Filenames: []string{"Cargo.lock", ".gitconfig", ".npmrc"},
		LineComment: "#", Brackets: cBrackets, AutoClose: cAutoClose("\"", "'"), TabSize: 2,
		Tokenizer: &tomlLexer{lexer{lineComments: []string{"#", ";"},
			strings: []lexString{{open: `"""`, close: `"""`, escape: '\\', multiline: true, kind: TokenString}, {open: "'''", close: "'''", multiline: true, kind: TokenString}, cString("\""), {open: "'", close: "'", kind: TokenString}},
			words:   words(TokenConstant, "true false"), identChars: "-.", operators: "="}},
	}
	sql := &Language{
		Name: "SQL", Extensions: []string{".sql"}, LineComment: "--", BlockComment: [2]string{"/*", "*/"},
		Brackets: cBrackets, AutoClose: cAutoClose("\"", "'"), TabSize: 2,
		Tokenizer: &lexer{lineComments: []string{"--"}, blockComments: [][2]string{{"/*", "*/"}}, foldCase: true,
			strings: []lexString{{open: "'", close: "'", multiline: true, kind: TokenString}, {open: "\"", close: "\"", kind: TokenVariable}, {open: "`", close: "`", kind: TokenVariable}},
			words: words(TokenKeyword, "select from where and or not insert into values update set delete create table drop alter add column index view as on join left right inner outer full cross group by order having limit offset union all distinct case when then else end is null in exists between like primary key foreign references default unique check constraint begin commit rollback transaction if returning with recursive asc desc trigger function procedure replace temporary temp database schema grant revoke cascade conflict do nothing using natural explain analyze pragma vacuum",
				TokenType, "int integer bigint smallint tinyint real float double decimal numeric text varchar char boolean bool date time timestamp blob json jsonb uuid serial",
				TokenConstant, "true false null"),
			calls: true, operators: "+-*/%=<>!|"},
	}
	ruby := &Language{
		Name: "Ruby", Extensions: []string{".rb", ".rake", ".gemspec"}, Filenames: []string{"Gemfile", "Rakefile", "Podfile", "Fastfile"}, LineComment: "#",
		Brackets: cBrackets, AutoClose: cAutoClose("\"", "'"), TabSize: 2,
		IncreaseIndent: regexp.MustCompile(`^\s*(class|module|def|if|unless|else|elsif|while|until|for|begin|rescue|ensure|case|when)\b|\bdo(\s*\|.*\|)?\s*$|[\[({]\s*$`),
		DecreaseIndent: regexp.MustCompile(`^\s*(end|else|elsif|rescue|ensure|when)\b`),
		Tokenizer: &lexer{lineComments: []string{"#"},
			strings: []lexString{{open: "\"", close: "\"", escape: '\\', multiline: true, interpOpen: "#{", interpClose: "}", kind: TokenString}, {open: "'", close: "'", escape: '\\', multiline: true, kind: TokenString}},
			words: words(TokenKeyword, "alias and begin break case class def defined? do else elsif end ensure for if in module next not or redo rescue retry return self super then undef unless until when while yield require require_relative attr_accessor attr_reader attr_writer private protected public",
				TokenConstant, "true false nil"),
			identChars: "?!", calls: true, capitalTypes: true, variables: false, operators: cOperators},
	}
	lua := &Language{
		Name: "Lua", Extensions: []string{".lua"}, LineComment: "--", BlockComment: [2]string{"--[[", "]]"},
		Brackets: cBrackets, AutoClose: cAutoClose("\"", "'"), TabSize: 2,
		IncreaseIndent: regexp.MustCompile(`\b(then|do|else|function\b.*\))\s*$|^\s*(local\s+)?function\b|[\[({]\s*$`),
		DecreaseIndent: regexp.MustCompile(`^\s*(end|else|elseif|until|\}|\))`),
		Tokenizer: &lexer{lineComments: []string{"--"}, blockComments: [][2]string{{"--[[", "]]"}},
			strings: []lexString{{open: "[[", close: "]]", multiline: true, kind: TokenString}, cString("\""), cString("'")},
			words:   words(TokenKeyword, "and break do else elseif end for function goto if in local not or repeat return then until while", TokenConstant, "true false nil"),
			calls:   true, operators: "+-*/%^#=<>~.:"},
	}
	php := &Language{
		Name: "PHP", Extensions: []string{".php"}, LineComment: "//", BlockComment: [2]string{"/*", "*/"},
		Brackets: cBrackets, AutoClose: cAutoClose("\"", "'"), TabSize: 4,
		Tokenizer: &lexer{lineComments: []string{"//", "#"}, blockComments: [][2]string{{"/*", "*/"}},
			strings: []lexString{{open: "\"", close: "\"", escape: '\\', multiline: true, dollar: true, kind: TokenString}, {open: "'", close: "'", escape: '\\', multiline: true, kind: TokenString}},
			words: words(TokenKeyword, "abstract and array as break callable case catch class clone const continue declare default do echo else elseif empty enddeclare endfor endforeach endif endswitch endwhile extends final finally fn for foreach function global goto if implements include include_once instanceof insteadof interface isset list match namespace new or print private protected public readonly require require_once return static switch throw trait try unset use var while xor yield",
				TokenConstant, "true false null TRUE FALSE NULL"),
			calls: true, capitalTypes: true, variables: true, operators: cOperators},
	}
	css := &Language{
		Name: "CSS", Extensions: []string{".css", ".scss", ".less", ".sass", ".pcss"}, BlockComment: [2]string{"/*", "*/"},
		Brackets: cBrackets, AutoClose: cAutoClose("\"", "'"), WordChars: "-", TabSize: 2,
		Tokenizer: &cssLexer{},
	}
	html := &Language{
		Name: "HTML", Extensions: []string{".html", ".htm", ".xhtml", ".vue", ".svelte"}, BlockComment: [2]string{"<!--", "-->"},
		Brackets:  []BracketPair{{Open: "<", Close: ">"}, {Open: "(", Close: ")"}, {Open: "[", Close: "]"}, {Open: "{", Close: "}"}},
		AutoClose: cAutoClose("\"", "'"), WordChars: "-", TabSize: 2,
		IncreaseIndent: regexp.MustCompile(`<(\w[\w-]*)(\s[^>]*)?>[^<]*$|[\[({]\s*$`),
		DecreaseIndent: regexp.MustCompile(`^\s*</`),
	}
	xml := &Language{
		Name: "XML", Extensions: []string{".xml", ".svg", ".plist", ".xsd", ".xsl", ".rss", ".atom", ".csproj", ".storyboard", ".xib"}, BlockComment: [2]string{"<!--", "-->"},
		Brackets: []BracketPair{{Open: "<", Close: ">"}}, AutoClose: cAutoClose("\"", "'"), WordChars: "-:", TabSize: 2,
		IncreaseIndent: regexp.MustCompile(`<(\w[\w:-]*)(\s[^>]*)?>[^<]*$`),
		DecreaseIndent: regexp.MustCompile(`^\s*</`),
		Tokenizer:      &markupLexer{},
	}
	markdown := &Language{
		Name: "Markdown", Extensions: []string{".md", ".markdown", ".mdx", ".mkd"}, Filenames: []string{"README", "CHANGELOG"},
		BlockComment: [2]string{"<!--", "-->"}, Brackets: cBrackets,
		AutoClose: []BracketPair{{Open: "(", Close: ")"}, {Open: "[", Close: "]"}, {Open: "{", Close: "}"}, {Open: "`", Close: "`", NotIn: notInText}},
		TabSize:   2,
		Tokenizer: &markdownLexer{},
	}
	diffLang := &Language{
		Name: "Diff", Extensions: []string{".diff", ".patch", ".rej"}, TabSize: 4,
		Tokenizer: TokenizerFunc(tokenizeDiff),
	}
	dockerfile := &Language{
		Name: "Dockerfile", Extensions: []string{".dockerfile"}, Filenames: []string{"Dockerfile", "Containerfile"}, LineComment: "#",
		Brackets: cBrackets, AutoClose: cAutoClose("\"", "'"), TabSize: 4,
		Tokenizer: &lexer{lineComments: []string{"#"}, foldCase: false,
			strings:   []lexString{{open: "\"", close: "\"", escape: '\\', dollar: true, kind: TokenString}, {open: "'", close: "'", kind: TokenString}},
			words:     words(TokenKeyword, "FROM AS RUN CMD LABEL MAINTAINER EXPOSE ENV ADD COPY ENTRYPOINT VOLUME USER WORKDIR ARG ONBUILD STOPSIGNAL HEALTHCHECK SHELL"),
			variables: true, identChars: "-", operators: "=&|;<>"},
	}
	makefile := &Language{
		Name: "Makefile", Extensions: []string{".mk", ".mak"}, Filenames: []string{"Makefile", "makefile", "GNUmakefile"}, LineComment: "#",
		Brackets: cBrackets, AutoClose: cAutoClose("\"", "'"), TabSize: 4, HardTabs: true,
		Tokenizer: &lexer{lineComments: []string{"#"},
			strings:   []lexString{{open: "\"", close: "\"", escape: '\\', dollar: true, kind: TokenString}, {open: "'", close: "'", kind: TokenString}},
			words:     words(TokenKeyword, "ifeq ifneq ifdef ifndef else endif include define endef export override .PHONY"),
			variables: true, keys: true, identChars: "-./", operators: "=:?+!@"},
	}
	protobuf := &Language{
		Name: "Protobuf", Extensions: []string{".proto"}, LineComment: "//", BlockComment: [2]string{"/*", "*/"},
		Brackets: cBrackets, AutoClose: cAutoClose("\""), TabSize: 2,
		Tokenizer: &lexer{lineComments: []string{"//"}, blockComments: [][2]string{{"/*", "*/"}}, strings: []lexString{cString("\""), cString("'")},
			words: words(TokenKeyword, "syntax package import option message enum service rpc returns repeated optional required oneof map reserved extend extensions to max stream public weak",
				TokenType, "double float int32 int64 uint32 uint64 sint32 sint64 fixed32 fixed64 sfixed32 sfixed64 bool string bytes",
				TokenConstant, "true false"),
			capitalTypes: true, operators: "=<>"},
	}
	zig := &Language{
		Name: "Zig", Extensions: []string{".zig", ".zon"}, LineComment: "//", Brackets: cBrackets, AutoClose: cAutoClose("\"", "'"), TabSize: 4,
		Tokenizer: &lexer{lineComments: []string{"//"}, strings: []lexString{cString("\""), cString("'"), {open: "\\\\", close: "\x00", kind: TokenString}},
			words: words(TokenKeyword, "addrspace align allowzero and anyframe anytype asm async await break callconv catch comptime const continue defer else enum errdefer error export extern fn for if inline linksection noalias noinline nosuspend opaque or orelse packed pub resume return struct suspend switch test threadlocal try union unreachable usingnamespace var volatile while",
				TokenType, "i8 u8 i16 u16 i32 u32 i64 u64 i128 u128 isize usize f16 f32 f64 f80 f128 bool void noreturn type anyerror comptime_int comptime_float",
				TokenConstant, "true false null undefined"),
			calls: true, capitalTypes: true, decorator: '@', operators: cOperators},
	}
	// HTML's tokenizer highlights the scripts and styles inside it.
	html.Tokenizer = &markupLexer{html: true, script: javascript.Tokenizer, style: css.Tokenizer}
	languageList = []*Language{goLang, javascript, typescript, json, python, rust, c, cpp, java, kotlin, swift, csharp,
		shell, yaml, toml, sql, ruby, lua, php, css, html, xml, markdown, diffLang, dockerfile, makefile, protobuf, zig}
}
