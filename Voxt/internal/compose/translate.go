package compose

import (
	"strings"
	"unicode"

	"github.com/sgl-project/sglang-omni/Voxt/internal/lang"
)

// TargetLanguages are the languages Translate offers, in the menu's
// order (Voxt's list, with Simplified and Traditional Chinese apart).
var TargetLanguages = []string{
	"en", "zh-Hans", "zh-Hant", "ja", "ko", "es", "fr", "de", "it", "pt", "ru",
	"nl", "tr", "pl", "uk", "ar", "he", "hi", "th", "vi", "id", "ms",
}

// LanguageName names a language in a template set's language: the lang
// table's name, with "zh-Hans" and "zh-Hant" as Simplified and
// Traditional Chinese.
func LanguageName(code, set string) string {
	zh := set == Chinese || strings.HasPrefix(strings.ToLower(set), "zh")
	switch strings.ToLower(strings.TrimSpace(code)) {
	case "zh-hans", "zh-cn", "zh-sg":
		if zh {
			return "简体中文"
		}
		return "Simplified Chinese"
	case "zh-hant", "zh-tw", "zh-hk", "zh-mo":
		if zh {
			return "繁体中文"
		}
		return "Traditional Chinese"
	}
	if l, ok := lang.Lookup(code); ok {
		return l.NameIn(set)
	}
	return code
}

// Untranslated reports whether a translation into target looks like it
// was not made, and is worth asking again (Input.Strict):
//
//   - it is the source again, ignoring case, spaces and punctuation;
//   - or the whole source appears unparenthesized beside its translation;
//   - or it is mostly outside the target's script while the source is too.
//     At least a quarter of the answer in that script can preserve names;
//     an echo of the whole source plus a word does not count.
//
// A source already in the target's script (Chinese into Chinese) is never
// judged untranslated. Ambiguous Latin text is still retried, then accepted
// unless LanguageMismatch gives positive evidence for a hard failure.
func Untranslated(source, result, target string) bool {
	tags := strings.NewReplacer("<指令>", "", "</指令>", "")
	source, result = tags.Replace(source), tags.Replace(result)
	src, res := fold(source), fold(result)
	if src == "" || res == "" {
		return false
	}
	is := scriptOf(target)
	if is == nil {
		return src == res
	}
	srcShare, srcLetters := share(source, is)
	resShare, resLetters := share(result, is)
	if srcLetters > 0 && srcShare >= 0.5 && !isLatin(target) {
		return false // the source is in the target's script already
	}
	if src == res {
		return isLatin(target) || ScriptMismatch(source, target)
	}
	glossed := parenthesizedSource(source, result)
	if containsWholeSource(source, result) && !glossed && (isLatin(target) || ScriptMismatch(source, target)) {
		return true // a whole source beside its translation is an echo, not a preserved name
	}
	if resLetters > 0 && resShare >= 0.25 && (!containsWholeSource(source, result) || glossed) {
		return false // enough target text to allow preserved names, not a source echo
	}
	return resLetters > 0 && resShare < 0.5 && srcShare < 0.5
}

// parenthesizedSource reports an original name kept as a gloss beside its
// translation, such as Doubao (豆包) or 谷歌（Google）.
func parenthesizedSource(source, result string) bool {
	return strings.Contains(result, "("+source+")") || strings.Contains(result, "（"+source+"）")
}

// containsWholeSource reports source as a whole letter-token sequence in
// result. Short words inside another word, such as it in petit, are not echoes.
func containsWholeSource(source, result string) bool {
	sourceTokens, resultTokens := letterTokens(source), letterTokens(result)
	if len(sourceTokens) == 0 || len(sourceTokens) > len(resultTokens) {
		return false
	}
	for start := 0; start+len(sourceTokens) <= len(resultTokens); start++ {
		matched := true
		for offset, token := range sourceTokens {
			if !strings.EqualFold(resultTokens[start+offset], token) {
				matched = false
				break
			}
		}
		if matched {
			return true
		}
	}
	return false
}

func letterTokens(text string) []string {
	var tokens []string
	start, latin := -1, false
	for i, r := range text {
		if !unicode.IsLetter(r) {
			if start >= 0 {
				tokens = append(tokens, text[start:i])
				start = -1
			}
			continue
		}
		currentLatin := unicode.Is(unicode.Latin, r)
		if start < 0 {
			start, latin = i, currentLatin
			continue
		}
		if currentLatin != latin {
			tokens = append(tokens, text[start:i])
			start, latin = i, currentLatin
		}
	}
	if start >= 0 {
		tokens = append(tokens, text[start:])
	}
	return tokens
}

// ScriptMismatch reports a letter outside target's script. Latin-only text
// has no such evidence, so it never distinguishes English from French.
func ScriptMismatch(text, target string) bool {
	is := scriptOf(target)
	if is == nil {
		return false
	}
	for _, r := range text {
		if unicode.IsLetter(r) && !is(r) {
			return true
		}
	}
	return false
}

// LanguageMismatch reports positive evidence that text is not in target.
// Script mismatches count. For Latin text, common vocabulary can establish a
// language; shared words score on every matching side, and unknown words and
// language-neutral names are not evidence.
func LanguageMismatch(text, target string) bool {
	if neutralName(text) {
		return false
	}
	if ScriptMismatch(text, target) {
		return true
	}
	if !isLatin(target) {
		return false
	}
	candidates := latinLanguages(text)
	return len(candidates) > 0 && !candidates[lang.Canonical(target)]
}

// neutralName reports a single capitalized Latin token with no vocabulary
// evidence. Product and personal names stay valid across target scripts.
func neutralName(text string) bool {
	tokens := letterTokens(text)
	if len(tokens) != 1 {
		return false
	}
	for _, words := range commonLatinWords {
		for _, common := range words {
			if strings.EqualFold(tokens[0], common) {
				return false
			}
		}
	}
	for _, r := range tokens[0] {
		return unicode.Is(unicode.Latin, r) && unicode.IsUpper(r)
	}
	return false
}

// commonLatinWords identifies a language from function words and common
// greetings. Shared words deliberately produce ties, not a forced mismatch.
var commonLatinWords = map[string][]string{
	"en": {"the", "a", "an", "and", "or", "but", "if", "then", "else", "when", "where", "why", "how", "what", "who", "which", "is", "are", "am", "was", "were", "be", "been", "do", "does", "did", "have", "has", "had", "will", "would", "can", "could", "should", "may", "might", "must", "to", "of", "in", "on", "at", "by", "from", "with", "without", "this", "that", "these", "those", "i", "me", "my", "we", "us", "our", "you", "your", "he", "she", "it", "its", "they", "them", "their", "good", "morning", "goodbye", "hello", "hi", "night", "evening", "day", "week", "month", "year", "time", "work", "life", "world", "friends", "people", "person", "man", "woman", "new", "old", "big", "small", "long", "short", "high", "low", "early", "late", "right", "left", "next", "last", "first", "second", "best", "better", "worse", "worst", "nice", "happy", "sad", "easy", "hard", "free", "full", "sure", "real", "great", "little", "own", "same", "different", "thank", "thanks", "please", "yes", "no", "son", "comment", "con", "non", "est", "english"},
	"fr": {"le", "la", "les", "des", "de", "du", "et", "ou", "est", "sont", "à", "au", "aux", "ce", "cette", "ces", "bonjour", "salut", "bonsoir", "bonne", "journée", "comment", "ça", "va", "merci", "je", "tu", "il", "elle", "une", "un", "nous", "vous", "ils", "elles", "pas", "oui", "non", "on", "a", "son", "français", "jour", "nuit", "semaine", "mois", "année", "temps", "travail", "vie", "monde", "gens", "homme", "femme", "bon", "nouveau", "vieux", "grand", "petit", "long", "court", "haut", "bas", "tôt", "tard", "premier", "dernier", "meilleur", "pire", "beau", "facile", "difficile", "libre", "plein", "sûr", "vrai", "super", "même", "autre"},
	"es": {"el", "la", "los", "las", "de", "del", "y", "o", "es", "son", "en", "con", "por", "para", "una", "un", "hola", "adiós", "gracias", "cómo", "como", "qué", "que", "sí", "no", "buenos", "buenas", "días", "noches", "señor", "señora", "yo", "tú", "usted", "nosotros", "ellos", "ellas", "a", "español", "día", "noche", "semana", "mes", "año", "tiempo", "trabajo", "vida", "mundo", "gente", "hombre", "mujer", "bueno", "nuevo", "viejo", "grande", "pequeño", "largo", "corto", "alto", "bajo", "temprano", "tarde", "primero", "último", "mejor", "peor", "bonito", "fácil", "difícil", "libre", "lleno", "seguro", "real", "genial", "mismo", "otro"},
	"de": {"guten", "tag", "hallo", "danke", "ich", "du", "ist", "nicht", "und", "der", "die", "das", "ein", "eine", "zu", "den", "mit", "für", "auf", "kein", "keine", "ja", "nein", "bitte", "morgen", "abend", "nacht", "heute", "gestern", "woche", "monat", "jahr", "zeit", "arbeit", "leben", "welt", "menschen", "gut", "neu", "alt", "groß", "klein", "lang", "kurz", "hoch", "früh", "spät", "erste", "letzte", "beste", "besser", "schlechter", "schön", "einfach", "schwer", "frei", "voll", "sicher", "echt", "super", "gleich", "anders", "deutsch"},
	"it": {"ciao", "buongiorno", "come", "stai", "il", "lo", "la", "gli", "le", "di", "a", "da", "in", "con", "su", "per", "non", "sono", "sei", "è", "grazie", "buon", "giorno", "sera", "notte", "oggi", "ieri", "domani", "settimana", "mese", "anno", "tempo", "vita", "mondo", "persone", "buono", "nuovo", "vecchio", "grande", "piccolo", "lungo", "corto", "alto", "basso", "presto", "tardi", "primo", "ultimo", "meglio", "peggio", "bello", "facile", "difficile", "libero", "pieno", "sicuro", "vero", "grande", "stesso", "diverso", "italiano"},
	"pt": {"olá", "oi", "obrigado", "obrigada", "como", "você", "está", "bom", "dia", "de", "do", "da", "em", "um", "uma", "não", "é", "para", "com", "boa", "tarde", "noite", "hoje", "ontem", "amanhã", "semana", "mês", "ano", "tempo", "vida", "mundo", "pessoas", "novo", "velho", "grande", "pequeno", "longo", "curto", "alto", "baixo", "cedo", "primeiro", "último", "melhor", "pior", "bonito", "fácil", "difícil", "livre", "cheio", "certo", "real", "ótimo", "mesmo", "diferente", "português"},
	"nl": {"hallo", "goedemorgen", "dank", "je", "jij", "de", "het", "een", "van", "voor", "met", "niet", "is", "zijn", "en", "goed", "morgen", "avond", "nacht", "vandaag", "gisteren", "week", "maand", "jaar", "tijd", "werk", "leven", "wereld", "mensen", "nieuw", "oud", "groot", "klein", "lang", "kort", "hoog", "laag", "vroeg", "laat", "eerste", "laatste", "beste", "beter", "slechter", "mooi", "makkelijk", "moeilijk", "vrij", "vol", "zeker", "echt", "geweldig", "zelfde", "anders", "nederlands"},
	"tr": {"merhaba", "günaydın", "teşekkür", "nasılsın", "bir", "ve", "bu", "için", "de", "mi", "mı", "yok", "var", "ben", "sen", "gün", "gece", "bugün", "dün", "yarın", "hafta", "ay", "yıl", "zaman", "iş", "hayat", "dünya", "insanlar", "iyi", "yeni", "eski", "büyük", "küçük", "uzun", "kısa", "yüksek", "alçak", "erken", "geç", "sağ", "sol", "ilk", "son", "en", "daha", "kötü", "güzel", "kolay", "zor", "özgür", "dolu", "emin", "gerçek", "harika", "aynı", "farklı", "türkçe"},
	"pl": {"cześć", "dzień", "dobry", "dziękuję", "jak", "się", "jest", "nie", "tak", "do", "na", "to", "za", "że", "wieczór", "noc", "dziś", "wczoraj", "jutro", "tydzień", "miesiąc", "rok", "czas", "praca", "życie", "świat", "ludzie", "nowy", "stary", "duży", "mały", "długi", "krótki", "wysoki", "niski", "wcześnie", "późno", "pierwszy", "ostatni", "najlepszy", "lepszy", "gorszy", "ładny", "łatwy", "trudny", "wolny", "pełny", "pewny", "prawdziwy", "świetny", "samy", "inny", "polski"},
	"vi": {"xin", "chào", "cảm", "ơn", "bạn", "khỏe", "không", "có", "của", "cho", "và", "là", "tôi", "buổi", "sáng", "chiều", "tối", "ngày", "hôm", "qua", "mai", "tuần", "tháng", "năm", "thời", "gian", "công", "việc", "cuộc", "sống", "thế", "giới", "người", "tốt", "mới", "cũ", "lớn", "nhỏ", "dài", "ngắn", "cao", "thấp", "sớm", "muộn", "đầu", "cuối", "nhất", "xấu", "đẹp", "dễ", "khó", "tự", "đầy", "chắc", "thật", "tuyệt", "cùng", "khác", "tiếng", "việt"},
	"id": {"halo", "selamat", "pagi", "terima", "kasih", "apa", "kabar", "saya", "kamu", "di", "dan", "yang", "untuk", "dengan", "tidak", "itu", "ini", "siang", "malam", "hari", "kemarin", "besok", "minggu", "bulan", "tahun", "waktu", "kerja", "hidup", "dunia", "orang", "baik", "baru", "lama", "besar", "kecil", "panjang", "pendek", "tinggi", "rendah", "awal", "akhir", "pertama", "terakhir", "terbaik", "lebih", "buruk", "bagus", "mudah", "sulit", "bebas", "penuh", "pasti", "benar", "hebat", "sama", "berbeda", "indonesia"},
	"ms": {"halo", "selamat", "pagi", "terima", "kasih", "apa", "kabar", "saya", "kamu", "di", "dan", "yang", "untuk", "dengan", "tidak", "itu", "ini", "siang", "malam", "hari", "kemarin", "besok", "minggu", "bulan", "tahun", "waktu", "kerja", "hidup", "dunia", "orang", "baik", "baru", "lama", "besar", "kecil", "panjang", "pendek", "tinggi", "rendah", "awal", "akhir", "pertama", "terakhir", "terbaik", "lebih", "buruk", "bagus", "mudah", "sulit", "bebas", "penuh", "pasti", "benar", "hebat", "sama", "berbeda", "melayu"},
}

func latinLanguages(text string) map[string]bool {
	scores := map[string]int{}
	for _, word := range strings.FieldsFunc(strings.ToLower(text), func(r rune) bool { return !unicode.IsLetter(r) }) {
		for code, words := range commonLatinWords {
			for _, common := range words {
				if word == common {
					scores[code]++
					break
				}
			}
		}
	}
	bestScore := 0
	for _, score := range scores {
		if score > bestScore {
			bestScore = score
		}
	}
	candidates := map[string]bool{}
	for code, score := range scores {
		if score == bestScore {
			candidates[code] = true
		}
	}
	return candidates
}

// Written says whether most of text's words are in the script of the
// language code; false for no words or an unknown script.
func Written(text, code string) bool {
	is := scriptOf(code)
	if is == nil {
		return false
	}
	s, n := share(text, is)
	return n > 0 && s >= 0.5
}

// fold is s without case, spaces and punctuation.
func fold(s string) string {
	return strings.Map(func(r rune) rune {
		if unicode.IsSpace(r) || unicode.IsPunct(r) || unicode.IsSymbol(r) {
			return -1
		}
		return unicode.ToLower(r)
	}, s)
}

// scriptOf is a test of the letters a target language is written in; nil
// when unknown.
func scriptOf(target string) func(rune) bool {
	code := strings.ToLower(target)
	script := ""
	switch {
	case strings.HasPrefix(code, "zh"), code == "yue":
		script = "Hani"
	default:
		if l, ok := lang.Lookup(target); ok {
			script = l.Script
		}
	}
	switch script {
	case "Hans", "Hant", "Hani":
		return func(r rune) bool { return unicode.Is(unicode.Han, r) }
	case "Jpan":
		return func(r rune) bool {
			return unicode.In(r, unicode.Han, unicode.Hiragana, unicode.Katakana)
		}
	case "Kore":
		return func(r rune) bool { return unicode.In(r, unicode.Hangul, unicode.Han) }
	case "":
		return nil
	}
	if t, ok := unicode.Scripts[scriptNames[script]]; ok {
		return func(r rune) bool { return unicode.Is(t, r) }
	}
	return nil
}

func isLatin(target string) bool {
	l, ok := lang.Lookup(target)
	return ok && l.Script == "Latn" && !strings.HasPrefix(strings.ToLower(target), "zh")
}

// scriptNames maps ISO 15924 codes to package unicode's script names.
var scriptNames = map[string]string{
	"Latn": "Latin", "Cyrl": "Cyrillic", "Arab": "Arabic", "Hebr": "Hebrew", "Deva": "Devanagari",
	"Thai": "Thai", "Grek": "Greek", "Hang": "Hangul", "Beng": "Bengali", "Taml": "Tamil",
	"Telu": "Telugu", "Gujr": "Gujarati", "Guru": "Gurmukhi", "Knda": "Kannada", "Mlym": "Malayalam",
	"Khmr": "Khmer", "Laoo": "Lao", "Mymr": "Myanmar", "Geor": "Georgian", "Armn": "Armenian",
	"Ethi": "Ethiopic", "Sinh": "Sinhala",
}

// share is the part of s's words that is in the script, and how many
// words s has. A word is a run of letters of an alphabet, or one
// ideograph, kana or Hangul syllable (a Chinese name in Latin text weighs
// what an English one in Chinese text does).
func share(s string, in func(rune) bool) (float64, int) {
	n, k := 0, 0
	inWord := false
	for _, r := range s {
		switch {
		case !unicode.IsLetter(r):
			inWord = false
		case unicode.In(r, unicode.Han, unicode.Hiragana, unicode.Katakana, unicode.Hangul):
			inWord = false
			n++
			if in(r) {
				k++
			}
		case !inWord:
			inWord = true
			n++
			if in(r) {
				k++
			}
		}
	}
	if n == 0 {
		return 0, 0
	}
	return float64(k) / float64(n), n
}
