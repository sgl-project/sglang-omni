package family

import "strings"

// Text is a reference wording from the spec, in English (the default) and
// Simplified Chinese. actnowkit does no i18n: apps copy these into their
// own catalogues. Placeholders are written {app}, {model}, {size}…; Fill
// replaces them.
type Text struct{ EN, ZH string }

// Fill replaces {name} placeholders in both languages.
func (t Text) Fill(kv ...string) Text {
	r := strings.NewReplacer(braced(kv)...)
	return Text{EN: r.Replace(t.EN), ZH: r.Replace(t.ZH)}
}

func braced(kv []string) []string {
	out := make([]string, 0, len(kv))
	for i := 0; i+1 < len(kv); i += 2 {
		out = append(out, "{"+kv[i]+"}", kv[i+1])
	}
	return out
}

// Reference texts of the family directory.
var (
	// TextNewerFormat: a file has a newer major schema (spec §5.1).
	TextNewerFormat = Text{
		EN: "Another ActNow app updated the shared format. Update this app to change these settings.",
		ZH: "另一个 ActNow app 更新了共享格式，请更新本 app 后再修改这些设置。",
	}
	// TextAppliesToAll: under a shared setting that follows the family (spec §9.3).
	TextAppliesToAll = Text{EN: "Applies to all ActNow apps", ZH: "应用于所有 ActNow app"}
	// TextUseOwnSetting: the link that stops following the family (spec §9.3).
	TextUseOwnSetting = Text{EN: "Use a different setting for this app", ZH: "仅为此 app 单独设置"}
)
