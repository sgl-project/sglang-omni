package compose

import (
	"regexp"
	"strconv"
	"strings"
)

var spokenParagraph = regexp.MustCompile(`(?i)(?:^|([。！？.!?；;])|[，,])\s*(?:另起一段|换段|new paragraph)[\t ]*(?:[。！？.!?；;，,]|$)`)

var spokenOrdinal = regexp.MustCompile(`(?i)(^|[\n。！？.!?；;，,：:])[\t ]*(第[一二三四五六七八九]|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|[1-9])(?:[点项]?(?:是[、，,:：]?|[、，,:：.)）]))[\t ]*`)

var spokenOrdinalNumber = map[string]int{
	"第一": 1, "第二": 2, "第三": 3, "第四": 4, "第五": 5, "第六": 6, "第七": 7, "第八": 8, "第九": 9,
	"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9,
}

// StructureSpoken turns explicit paragraph commands and consecutive items
// into formatting. Ordinary speech and ambiguous lists stay as recognised.
func StructureSpoken(text string) string {
	formatted := spokenParagraph.ReplaceAllString(text, "${1}\n\n")
	if formatted == text {
		return structureItems(text)
	}
	paragraphs := strings.Split(formatted, "\n\n")
	result := paragraphs[:0]
	for _, paragraph := range paragraphs {
		if len(paragraphs) > 1 {
			paragraph = strings.TrimSpace(paragraph)
		}
		if paragraph != "" {
			result = append(result, structureItems(paragraph))
		}
	}
	if len(result) == 0 {
		return text
	}
	return strings.Join(result, "\n\n")
}

func structureItems(text string) string {
	markers := spokenOrdinal.FindAllStringSubmatchIndex(text, -1)
	if len(markers) < 2 {
		return text
	}
	lines := make([]string, 0, len(markers))
	for i, marker := range markers {
		ordinal := strings.ToLower(text[marker[4]:marker[5]])
		number, err := strconv.Atoi(ordinal)
		if err != nil {
			number = spokenOrdinalNumber[ordinal]
		} else if text[marker[5]:marker[1]] == "." {
			// A period followed by a digit belongs to a decimal.
			if next := text[marker[1]:]; next != "" && next[0] >= '0' && next[0] <= '9' {
				return text
			}
		}
		if number != i+1 {
			return text
		}
		end := len(text)
		if i+1 < len(markers) {
			end = markers[i+1][3]
		}
		item := strings.TrimSpace(text[marker[1]:end])
		if i+1 < len(markers) {
			item = strings.TrimRight(item, "，,；;\r\n")
		}
		if item == "" {
			return text
		}
		lines = append(lines, strconv.Itoa(number)+". "+item)
	}
	list := strings.Join(lines, "\n")
	if lead := strings.TrimSpace(text[:markers[0][3]]); lead != "" {
		return lead + "\n\n" + list
	}
	return list
}
