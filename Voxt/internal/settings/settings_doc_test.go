package settings

import (
	"encoding/json"
	"os"
	"reflect"
	"regexp"
	"strings"
	"testing"
	"time"
)

// settingsDocPath is the settings reference (#4050); this test keeps it in
// step with Settings, so a new field fails here until its row is added.
const settingsDocPath = "../../docs/settings.md"

// settingsField is one JSON path of the settings document: a leaf (a value
// the reference gives a row) or a struct the reference may also name.
type settingsField struct {
	path   string
	goType reflect.Type
	isLeaf bool
}

// docRow is one row of a "Path" or "Field" table of the reference.
type docRow struct {
	paths        []string
	typeCell     string
	defaultCell  string
	hasDefault   bool
	isRelative   bool
	headingTitle string
}

var backtickedPattern = regexp.MustCompile("`([^`]*)`")

// relativeFeatureField matches the fields the reference lists once for
// Transcribe, Translate, Rewrite and Meetings.
var relativeFeatureField = regexp.MustCompile(`^features\.(transcribe|translate|rewrite|meeting)\.(asr|llm|prompt|showCard)$`)

// docKey is the name a field's row uses: a provider's fields and the
// features' shared fields are listed once, by their own name.
func docKey(path string) string {
	if strings.HasPrefix(path, "providers.llm[].") || strings.HasPrefix(path, "providers.asr[].") {
		return path[strings.LastIndex(path, ".")+1:]
	} else if relativeFeatureField.MatchString(path) {
		return path[strings.LastIndex(path, ".")+1:]
	}
	return path
}

// isValueType reports whether a settings type has one row of its own
// rather than a row per field: a shortcut, a browser entry, a time.
func isValueType(goType reflect.Type) bool {
	return goType == reflect.TypeOf(Hotkey{}) || goType == reflect.TypeOf(Browser{}) || goType == reflect.TypeOf(time.Time{})
}

// isSettingsStruct reports whether the walk goes into a type's fields.
func isSettingsStruct(goType reflect.Type) bool {
	return goType.Kind() == reflect.Struct && goType.PkgPath() == reflect.TypeOf(Settings{}).PkgPath() && !isValueType(goType)
}

// collectSettingsFields walks the JSON tags of the settings structs, into
// pointers and into a list of providers ("providers.llm[].endpoint").
func collectSettingsFields(goType reflect.Type, prefix string, fields *[]settingsField) {
	valueType := goType
	for valueType.Kind() == reflect.Pointer {
		valueType = valueType.Elem()
	}
	if valueType.Kind() == reflect.Slice && isSettingsStruct(valueType.Elem()) {
		*fields = append(*fields, settingsField{path: prefix, goType: goType})
		collectSettingsFields(valueType.Elem(), prefix+"[]", fields)
		return
	} else if !isSettingsStruct(valueType) {
		*fields = append(*fields, settingsField{path: prefix, goType: goType, isLeaf: true})
		return
	} else if prefix != "" {
		*fields = append(*fields, settingsField{path: prefix, goType: goType})
	}
	for index := 0; index < valueType.NumField(); index++ {
		structField := valueType.Field(index)
		name := strings.Split(structField.Tag.Get("json"), ",")[0]
		if structField.Anonymous {
			collectSettingsFields(structField.Type, prefix, fields)
			continue
		} else if name == "" || name == "-" {
			continue
		}
		path := name
		if prefix != "" {
			path = prefix + "." + name
		}
		collectSettingsFields(structField.Type, path, fields)
	}
}

// readDocRows reads the rows of the reference's tables whose first column
// is "Path" (paths in the document) or "Field" (relative to a feature or a
// provider).
func readDocRows(t *testing.T) []docRow {
	content, err := os.ReadFile(settingsDocPath)
	if err != nil {
		t.Fatalf("read %s: %v", settingsDocPath, err)
	}
	var rows []docRow
	var header []string
	heading := ""
	for _, line := range strings.Split(string(content), "\n") {
		if strings.HasPrefix(line, "#") {
			heading = line
			continue
		} else if !strings.HasPrefix(line, "|") {
			header = nil
			continue
		}
		cells := strings.Split(strings.Trim(line, "|"), "|")
		for index := range cells {
			cells[index] = strings.TrimSpace(cells[index])
		}
		if header == nil {
			header = cells
			continue
		} else if strings.HasPrefix(cells[0], "---") || (header[0] != "Path" && header[0] != "Field") {
			continue
		}
		row := docRow{isRelative: header[0] == "Field", headingTitle: heading}
		for _, match := range backtickedPattern.FindAllStringSubmatch(cells[0], -1) {
			row.paths = append(row.paths, match[1])
		}
		for index, title := range header {
			if index >= len(cells) {
				break
			} else if title == "Type" {
				row.typeCell = strings.ReplaceAll(cells[index], "`", "")
			} else if title == "Default" {
				row.defaultCell, row.hasDefault = cells[index], true
			}
		}
		rows = append(rows, row)
	}
	return rows
}

// docTypeWord is the word the reference's Type column starts with for a
// field of this Go type.
func docTypeWord(goType reflect.Type) string {
	for goType.Kind() == reflect.Pointer {
		goType = goType.Elem()
	}
	switch {
	case goType == reflect.TypeOf(time.Time{}), goType.Kind() == reflect.String:
		return "string"
	case goType == reflect.TypeOf(Hotkey{}):
		return "shortcut"
	case goType.Kind() == reflect.Bool:
		return "bool"
	case goType.Kind() == reflect.Int, goType.Kind() == reflect.Int64:
		return "integer"
	case goType.Kind() == reflect.Float64:
		return "number"
	case goType.Kind() == reflect.Slice:
		return "array"
	case goType.Kind() == reflect.Map, goType.Kind() == reflect.Struct:
		return "object"
	default:
		return goType.Kind().String()
	}
}

func TestSettingsDocListsEveryField(t *testing.T) {
	var fields []settingsField
	collectSettingsFields(reflect.TypeOf(Settings{}), "", &fields)
	fieldByKey := map[string]settingsField{}
	for _, field := range fields {
		fieldByKey[docKey(field.path)] = field
	}
	rowByKey := map[string]docRow{}
	for _, row := range readDocRows(t) {
		for _, path := range row.paths {
			if _, ok := fieldByKey[path]; !ok {
				t.Errorf("%s: row %q is not a field of Settings", row.headingTitle, path)
			} else {
				rowByKey[path] = row
			}
		}
	}
	for _, field := range fields {
		row, ok := rowByKey[docKey(field.path)]
		if !ok && field.isLeaf {
			t.Errorf("%s has no row in %s", field.path, settingsDocPath)
			continue
		} else if !ok {
			continue
		}
		word := docTypeWord(field.goType)
		if !strings.HasPrefix(row.typeCell, word) {
			t.Errorf("%s: the reference says %q, the field is %s", field.path, row.typeCell, word)
		} else if field.goType.Kind() == reflect.Pointer && field.isLeaf && !strings.Contains(row.typeCell, "or absent") {
			t.Errorf("%s: the reference says %q, but the field may be absent", field.path, row.typeCell)
		}
	}
}

func TestSettingsDocDefaultsMatchDefaults(t *testing.T) {
	encoded, err := json.Marshal(Defaults())
	if err != nil {
		t.Fatal(err)
	}
	var defaultsDocument map[string]any
	if err := json.Unmarshal(encoded, &defaultsDocument); err != nil {
		t.Fatal(err)
	}
	var fields []settingsField
	collectSettingsFields(reflect.TypeOf(Settings{}), "", &fields)
	isPointerByPath := map[string]bool{}
	for _, field := range fields {
		isPointerByPath[field.path] = field.goType.Kind() == reflect.Pointer
	}
	for _, row := range readDocRows(t) {
		if row.isRelative || !row.hasDefault {
			continue
		}
		for _, path := range row.paths {
			value, found := lookupDefault(defaultsDocument, path)
			if strings.HasPrefix(row.defaultCell, "absent") {
				if found && value != nil {
					t.Errorf("%s: the reference says absent, Defaults() has %v", path, value)
				}
				continue
			}
			match := backtickedPattern.FindStringSubmatch(row.defaultCell)
			if match == nil {
				t.Errorf("%s: no default value in %q", path, row.defaultCell)
				continue
			}
			var expected any
			if json.Unmarshal([]byte(match[1]), &expected) != nil {
				expected = match[1]
			}
			if !found && isPointerByPath[path] {
				t.Errorf("%s: the reference says %s, Defaults() leaves it absent", path, match[1])
			} else if !found && !isZeroValue(expected) {
				t.Errorf("%s: the reference says %s, Defaults() leaves it at its zero value", path, match[1])
			} else if found && !reflect.DeepEqual(expected, value) {
				t.Errorf("%s: the reference says %s, Defaults() has %v", path, match[1], value)
			}
		}
	}
}

// lookupDefault reads a dotted path from the encoded defaults; an omitted
// field is not found.
func lookupDefault(document map[string]any, path string) (any, bool) {
	var current any = document
	for _, name := range strings.Split(path, ".") {
		object, ok := current.(map[string]any)
		if !ok {
			return nil, false
		}
		current, ok = object[name]
		if !ok {
			return nil, false
		}
	}
	return current, true
}

// isZeroValue reports whether a decoded JSON value is what an omitted
// field decodes to.
func isZeroValue(value any) bool {
	switch typed := value.(type) {
	case string:
		return typed == ""
	case bool:
		return !typed
	case float64:
		return typed == 0
	case []any:
		return len(typed) == 0
	case map[string]any:
		return len(typed) == 0
	default:
		return value == nil
	}
}
