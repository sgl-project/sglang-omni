package archtest

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strconv"
	"strings"
	"testing"
)

// TestSleepsInTests keeps time.Sleep out of new tests: a test waits on an
// event or a fake clock, so the suite does not grow slower or flaky with
// the machine's load. Diff review keeps the allowances in sleeps.txt on a
// ratchet; comparing the final counts alone cannot show whether a PR raised
// one.
func TestSleepsInTests(t *testing.T) {
	data, err := os.ReadFile("sleeps.txt")
	if err != nil {
		t.Fatal(err)
	}
	allowed, err := readSleepAllowance(data)
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range goFiles(t) {
		if !f.test() {
			continue
		}
		got, want := sleepCallCount(t, f.path, f.data), allowed[f.path]
		delete(allowed, f.path)
		switch {
		case got > want:
			t.Errorf("%s calls time.Sleep %d times, sleeps.txt allows %d: wait on an event or a fake clock", f.path, got, want)
		case got < want:
			t.Errorf("%s calls time.Sleep %d times: lower its count in sleeps.txt from %d (remove the line at 0)", f.path, got, want)
		}
	}
	for path := range allowed {
		t.Errorf("sleeps.txt lists %s, which is not a test file of the module: remove the line", path)
	}
}

func readSleepAllowance(data []byte) (map[string]int, error) {
	allowed := map[string]int{}
	lines := map[string]int{}
	for n, line := range strings.Split(string(data), "\n") {
		text, _, _ := strings.Cut(line, "#")
		fields := strings.Fields(text)
		if len(fields) == 0 {
			continue
		}
		count := 0
		var err error
		if len(fields) == 2 {
			count, err = strconv.Atoi(fields[1])
		}
		if len(fields) != 2 || err != nil || count <= 0 {
			return nil, fmt.Errorf("sleeps.txt:%d: want \"path count\", got %q", n+1, line)
		}
		if previous, exists := lines[fields[0]]; exists {
			return nil, fmt.Errorf("sleeps.txt:%d: %s is on line %d already", n+1, fields[0], previous)
		}
		allowed[fields[0]] = count
		lines[fields[0]] = n + 1
	}
	return allowed, nil
}

func sleepCallCount(t *testing.T, path string, data []byte) int {
	t.Helper()
	file, err := parser.ParseFile(token.NewFileSet(), path, data, parser.SkipObjectResolution)
	if err != nil {
		t.Fatal(err)
	}
	timeNames := map[string]bool{}
	dotTime := false
	for _, spec := range file.Imports {
		importPath, err := strconv.Unquote(spec.Path.Value)
		if err != nil {
			t.Fatal(err)
		}
		if importPath != "time" {
			continue
		}
		name := "time"
		if spec.Name != nil {
			name = spec.Name.Name
		}
		switch name {
		case ".":
			dotTime = true
		case "_":
		default:
			timeNames[name] = true
		}
	}
	sleepVariables := map[string]bool{}
	var isSleep func(ast.Expr) bool
	isSleep = func(expression ast.Expr) bool {
		switch expression := expression.(type) {
		case *ast.Ident:
			return sleepVariables[expression.Name] || dotTime && expression.Name == "Sleep"
		case *ast.SelectorExpr:
			packageName, ok := expression.X.(*ast.Ident)
			return ok && timeNames[packageName.Name] && expression.Sel.Name == "Sleep"
		case *ast.ParenExpr:
			return isSleep(expression.X)
		default:
			return false
		}
	}
	count := 0
	ast.Inspect(file, func(node ast.Node) bool {
		switch node := node.(type) {
		case *ast.AssignStmt:
			for i, value := range node.Rhs {
				if i < len(node.Lhs) && isSleep(value) {
					if name, ok := node.Lhs[i].(*ast.Ident); ok {
						sleepVariables[name.Name] = true
					}
				}
			}
		case *ast.ValueSpec:
			for i, value := range node.Values {
				if i < len(node.Names) && isSleep(value) {
					sleepVariables[node.Names[i].Name] = true
				}
			}
		case *ast.CallExpr:
			if isSleep(node.Fun) {
				count++
			}
		}
		return true
	})
	return count
}

func TestSleepCallCount(t *testing.T) {
	tests := []struct {
		name   string
		source string
		want   int
	}{
		{"direct", "package p\nimport \"time\"\nfunc f() { time.Sleep(0) }", 1},
		{"import alias", "package p\nimport clock \"time\"\nfunc f() { clock.Sleep(0) }", 1},
		{"function value", "package p\nimport \"time\"\nfunc f() { sleep := time.Sleep; sleep(0) }", 1},
		{"parenthesized direct", "package p\nimport \"time\"\nfunc f() { (time.Sleep)(0) }", 1},
		{"parenthesized function value", "package p\nimport \"time\"\nfunc f() { sleep := (time.Sleep); sleep(0) }", 1},
		{"dot import", "package p\nimport . \"time\"\nfunc f() { Sleep(0) }", 1},
		{"comment", "package p\nimport \"time\"\n// time.Sleep(0)\nfunc f() {}", 0},
		{"string", "package p\nvar source = \"time.Sleep(0)\"", 0},
		{"unrelated package", "package p\nimport clock \"example/clock\"\nfunc f() { clock.Sleep(0) }", 0},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := sleepCallCount(t, "fixture_test.go", []byte(test.source)); got != test.want {
				t.Fatalf("got %d calls, want %d", got, test.want)
			}
		})
	}
}

func TestSleepAllowanceRejectsDuplicate(t *testing.T) {
	_, err := readSleepAllowance([]byte("foo_test.go 1\nfoo_test.go 2\n"))
	want := "sleeps.txt:2: foo_test.go is on line 1 already"
	if err == nil || err.Error() != want {
		t.Fatalf("got error %v, want %q", err, want)
	}
}
