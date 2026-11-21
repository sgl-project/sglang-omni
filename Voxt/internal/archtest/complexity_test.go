package archtest

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"maps"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"
)

// complexLimit is the most cognitive complexity a function of code may
// have. Measured on 2026-11-21 over 4424 functions: the median is 2, the
// 95th percentile 19, the 98th 32; 129 functions are over 25.
// proseTokenMention (internal/compose/answer_control.go), the example of a
// function hard to read, scores 28.
const complexLimit = 25

// TestComplexFuncs keeps functions of code within complexLimit. The score
// is gocognit's cognitive complexity (golangci-lint's gocognit linter):
// each if, else, switch, select, for and labelled jump adds one, plus one
// per level it is nested in; each run of && or || adds one. Functions over
// the limit today are in complexity.txt with their score, which must match:
// a listed function may not grow more complex, and one that gets simpler
// lowers its line.
func TestComplexFuncs(t *testing.T) {
	data, err := os.ReadFile("complexity.txt")
	if err != nil {
		t.Fatal(err)
	}
	allowed, err := readComplexityAllowance(data)
	if err != nil {
		t.Fatal(err)
	}
	var fails []string
	for _, f := range goFiles(t) {
		if f.test() || f.generated() {
			continue
		}
		for name, got := range funcComplexities(t, f.path, f.data) {
			key := f.path + " " + name
			want, listed := allowed[key]
			delete(allowed, key)
			switch {
			case !listed && got > complexLimit:
				fails = append(fails, fmt.Sprintf("%s %s has cognitive complexity %d, over the limit of %d: split it into steps "+
					"that each have a name (a helper per case, a guard per early return).", f.path, name, got, complexLimit))
			case listed && got <= complexLimit:
				fails = append(fails, fmt.Sprintf("%s %s has cognitive complexity %d, within the limit of %d: remove its line from internal/archtest/complexity.txt.",
					f.path, name, got, complexLimit))
			case listed && got > want:
				fails = append(fails, fmt.Sprintf("%s %s has cognitive complexity %d, over its ceiling of %d in complexity.txt: "+
					"functions over the limit of %d may not grow more complex. Move the new case into a function of its own.",
					f.path, name, got, want, complexLimit))
			case listed && got < want:
				fails = append(fails, fmt.Sprintf("%s %s has cognitive complexity %d: lower its ceiling in internal/archtest/complexity.txt from %d to %d.",
					f.path, name, got, want, got))
			}
		}
	}
	for key := range allowed {
		fails = append(fails, fmt.Sprintf("complexity.txt lists %s, which is not a function of the module: remove the line.", key))
	}
	slices.Sort(fails)
	for _, f := range fails {
		t.Error(f)
	}
}

// readComplexityAllowance reads complexity.txt: "path func score" per line,
// keyed by "path func".
func readComplexityAllowance(data []byte) (map[string]int, error) {
	allowed := map[string]int{}
	lines := map[string]int{}
	for n, line := range strings.Split(string(data), "\n") {
		text, _, _ := strings.Cut(line, "#")
		fields := strings.Fields(text)
		if len(fields) == 0 {
			continue
		}
		score := 0
		var err error
		if len(fields) == 3 {
			score, err = strconv.Atoi(fields[2])
		}
		if len(fields) != 3 || err != nil || score <= complexLimit {
			return nil, fmt.Errorf("complexity.txt:%d: want \"path func score\" with a score over %d, got %q", n+1, complexLimit, line)
		}
		key := fields[0] + " " + fields[1]
		if previous, exists := lines[key]; exists {
			return nil, fmt.Errorf("complexity.txt:%d: %s is on line %d already", n+1, key, previous)
		}
		allowed[key] = score
		lines[key] = n + 1
	}
	return allowed, nil
}

// funcComplexities scores each function declared in a file, named as
// "Type.Method" or "func". Function literals count toward the function
// they are in. Of two declarations with one name (init), the higher counts.
func funcComplexities(t *testing.T, path string, data []byte) map[string]int {
	t.Helper()
	file, err := parser.ParseFile(token.NewFileSet(), path, data, parser.SkipObjectResolution)
	if err != nil {
		t.Fatal(err)
	}
	scores := map[string]int{}
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Body == nil {
			continue
		}
		name := fn.Name.Name
		if fn.Recv != nil && len(fn.Recv.List) > 0 {
			name = receiverType(fn.Recv.List[0].Type) + "." + name
		}
		c := &cognitive{name: fn.Name.Name, counted: map[*ast.BinaryExpr]bool{}}
		ast.Walk(c, fn.Body)
		scores[name] = max(scores[name], c.score)
	}
	return scores
}

func receiverType(expr ast.Expr) string {
	switch expr := expr.(type) {
	case *ast.StarExpr:
		return receiverType(expr.X)
	case *ast.IndexExpr:
		return receiverType(expr.X)
	case *ast.IndexListExpr:
		return receiverType(expr.X)
	case *ast.Ident:
		return expr.Name
	}
	return "?"
}

// cognitive scores one function body as gocognit does, so that
// golangci-lint's gocognit gives the same numbers: an else if adds one
// whatever its nesting, an else body is not nested one level deeper, and
// any plain call by the function's own name counts as recursion, even from
// a method or to a builtin (len inside a method len).
type cognitive struct {
	name           string
	nesting, score int
	counted        map[*ast.BinaryExpr]bool // inner operators of a run already scored
}

func (c *cognitive) walk(n ast.Node) {
	if n != nil {
		ast.Walk(c, n)
	}
}

func (c *cognitive) nested(n ast.Node) {
	c.nesting++
	c.walk(n)
	c.nesting--
}

// branch scores an if, switch, select or loop: its header at this level,
// then its body one level deeper.
func (c *cognitive) branch(body *ast.BlockStmt, header ...ast.Node) {
	c.score += 1 + c.nesting
	for _, h := range header {
		c.walk(h)
	}
	c.nested(body)
}

func (c *cognitive) ifStmt(n *ast.IfStmt) {
	c.walk(n.Init)
	c.walk(n.Cond)
	c.nested(n.Body)
	switch e := n.Else.(type) {
	case *ast.IfStmt:
		c.score++
		c.ifStmt(e)
	case *ast.BlockStmt:
		c.score++
		c.walk(e)
	}
}

func (c *cognitive) Visit(n ast.Node) ast.Visitor {
	switch n := n.(type) {
	case *ast.IfStmt:
		c.score += 1 + c.nesting
		c.ifStmt(n)
		return nil
	case *ast.SwitchStmt:
		c.branch(n.Body, n.Init, n.Tag)
		return nil
	case *ast.TypeSwitchStmt:
		c.branch(n.Body, n.Init, n.Assign)
		return nil
	case *ast.SelectStmt:
		c.branch(n.Body)
		return nil
	case *ast.ForStmt:
		c.branch(n.Body, n.Init, n.Cond, n.Post)
		return nil
	case *ast.RangeStmt:
		c.branch(n.Body, n.X)
		return nil
	case *ast.FuncLit:
		c.nested(n.Body)
		return nil
	case *ast.BranchStmt:
		if n.Label != nil {
			c.score++
		}
	case *ast.BinaryExpr:
		if !c.counted[n] {
			var last token.Token
			for _, op := range c.logicalOps(n) {
				if op != last {
					c.score++
					last = op
				}
			}
		}
	case *ast.CallExpr:
		if id, ok := n.Fun.(*ast.Ident); ok && id.Name == c.name {
			c.score++
		}
	}
	return c
}

// logicalOps lists the && and || of a chain of binary expressions in order;
// parentheses start a chain of their own.
func (c *cognitive) logicalOps(e ast.Expr) []token.Token {
	b, ok := e.(*ast.BinaryExpr)
	if !ok {
		return nil
	}
	c.counted[b] = true
	ops := c.logicalOps(b.X)
	if b.Op == token.LAND || b.Op == token.LOR {
		ops = append(ops, b.Op)
	}
	return append(ops, c.logicalOps(b.Y)...)
}

func TestCognitiveComplexity(t *testing.T) {
	tests := []struct {
		name   string
		source string
		want   int
	}{
		{"straight line", "func f() { g(); h() }", 0},
		{"if, else if, else", "func f(x int) { if x > 0 { g() } else if x < 0 { h() } else { g() } }", 3},
		{"nested if", "func f(x int) { for { if x > 0 { if x > 1 { g() } } } }", 6},
		{"else body is not nested", "func f(x int) { if x > 0 { g() } else { if x < 0 { h() } } }", 3},
		{"else if body nests", "func f(x int) { for { if x > 0 { g() } else if x < 0 { if x < -1 { h() } } } }", 7},
		{"one run of &&", "func f(a, b, c bool) bool { return a && b && c }", 1},
		{"&& then ||", "func f(a, b, c bool) bool { return a && b || c }", 2},
		{"parentheses start a run", "func f(a, b, c bool) bool { return a && (b && c) }", 2},
		{"switch with cases", "func f(x int) { switch x { case 1: g(); case 2: h() } }", 1},
		{"nested type switch", "func f(x any) { for { switch x.(type) { case int: g() } } }", 3},
		{"nested select", "func f(c chan int) { for { select { case <-c: g() } } }", 3},
		{"closure nests", "func f(x int) { g := func() { if x > 0 { h() } }; g() }", 2},
		{"labelled continue", "func f(xs []int) { outer: for range xs { for range xs { continue outer } } }", 4},
		{"recursion", "func f(n int) int { if n == 0 { return 0 }; return f(n - 1) }", 2},
		{"call by the name of the method", "func (s *t) f() { f() }", 1},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			scores := funcComplexities(t, "fixture.go", []byte("package p\n"+test.source))
			name := "f"
			if strings.HasPrefix(test.source, "func (s *t)") {
				name = "t.f"
			}
			if got := scores[name]; got != test.want {
				t.Fatalf("got %d, want %d (%v)", got, test.want, scores)
			}
		})
	}
}

func TestComplexityKeys(t *testing.T) {
	source := "package p\nfunc f() {}\nfunc (s *t) m() {}\nfunc (l list[T]) m() {}\nfunc (m *pairs[K, V]) m() {}\n"
	got := slices.Sorted(maps.Keys(funcComplexities(t, "fixture.go", []byte(source))))
	if want := []string{"f", "list.m", "pairs.m", "t.m"}; !slices.Equal(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestComplexityAllowanceRejects(t *testing.T) {
	for _, test := range []struct{ data, want string }{
		{"a.go f 30\na.go f 31\n", "complexity.txt:2: a.go f is on line 1 already"},
		{"a.go f 20\n", "complexity.txt:1: want \"path func score\" with a score over 25, got \"a.go f 20\""},
		{"a.go 30\n", "complexity.txt:1: want \"path func score\" with a score over 25, got \"a.go 30\""},
	} {
		_, err := readComplexityAllowance([]byte(test.data))
		if err == nil || err.Error() != test.want {
			t.Errorf("%q: got error %v, want %q", test.data, err, test.want)
		}
	}
}
