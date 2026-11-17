package compose

import "testing"

func TestStructureSpoken(t *testing.T) {
	for _, example := range []struct{ text, want string }{
		{"请原样转录我说的话。", "请原样转录我说的话。"},
		{"Normal prose.\n\n  Keep this paragraph as typed.", "Normal prose.\n\n  Keep this paragraph as typed."},
		{"第一次发布，第二天再检查。", "第一次发布，第二天再检查。"},
		{"1.5 liters. 2.5 liters.", "1.5 liters. 2.5 liters."},
		{"第一，准备材料。第二，发给客户。", "1. 准备材料。\n2. 发给客户。"},
		{"第一项是核对时间，第二项是发邮件。", "1. 核对时间\n2. 发邮件。"},
		{"Steps: First, read the notes. Second, send the link.", "Steps:\n\n1. read the notes.\n2. send the link."},
		{"This morning. New paragraph. This afternoon.", "This morning.\n\nThis afternoon."},
		{"今天开会。另起一段。明天发邮件。", "今天开会。\n\n明天发邮件。"},
		{"First, check the address. Third, send the email.", "First, check the address. Third, send the email."},
	} {
		if got := StructureSpoken(example.text); got != example.want {
			t.Errorf("%q = %q, want %q", example.text, got, example.want)
		}
	}
}
