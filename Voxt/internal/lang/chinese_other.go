//go:build !darwin

package lang

func convertChineseScript(string, string) (string, bool) {
	return "", false
}
