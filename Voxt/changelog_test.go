package main

import (
	"os"
	"strings"
	"testing"
)

func TestChangelogChineseCoversRewritePipeAndQueuedEnter(t *testing.T) {
	content, err := os.ReadFile("CHANGELOG.md")
	if err != nil {
		t.Fatal(err)
	}
	want := "改写保留代码末尾的竖线，旧输入框排队的 Enter 不会发送新录音"
	if !strings.Contains(string(content), want) {
		t.Errorf("Chinese changelog is missing %q", want)
	}
}
