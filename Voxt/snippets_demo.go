package main

func (host *meetingDemoHost) saveSnippet(id, trigger, text string) string {
	host.record("saveSnippet %q %q %q", id, trigger, text)
	return ""
}

func (host *meetingDemoHost) deleteSnippet(id string) { host.record("deleteSnippet %s", id) }

func (host *meetingDemoHost) importSnippets() { host.record("importSnippets") }

func (host *meetingDemoHost) loadSnippets() { host.record("loadSnippets") }
