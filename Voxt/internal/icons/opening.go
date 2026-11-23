package icons

// The opening speed's ends in Settings → Appearance: Lucide's rabbit and
// turtle (ISC, see NOTICE), unchanged.

func lucide(name, body string) Icon {
	s := parse(`<g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">` + body + `</g>`)
	return Icon{Name: "lucide-" + name, Linear: s, Bold: s}
}

var (
	// OpeningFast is the rabbit: the opening in a second.
	OpeningFast = lucide("rabbit", `<path d="M13 16a3 3 0 0 1 2.24 5"/><path d="M18 12h.01"/><path d="M18 21h-8a4 4 0 0 1-4-4 7 7 0 0 1 7-7h.2L9.6 6.4a1 1 0 1 1 2.8-2.8L15.8 7h.2c3.3 0 6 2.7 6 6v1a2 2 0 0 1-2 2h-1a3 3 0 0 0-3 3"/><path d="M20 8.54V4a2 2 0 1 0-4 0v3"/><path d="M7.612 12.524a3 3 0 1 0-1.6 4.3"/>`)
	// OpeningSlow is the turtle: the opening in three seconds.
	OpeningSlow = lucide("turtle", `<path d="m12 10 2 4v3a1 1 0 0 0 1 1h2a1 1 0 0 0 1-1v-3a8 8 0 1 0-16 0v3a1 1 0 0 0 1 1h2a1 1 0 0 0 1-1v-3l2-4h4Z"/><path d="M4.82 7.9 8 10"/><path d="M15.18 7.9 12 10"/><path d="M16.93 10H20a2 2 0 0 1 0 4H2"/>`)
)
