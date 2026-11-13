package sounds

import (
	"slices"
	"strings"
)

// Filter is what a picker shows: a group ("" = all), tags the sounds must
// all carry, a text to search for, and uses (any of them; for "suggested").
type Filter struct {
	Group string
	Tags  []string
	Query string
	Uses  []string
}

// Match reports whether s passes f. The query matches, word by word, the
// sound's id, names, description, tags and group (case-insensitive).
func (c *Catalog) Match(s Sound, f Filter) bool {
	if f.Group != "" && s.Group != f.Group {
		return false
	}
	for _, t := range f.Tags {
		if !s.HasTag(t) {
			return false
		}
	}
	if len(f.Uses) > 0 && !slices.ContainsFunc(f.Uses, func(u string) bool { return slices.Contains(s.Uses, u) }) {
		return false
	}
	q := strings.Fields(strings.ToLower(f.Query))
	if len(q) == 0 {
		return true
	}
	hay := []string{s.ID, s.Name.En, s.Name.Zh, s.Desc.En, s.Desc.Zh, s.Group}
	if g, ok := c.Group(s.Group); ok {
		hay = append(hay, g.Name.En, g.Name.Zh)
	}
	for _, t := range s.Tags {
		hay = append(hay, t, c.TagName(t, "en"), c.TagName(t, "zh"))
	}
	text := strings.ToLower(strings.Join(hay, " "))
	for _, w := range q {
		if !strings.Contains(text, w) {
			return false
		}
	}
	return true
}

// Filter lists the sounds that pass f, in catalogue order.
func (c *Catalog) Filter(f Filter) []Sound {
	if c == nil {
		return nil
	}
	var out []Sound
	for _, s := range c.Sounds {
		if c.Match(s, f) {
			out = append(out, s)
		}
	}
	return out
}

// Suggested lists the sounds that suit ev (its Uses), its default first.
func (c *Catalog) Suggested(ev Event) []Sound {
	if c == nil {
		return nil
	}
	var out []Sound
	if s, ok := c.Sound(ev.Default); ok {
		out = append(out, s)
	}
	if len(ev.Uses) == 0 {
		return out
	}
	for _, s := range c.Filter(Filter{Uses: ev.Uses}) {
		if s.ID != ev.Default {
			out = append(out, s)
		}
	}
	return out
}

// TagCount is a tag and how many sounds carry it.
type TagCount struct {
	Tag
	Count int
}

// TagCounts lists the tags of the sounds that pass f without its own tags
// (the chips a picker can still offer), featured tags first and then by
// count, each with how many of those sounds carry it. featuredOnly keeps
// the catalogue's featured tags alone.
func (c *Catalog) TagCounts(f Filter, featuredOnly bool) []TagCount {
	if c == nil {
		return nil
	}
	f.Tags = nil
	counts := map[string]int{}
	for _, s := range c.Filter(f) {
		for _, t := range s.Tags {
			counts[t]++
		}
	}
	var out []TagCount
	for _, t := range c.Tags {
		if n := counts[t.ID]; n > 0 && (!featuredOnly || t.Featured) {
			out = append(out, TagCount{t, n})
			delete(counts, t.ID)
		}
	}
	if !featuredOnly {
		for id, n := range counts {
			out = append(out, TagCount{Tag{ID: id, Name: Text{En: id}}, n})
		}
	}
	slices.SortStableFunc(out, func(a, b TagCount) int {
		if a.Featured != b.Featured {
			if a.Featured {
				return -1
			}
			return 1
		}
		if a.Count != b.Count {
			return b.Count - a.Count
		}
		return strings.Compare(a.ID, b.ID)
	})
	return out
}

// GroupCounts lists the groups with how many sounds of each pass f
// (ignoring its group), in catalogue order; groups with none are left out.
func (c *Catalog) GroupCounts(f Filter) []GroupCount {
	if c == nil {
		return nil
	}
	f.Group = ""
	counts := map[string]int{}
	for _, s := range c.Filter(f) {
		counts[s.Group]++
	}
	var out []GroupCount
	for _, g := range c.Groups {
		if n := counts[g.ID]; n > 0 {
			out = append(out, GroupCount{g, n})
		}
	}
	return out
}

// GroupCount is a group and how many sounds pass a filter in it.
type GroupCount struct {
	Group
	Count int
}

// ToggleTag adds tag to f.Tags, or takes it out when present.
func (f *Filter) ToggleTag(tag string) {
	if i := slices.Index(f.Tags, tag); i >= 0 {
		f.Tags = slices.Delete(f.Tags, i, i+1)
		return
	}
	f.Tags = append(f.Tags, tag)
}
