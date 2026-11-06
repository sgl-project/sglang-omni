package shaderir

// FragmentLibrary returns the program as a library for a renderer that
// draws with a vertex stage of its own, as MyGo's do: a program without
// entry points, whose fragment entry point is the function entry, taking
// the fragment's position (vec4) and the varyings, and the functions entry
// reaches, entry last. Backends emit it as a function of the renderer's
// fragment shader (Fragment in packages glsl, hlsl and msl).
func (p *Program) FragmentLibrary() (lib *Program, entry *Func, funcs []*Func) {
	if p.FragmentFunc.Block == nil {
		return p, nil, nil
	}
	reached := p.ReachableFuncsFromBlock(p.FragmentFunc.Block)

	l := *p
	l.VertexFunc, l.FragmentFunc = VertexFunc{}, FragmentFunc{}
	index := 0
	for _, f := range p.Funcs {
		index = max(index, f.Index+1)
	}
	in := append([]Type{{Main: Vec4}}, p.Varyings...)
	l.Funcs = append(append([]Func(nil), p.Funcs...), Func{
		Index:    index,
		InParams: in,
		Return:   Type{Main: Vec4},
		Block:    p.FragmentFunc.Block,
	})
	entry = &l.Funcs[len(l.Funcs)-1]
	for _, f := range reached {
		for i := range l.Funcs {
			if l.Funcs[i].Index == f.Index {
				funcs = append(funcs, &l.Funcs[i])
			}
		}
	}
	return &l, entry, append(funcs, entry)
}
