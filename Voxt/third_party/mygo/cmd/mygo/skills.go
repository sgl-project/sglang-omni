package main

import (
	"fmt"
	"os"
	"path/filepath"
)

func runInstallSkills(args []string) error {
	flags := newFlags("install-skills", "[dir]", "Installs or updates the bundled MyGo agent skills in .agents/skills.\nReplaces bundled skill files, preserving unrelated files and skills.")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() > 1 {
		flags.Usage()
		return flag2Err("expected at most one project directory")
	}
	dir := dirArg(flags.Args())
	info, err := os.Stat(dir)
	if err != nil {
		return err
	}
	if !info.IsDir() {
		return fmt.Errorf("%s is not a directory", dir)
	}
	if err := installSkills(dir); err != nil {
		return err
	}
	logf("Installed skills in %s", filepath.Join(dir, ".agents", "skills"))
	return nil
}

// installSkills refreshes only files bundled under the skills directory,
// without removing other skills or extra files in a bundled skill.
func installSkills(dir string) error {
	return writeTemplateDir(filepath.Join(dir, ".agents", "skills"), "shared/.agents/skills", templateData{})
}
