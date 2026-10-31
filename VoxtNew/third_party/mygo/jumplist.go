package mygo

import (
	"errors"
	"slices"
	"sync"

	"github.com/egoist/mygo/internal/platform"
)

// JumpList controls the app's jump list on Windows, the menu of its
// taskbar button: the documents the user opened recently (see
// App.AddRecentDocument) and tasks. Use App.JumpList. Its methods do
// nothing on other platforms.
//
// Windows replaces the whole list each time the app changes a part of it,
// so the JumpList keeps every part and commits them together.
type JumpList struct {
	mu    sync.Mutex
	tasks []JumpListTask
}

// JumpListTask is an entry of the Tasks category of the app's jump list on
// Windows. It runs a program, the app itself by default, whether the app
// is running or not.
type JumpListTask struct {
	// Title is the entry's label. It is required.
	Title string
	// Description is the entry's tooltip.
	Description string
	// Path is the program to run; empty is the app's own executable.
	Path string
	// Args are the program's arguments, each passed as it is: MyGo quotes
	// them for the command line.
	Args []string
	// WorkingDir is the directory the program runs in; empty is the
	// system's choice.
	WorkingDir string
	// IconPath and IconIndex choose the icon: the file holding it and the
	// zero-based index of the icon in that file. An empty IconPath uses the
	// program's own icon.
	IconPath  string
	IconIndex int
}

// SetTasks sets the Tasks category of the jump list, like Electron's
// app.setUserTasks; no tasks removes them. Windows keeps the list after the
// app quits, so apps usually set it once at startup. A task that runs the
// app is best a URL of the app's own scheme, which reaches App.OnOpenURL
// whether it starts the app or, with App.RequestSingleInstanceLock, the
// running instance gets it:
//
//	mygo.App.RegisterURLScheme("myapp") // or urlSchemes in mygo.config.ts
//	mygo.App.OnOpenURL(func(u string) {
//		if u == "myapp://new-note" { … }
//	})
//	if err := mygo.App.JumpList.SetTasks([]mygo.JumpListTask{
//		{Title: "New note", Args: []string{"myapp://new-note"}},
//	}); err != nil { … }
func (j *JumpList) SetTasks(tasks []JumpListTask) error {
	needsApp("App.JumpList.SetTasks")
	for _, t := range tasks {
		if t.Title == "" {
			return errors.New("mygo: App.JumpList.SetTasks: every task needs a title")
		}
	}
	tasks = slices.Clone(tasks)
	for i := range tasks {
		tasks[i].Args = slices.Clone(tasks[i].Args)
	}
	return onMainValue(func() error {
		// The main thread orders the calls, and commits what the last
		// one set.
		j.mu.Lock()
		old := j.tasks
		j.tasks = tasks
		list := j.snapshot()
		j.mu.Unlock()
		err := backend().App().SetJumpList(list)
		if err != nil {
			j.mu.Lock()
			j.tasks = old
			j.mu.Unlock()
		}
		return err
	})
}

// Tasks returns the tasks set with SetTasks.
func (j *JumpList) Tasks() []JumpListTask {
	j.mu.Lock()
	defer j.mu.Unlock()
	tasks := slices.Clone(j.tasks)
	for i := range tasks {
		tasks[i].Args = slices.Clone(tasks[i].Args)
	}
	return tasks
}

// snapshot is the whole list for the backend. j.mu is held.
func (j *JumpList) snapshot() platform.JumpList {
	var list platform.JumpList
	for _, t := range j.tasks {
		list.Tasks = append(list.Tasks, platform.JumpListTask{
			Title: t.Title, Description: t.Description, Path: t.Path, Args: slices.Clone(t.Args),
			WorkingDir: t.WorkingDir, IconPath: t.IconPath, IconIndex: t.IconIndex,
		})
	}
	return list
}
