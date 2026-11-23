// Command idlemem launches apps and reports how much memory each takes
// once idle, alone and with the processes it runs, its webview's:
// scripts/bench.ts measures release builds of examples/hello and
// examples/counter-native with it on every push to main. It needs a
// desktop session.
//
//	go run ./internal/idlemem [-count 3] [-idle 6s] app...
//
// Each app runs count times, for idle each time, and the medians print as
// JSON, by the name of the app's file: {"hello": {"app": bytes, "total":
// bytes}}. Memory is what each platform shows for a process:
//
//   - macOS: the physical footprint, as Activity Monitor shows it, of the
//     app and of the WebKit processes that started with it and share its
//     responsible process (a terminal, say, for an app not in a bundle);
//   - Linux: the proportional set size of the app and of its descendants,
//     WebKitGTK's processes, so that pages they share count once;
//   - Windows: the private working set, as Task Manager shows it, of the
//     app and of its descendants, WebView2's processes.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"time"
)

func main() {
	count := flag.Int("count", 3, "how many times each app runs")
	idle := flag.Duration("idle", 6*time.Second, "how long an app runs before it is measured")
	flag.Parse()
	out := map[string]map[string]uint64{}
	for _, bin := range flag.Args() {
		name := strings.TrimSuffix(filepath.Base(bin), ".exe")
		var apps, totals []uint64
		for range *count {
			app, total, err := measure(bin, *idle)
			if err != nil {
				fmt.Fprintf(os.Stderr, "idlemem: %s: %v\n", name, err)
				os.Exit(1)
			}
			fmt.Fprintf(os.Stderr, "%s: %.1f MB, %.1f MB with its processes\n", name, float64(app)/1e6, float64(total)/1e6)
			apps, totals = append(apps, app), append(totals, total)
		}
		out[name] = map[string]uint64{"app": median(apps), "total": median(totals)}
	}
	if err := json.NewEncoder(os.Stdout).Encode(out); err != nil {
		fmt.Fprintln(os.Stderr, "idlemem:", err)
		os.Exit(1)
	}
}

// measure runs bin for idle, and returns the memory of its process, and of
// it with the processes it runs.
func measure(bin string, idle time.Duration) (app, total uint64, err error) {
	s := before()
	cmd := exec.Command(bin)
	cmd.Env = append(os.Environ(), "MYGO_ENV=production")
	if err := cmd.Start(); err != nil {
		return 0, 0, err
	}
	exited := make(chan error, 1)
	go func() { exited <- cmd.Wait() }()
	select {
	case err := <-exited:
		return 0, 0, fmt.Errorf("exited after starting: %v", err)
	case <-time.After(idle):
	}
	app, total, procs, err := usage(cmd.Process.Pid, s)
	stop(cmd.Process, procs)
	select {
	case <-exited:
	case <-time.After(10 * time.Second):
		cmd.Process.Kill()
		<-exited
	}
	return app, total, err
}

func median(v []uint64) uint64 {
	s := slices.Clone(v)
	slices.Sort(s)
	if len(s)%2 == 1 {
		return s[len(s)/2]
	}
	return (s[len(s)/2-1] + s[len(s)/2]) / 2
}
