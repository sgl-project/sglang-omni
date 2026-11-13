//go:build windows

package main

import "github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"

func (server *helperServer) serveApps(message proto.Message) {
	switch message.Method {
	case proto.MethodAppList:
		var params proto.AppListParams
		if err := message.DecodeParams(&params); err != nil {
			server.fail(message.ID, proto.CodeBadRequest, "Bad params: %v", err)
			return
		}
		var apps []proto.AppInfo
		var listErr error
		if err := runShellCOM(func() { apps, listErr = listWindowsApps(params.Installed) }); err != nil {
			server.fail(message.ID, proto.CodeInternal, "Could not initialize the app catalogue: %v", err)
			return
		}
		if listErr != nil {
			server.fail(message.ID, proto.CodeInternal, "Could not enumerate apps: %v", listErr)
			return
		}
		server.result(message.ID, proto.AppListResult{Apps: apps})
	case proto.MethodAppIcons:
		var params proto.AppIconsParams
		if err := message.DecodeParams(&params); err != nil {
			server.fail(message.ID, proto.CodeBadRequest, "Bad params: %v", err)
			return
		}
		if len(params.Apps) > proto.MaxIcons {
			server.fail(message.ID, proto.CodeBadRequest, "At most %d apps a request", proto.MaxIcons)
			return
		}
		var result proto.AppIconsResult
		var iconErr error
		if err := runShellCOM(func() {
			running, _ := runningApps()
			catalog := mergeWindowsApps(running, installedApps())
			result, iconErr = makeAppIcons(params, catalog, extractAppIcon)
		}); err != nil {
			server.fail(message.ID, proto.CodeInternal, "Could not initialize icon extraction: %v", err)
			return
		}
		if iconErr != nil {
			server.fail(message.ID, proto.CodeBadRequest, "Bad icon request: %v", iconErr)
			return
		}
		server.result(message.ID, result)
	}
}
