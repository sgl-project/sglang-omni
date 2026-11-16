# 用 fixture 开发会议视图

这份指南帮助你在不启动 Voxt 辅助进程、不加载模型、不授予麦克风权限、不打开 Voxt 数据目录的情况下修改会议视图。所有命令都在 `Voxt/` 下执行。

## 打开第一个 fixture

1. 构建并启动 demo：

   ```sh
   MYGO_LOCALE=en VOXTNEW_DEMO_DIR="$PWD/testdata/meeting" GOFLAGS='-tags=meetingdemo' go tool mygo dev ./demo
   ```

2. 点 **Continue** 或 **Back**，不用重启窗口就能切换 fixture。

demo 只读取 `testdata/meeting`。它的 MyGo 配置不声明辅助进程、模型运行时或权限。窗口不保存状态。操作只把 action 记录打印到 stdout，不修改 fixture。

正式构建默认不启用 demo。只设置 `VOXTNEW_DEMO_DIR`、不加 `-tags=meetingdemo` 时，程序会在正常启动之前报错退出。

要在不打开窗口的情况下检查 demo 构建，执行：

```sh
GOFLAGS='-tags=meetingdemo' go tool mygo build -skip-dmg ./demo
```

## 修改视图

会议视图包括 `meetings_page.go`、`meetings_detail.go`、`meetings_playback.go`、`meeting_overlay.go` 和 `meeting_overlay_view.go`。视图只读取 view state、调用 actions，并持有滚动位置、打开的菜单等界面本地状态，不读取 store、snapshot、settings 或 `app`。

对应的逻辑文件负责构造 view state 和实现 actions：

- `meetings_view_state.go` 构造列表、详情和播放的显示数据。
- `meeting_overlay_state.go` 构造浮窗的显示数据。
- `meetings_controller.go`、`meetings_detail_controller.go` 和 `meetings_playback_controller.go` 负责 store 读取、详情更新和音频工作。
- `meeting_overlay_window.go` 负责原生浮窗和缓存的浮窗状态。
- `meeting_actions.go` 定义用户意图。正式 app 由 `app` 实现，demo 由 `meetingDemoHost` 记录。

尺寸、颜色和控件选择只放在视图文件里。移动代码后运行 `go tool mygo vet .` 和 `go test ./internal/archtest`。

## 新增或修改 fixture 数据

不要手改 fixture JSON。先改生产代码里的构造器和 `meeting_fixture_scenarios_test.go` 中的场景，再重新生成中英文 fixture：

```sh
VOXTNEW_UPDATE_GOLDEN=1 go test -count=1 -run '^TestMeetingViewStateGolden$' .
```

`TestMeetingViewStateGolden` 通过生产代码的 view state 构造器生成全部 fixture。`TestMeetingFixtureCoverage` 从领域声明中读取状态和问题代码，缺 fixture 时会失败。

视图需要新数据时，由功能侧把数据加入构造器输入和 view state，并增加一个能证明该值来自真实逻辑的场景。新增用户意图时，在 `meeting_actions.go` 中加入 action，在 `app` 上实现，在 `meetingDemoHost` 上记录，并在 `TestMeetingDemoActionControls` 中加入点击测试。

## 新增一个 fixture 驱动的小页面

给新页面选一个简短的 kind，例如 `note`。视图放在 `*_page.go` 中，只接收 view state、actions 和界面本地状态。然后接好以下各处：

1. 在 `meeting_fixture.go` 的 `meetingFixtureState` 中加入对应字段，并在 `meeting_fixture_test.go` 的 `buildMeetingFixture` 中加入构造分支。在 `meeting_fixture_scenarios_test.go` 中加入场景，然后重新生成 fixture。
2. 在 `meeting_demo.go` 的 `readMeetingDemoFixture` 中接受这个 kind，并在 `meeting_demo_view.go` 的 `meetingDemoView` 中加入渲染分支。在 `selectMeetingFixture` 中初始化和重置界面本地状态。
3. 在 `meeting_actions.go` 中加入 action，在 `app` 上实现，并在 `meetingDemoHost` 上记录。
4. 在 `meeting_demo_test.go` 的 `meetingDemoSentinel` 中加入预期可见文字，并在 `TestMeetingDemoActionControls` 中加入控件测试。
5. 正式页面还要由所属控制器构造状态，并在 `pages.go` 中注册视图。只做 demo 练习时，完成 fixture、视图和 action 测试即可。

运行 `go test ./internal/archtest` 检查新视图的 import，运行 `go test -run '^TestMeeting(ViewStateGolden|FixtureCoverage|Demo)' .` 检查 fixture、渲染和 action。

## 出截图并检查操作

生成中英文全部 fixture 的截图：

```sh
VOXTNEW_DEMO_SHOTS=/tmp/voxt-meeting-demo go test -count=1 -run '^TestMeetingDemoScreenshots$' .
```

检查 fixture 切换和控件到 action 的接线：

```sh
go test -count=1 -run '^TestMeetingDemo(ActionControls|SwitchesFixtures)$' .
```

这些测试使用 `ui.Tester`，不需要窗口、辅助进程二进制、模型、麦克风、网络或 Voxt 数据目录，也能在 Linux 上运行。

demo 不能验证原生浮窗位置、材质、防共享、真实音频采集、辅助进程行为、模型输出、全局快捷键、持久化或文件输出。这些仍要在 Mac 上用正式 app 检查。设置区域的对话框不在本次试点范围内，所以 demo 隐藏了对应按钮。
