# 用 fixture 开发视图

这份指南帮助你在不启动 Voxt 辅助进程、不加载模型、不授予麦克风权限、不打开 Voxt 数据目录的情况下修改会议、历史、会话浮窗、更新、首页、引导、应用增强、词典、笔记、语言模型或模型视图。

## 打开第一个 fixture

先按[开发指南](development.zh-CN.md)安装 Go 1.27.1 或更新的 Go 1.27 补丁版本，再在 `Voxt/` 下执行所有命令。

1. 构建并启动 demo：

   ```sh
   MYGO_LOCALE=en VOXTNEW_DEMO_DIR="$PWD/testdata/meeting" GOFLAGS='-tags=meetingdemo' go tool mygo dev ./demo
   ```

2. 确认 **Voxt UI Demo** 窗口显示 `en/appenhance-empty`，这是按名称排序后第一个不带模态对话框的 fixture。

3. 点 **Continue** 或 **Back**，不用重启窗口就能切换 fixture。到首尾后会循环切换，界面本地状态会重置。
4. 在详情 fixture 里点 **Close**。终端会打印记录，fixture 不会改变：

   ```text
   meeting demo action: closeMeetingDetail
   ```

修改视图代码时保持 `mygo dev` 运行，构建成功后它会重新构建并启动 demo。fixture 选择只在启动时读取一次 JSON，所以重新生成 fixture 文件后要重启 demo。

demo 只读取 `testdata/meeting`。它的 MyGo 配置不声明辅助进程、模型运行时或权限。`MYGO_LOCALE` 阻止 MyGo 读取或保存语言偏好；没有设置这个变量时，demo 会把它设为第一个 fixture 的语言。操作只把 action 记录打印到 stdout。

正式构建默认不启用 demo。只设置 `VOXTNEW_DEMO_DIR`、不加 `-tags=meetingdemo` 时，程序会在正常启动之前报错退出。用户界面上没有启用 demo 的开关。

要在不打开窗口的情况下检查 demo 构建，执行：

```sh
GOFLAGS='-tags=meetingdemo' go tool mygo build -skip-dmg ./demo
```

要在不打开窗口的情况下查看 fixture，使用[截图命令](#出截图并检查操作)。测试会创建输出目录。

## 修改视图

会议视图包括 `meetings_page.go`、`meetings_detail.go`、`meetings_playback.go`、`meeting_overlay.go` 和 `meeting_overlay_view.go`。历史视图包括 `history_page.go`、`history_detail.go`、`history_menu.go`、`history_retry.go`、`history_settings.go`、`history_export.go` 和 `history_access.go`。会话浮窗视图包括 `overlay.go`、`overlay_face.go`、`overlay_card.go`、`overlay_card_view.go`、`overlay_lang.go`、`overlay_size.go`、`overlay_think.go` 和 `overlay_wave.go`。更新区域在 `updates_view.go` 中绘制设置 → 诊断与关于和更新对话框。首页视图包括 `home_page.go` 和 `home_ready.go`。引导视图包括 `onboarding_keys.go`、`onboarding_models.go`、`onboarding_steps.go`、`onboarding_try_guide.go` 和 `onboarding_view.go`。应用增强视图包括 `appenhance_page.go`、`appenhance_group.go`、`appenhance_dialogs.go`、`appenhance_preview.go` 和 `appicons.go`。词典视图包括 `dictionary_page.go`、`dictionary_notes.go`、`dictionary_scan.go`、`dictionary_suggest.go` 和 `notes_page.go`。语言模型区域视图包括 `remote_page.go`、`remote_dialog.go`、`remote_generation.go`、`remote_chatgpt.go`、`llm_wait.go` 和 `llm_debug.go`。模型视图包括 `models_page.go`、`models_cards.go`、`models_context.go`、`models_delete.go`、`models_keep.go` 和 `models_network.go`。视图只读取 view state、调用 actions，并持有滚动位置、打开的菜单等界面本地状态，不读取 store、snapshot、settings 或 `app`。

对应的逻辑文件负责构造 view state 和实现 actions：

- `meetings_view_state.go` 和 `meeting_overlay_state.go` 构造会议显示数据。`meetings_controller.go`、`meetings_detail_controller.go`、`meetings_playback_controller.go` 和 `meeting_overlay_window.go` 负责 store 读取、详情更新、音频工作和原生资源。
- `history_view_state.go`、`history_settings_view_state.go` 和 `history_access_view_state.go` 构造历史显示数据。历史控制器负责 store 读取、重试、录音任务、导出和访问 socket。
- `session_overlay_state.go`、`session_overlay_card_state.go` 和 `session_overlay_menu_state.go` 构造会话显示数据。`session_overlay_window.go`、`session_overlay_size.go`、`session_overlay_menu.go` 和 `session_overlay_stream.go` 负责原生资源、测量、菜单和流式文本的时序。`session_overlay_state.go` 中的 `sessionOverlayActions` 定义会话浮窗的用户意图。
- `updates_view_state.go` 构造设置 → 诊断与关于和更新对话框的显示数据。`updates.go` 和 `updates_github.go` 负责更新器 I/O、Release 检查、后台监听和卡片。
- `home_view_state.go` 和 `home_readiness_state.go` 构造首页显示数据。`home_controller.go` 负责历史读取、就绪状态和 actions。
- `onboarding_view_state.go`、`onboarding_models_state.go`、`onboarding_steps_state.go` 和 `onboarding_try_state.go` 构造引导显示数据。`onboarding.go` 负责引导生命周期和模型调查。
- `appenhance_view_state.go` 构造应用增强显示数据。`appenhance_controller.go` 负责规则编辑、设置修改和 App 列表读取，`appicons_controller.go` 负责图标读取和位图解码。
- `dictionary_view_state.go`、`dictionary_notes_state.go` 和 `notes_view_state.go` 构造词典和笔记显示数据。词典和笔记控制器负责 store 读取、设置写入、导入导出、扫描、自动保存和快捷键录制。
- `remote_view_state.go`、`remote_dialog_view_state.go` 和 `llm_debug_view_state.go` 构造语言模型区域的显示数据。控制器负责 settings、凭据、模型发现、连接测试、登录和调试请求。
- `models_view_state.go` 构造模型显示数据。模型控制器负责下载、删除、家族共享设置、运行时内存测量和页面状态缓存。
- `meeting_actions.go`、`history_actions.go`、`updates_actions.go`、`home_actions.go`、`onboarding_actions.go`、`appenhance_actions.go`、`dictionary_actions.go`、`remote_actions.go` 和 `models_actions.go` 定义用户意图。正式 app 由 `app` 实现全部意图，demo 由 `meetingDemoHost` 记录。

尺寸、颜色和控件选择只放在视图文件里。移动代码后运行 `go tool mygo vet .` 和 `go test ./internal/archtest`。

## 新增或修改 fixture 数据

不要手改 fixture JSON。先改生产代码里的构造器和 `meeting_fixture_scenarios_test.go`、`history_fixture_scenarios_test.go`、`session_fixture_scenarios_test.go`、`updates_fixture_scenarios_test.go`、`home_fixture_scenarios_test.go`、`onboarding_fixture_scenarios_test.go`、`appenhance_fixture_scenarios_test.go`、`dictionary_fixture_scenarios_test.go`、`remote_fixture_scenarios_test.go` 或 `models_fixture_scenarios_test.go` 中的场景，再重新生成中英文 fixture：

```sh
VOXTNEW_UPDATE_GOLDEN=1 go test -count=1 -run '^TestMeetingViewStateGolden$' .
```

`TestMeetingViewStateGolden` 通过生产代码的 view state 构造器生成全部 fixture。`TestMeetingFixtureCoverage`、`TestHistoryFixtureCoverage` 和 `TestSessionFixtureCoverage` 从领域声明中读取状态和问题代码。`TestUpdatesFixtureCoverage` 枚举更新区域的显示分支和 Release 检查错误分支，`TestHomeFixtureCoverage` 枚举首页的就绪项动作，`TestOnboardingFixtureCoverage` 枚举引导的步骤、权限状态和快捷键预设，`TestAppEnhancementFixtureCoverage` 枚举应用增强的匹配来源，`TestDictionaryFixtureCoverage` 枚举词典的建议来源和扫描阶段，`TestRemoteFixtureCoverage` 枚举语言模型区域的缺失项、问题代码、生成方式和模型等待，`TestModelsFixtureCoverage` 枚举模型种类、运行时、下载错误和模型驻留选择。这些测试在缺 fixture 时都会失败。

视图需要新数据时，由功能侧把数据加入构造器输入和 view state，并增加一个能证明该值来自真实逻辑的场景。新增用户意图时，在所属区域的 `*_actions.go` 中加入 action，在 `app` 上实现，在 `meetingDemoHost` 上记录，并在该区域的 demo action 测试中加入点击测试。

## 接入下一个区域或 fixture 驱动的小页面

在现有 fixture 窗口上扩展，不要另加启动路径。新区域提供自己的 view state、actions、场景和视图，只需要改下面列出的接入点；`main.go`、`meeting_demo_enabled.go`、`meeting_demo_disabled.go` 和 `demo/mygo.json` 保持不变。其他区域需要共享加载或切换代码时再提取，不要增加 UI 版本选择开关。

给新页面选一个简短的 kind，例如 `note`。视图放在 `*_page.go` 中，只接收 view state、actions 和界面本地状态。然后接好以下各处：

1. 在 `meeting_fixture.go` 的 `meetingFixtureState` 中加入对应字段，并在 `meeting_fixture_test.go` 的 `buildMeetingFixture` 中加入构造分支。在 `meeting_fixture_scenarios_test.go` 中加入场景，然后重新生成 fixture。
2. 在 `meeting_demo.go` 的 `readMeetingDemoFixture` 中接受这个 kind，并在 `meeting_demo_view.go` 的 `meetingDemoView` 中加入渲染分支。在 `selectMeetingFixture` 中初始化和重置界面本地状态。
3. 在所属区域的 `*_actions.go` 中加入 action，在 `app` 上实现，并在 `meetingDemoHost` 上记录。
4. 在 `meeting_demo_test.go` 的 `meetingDemoSentinel` 中加入预期可见文字，并在该区域的 demo action 测试中加入控件测试。
5. 正式页面还要由所属控制器构造状态，并在 `pages.go` 中注册视图。只做 demo 练习时，完成 fixture、视图和 action 测试即可。

运行 `go test ./internal/archtest` 检查新视图的 import，运行 `go test -run '^(TestMeeting(ViewStateGolden|FixtureCoverage)|TestHistoryFixtureCoverage|TestSessionFixtureCoverage|TestUpdatesFixtureCoverage|TestHomeFixtureCoverage|TestOnboardingFixtureCoverage|TestAppEnhancementFixtureCoverage|TestDictionaryFixtureCoverage|TestRemoteFixtureCoverage|TestModelsFixtureCoverage)$' .` 检查现有 fixture 覆盖。新区域的 golden、渲染和 action 测试放在这些测试旁边。

## 出截图并检查操作

生成中英文全部 fixture 的截图：

```sh
VOXTNEW_DEMO_SHOTS=/tmp/voxt-meeting-demo go test -count=1 -run '^TestMeetingDemoScreenshots$' .
```

测试为每个 fixture 写一张 `<locale>-<scenario>.png`，并检查预期文字和非空图像。

检查 fixture 切换和控件到 action 的接线：

```sh
go test -count=1 -run '^Test(MeetingDemo(ActionControls|SwitchesFixtures)|HistoryDemoActionControls|SessionDemo.*|SessionOverlayDemoMatchesProduction|UpdatesDemoActionControls|HomeDemoActionControls|OnboardingDemo|AppEnhancementDemoActionControls|DictionaryDemoActionControls|RemoteDemoActionControls|ModelsDemo)$' .
go test -tags=meetingdemo -count=1 -run '^Test(Meeting|History|Session|Updates|Home|Onboarding|AppEnhancement|Dictionary|Remote|Models)(Demo|OverlayDemoMatchesProduction)' .
```

第一条命令点击每个会触发 action 的会议、历史、会话浮窗、更新、首页、引导、应用增强、词典、语言模型和模型控件以及本地取消控件，再比较完整的调用记录，覆盖菜单、确认对话框、播放、搜索、历史设置、导出、清理、“其他程序”、录音胶囊、答案卡片、更新偏好、Release 对话框、就绪项修复、最近结果、权限、模型选择、引导试用、规则编辑器、浏览器列表、词典词条、笔记、服务卡片、服务对话框、LLM 调试视图、模型下载、删除、网络和模型驻留时长。它还会比较正式代码和 JSON fixture 的渲染结果。第二条命令还会编译启用 demo 的入口。

这些测试使用 `ui.Tester`，不需要窗口、辅助进程二进制、模型、麦克风、网络或 Voxt 数据目录，也能在 Linux 上运行。

demo 不能验证原生浮窗位置、材质、防共享、真实音频采集、辅助进程行为、模型输出、真实下载和删除、运行时行为、代理和网络行为、全局快捷键、持久化或文件输出。这些仍要在 Mac 上用正式 app 检查。会议设置区域的对话框仍未接入 demo，所以会议页面 fixture 隐藏了对应按钮。功能区域的“试一试”卡片也不属于首页 demo，首页 fixture 会隐藏这一节。模型 fixture 也不接入设置引导和远程服务视图。
