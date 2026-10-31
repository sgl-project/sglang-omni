# 发布与更新

VoxtNew 是 sglang-omni 仓库里的开源 app（Apache-2.0），ActNow 家族的一员。产品页、下载、更新和更新日志都在官网 actnow.dev：产品页 <https://actnow.dev/voxtnew>，下载 <https://actnow.dev/download/voxtnew>（要登录），更新源 `https://actnow.dev/updates/voxtnew`（beta 通道是 `…/voxtnew/beta`）。VoxtNew 有**自己的**更新密钥（§1.3），和家族其他 app 一样用 **wei guan（779M533DL5）** 的 Developer ID 签名、用 york@actnow.dev 公证（用户 10-19 决定；之前用 RY2TQTYSKQ，§9）。**不用 GitHub Release**。

> **还差官网这一边**：VoxtNew 的产品页已经上线，ingest token 已经发（`RELEASE_INGEST_TOKEN_VOXTNEW`）。还要 ActNow 项目在面板 /panel/releases 的 voxtnew 设置里填上公钥 `MbROqANE8pK8IQ8hdtUtR/m5SqUY2nTgzHwrQt04Zps=`（已在 `notices/from-voxtnew.md` 里请他们填）：填了以后后台才校验更新包的签名。

发布版本在一台 Apple 芯片的 Mac 上用一条命令构建：[scripts/release.sh](../scripts/release.sh)。它编译辅助进程、Omni 运行时（语音识别）和 voxtnew-llm（语言模型），把三者放进 app，`go tool mygo build` 打包、签名、（可选）公证，[scripts/release-verify.sh](../scripts/release-verify.sh) 检查结果，最后用官网的发布脚本 [scripts/actnow-publish.sh](../scripts/actnow-publish.sh)（`https://actnow.dev/actnow-publish.sh` 的原样副本，见 §2.3）把文件交给官网后台，或者只做 `--dry-run`。

```
scripts/mac-gui.sh scripts/release.sh（在 Mac 上，ssh 里经 mac-gui.sh 进图形会话；VOXTNEW_SIGN_IDENTITY = 779M533DL5 的 Developer ID）
   │  先试签一个文件（ssh 里钥匙串锁着就立刻停下）；更新私钥和 release/mygo.release.json 的公钥配对
   │  检查版本号（mygo.json、helper 和 llm 的 Info.plist）、CHANGELOG
   ├─ build-helper.sh              → resources/darwin-arm64/helper/voxtnew-helper
   ├─ build-omni.sh --clean        → resources/darwin-arm64/omni/bin/（qwen3_asr_server、MLX 的库、mlx.metallib）
   ├─ build-llm.sh                 → resources/darwin-arm64/llm/（voxtnew-llm、mlx.metallib、llama.framework、Hub bundle）
   │     （--runtimes DIR：直接用 DIR/omni 和 DIR/llm，不重新编译）
   ├─ release-config.sh            合并 release/mygo.release.json（公钥、url、prereleases）、签名身份、公证配置、
   │                               更新私钥的文件（updates.privateKey）
   ├─ go tool mygo build           签名每个嵌套的可执行文件、库和 framework（hardened runtime、时间戳），
   │                               dmg；--notarize 时公证并 staple；有更新私钥时还有更新包、增量、清单（不加 -upload）
   ├─ release-verify.sh            签名、entitlement、运行时都在而且能从 app 里启动、hardened runtime、Team、
   │                               （公证后）Gatekeeper、staple、清单
   └─ actnow-publish.sh --app voxtnew --dir build/darwin-arm64 --changelog CHANGELOG.md --min-macos 15.0
                                   --publish 时上传（ACTNOW_INGEST_TOKEN）；否则有更新文件就 --dry-run
   结束时 mygo.json 恢复原样，暂存的运行时删掉（--keep-runtimes 保留）
```

只发 Apple 芯片（darwin/arm64），macOS 15 起：辅助进程、Omni 运行时和 voxtnew-llm 都只编 arm64，MLX 只支持 Apple 芯片。

**app 里有什么**（0.1.0-beta.1：app 280 MB，dmg 45 MB；见文末“实测记录”）：

| 路径（`VoxtNew.app/Contents/…`） | 是什么 | 签名 |
|---|---|---|
| `MacOS/VoxtNew` | Go + MyGo 的 app | `helper/Support/app.entitlements`（`audio-input`） |
| `Resources/helper/voxtnew-helper` | Swift 辅助进程：麦克风、快捷键、辅助功能、粘贴 | `helper/Support/helper.entitlements`（`audio-input`） |
| `Resources/omni/bin/qwen3_asr_server`、`*.dylib`、`mlx.metallib` | Omni 原生运行时（sglang_omni_mlx/native）：Qwen3-ASR 和 Silero VAD | 保留原 entitlement（没有） |
| `Resources/llm/voxtnew-llm`、`llama.framework`、`mlx.metallib`、`swift-transformers_Hub.bundle` | 语言模型进程（mlx-swift-lm 和 llama.cpp） | `llm/Support/llm.entitlements`（空：Metal 不需要） |

`mlx.metallib` 不是代码，是资源，由 app 的签名封住。模型本身不在 app 里，用户在 app 里下载到 ActNow 家族的模型库 `$HUB`（默认 `~/.cache/huggingface/hub`，和家族其他 app 共用；见 [architecture.md](architecture.md)）。

更新用的是 MyGo 官方方案（`mygo.Updater` + updater 插件的 `native.Plugin` 窗口），说明见 MyGo 的 `docs/updates.md`、`docs/plugins/updater.md`；beta 通道是我们的补丁 0006（`third_party/mygo-patches/`，和 Auralink 的同一个）。

## 0. 两个通道

| | 正式版 | beta 版 |
|---|---|---|
| 版本号 | `0.2.0` | `0.2.0-beta.1`（版本号里有 `-`） |
| 更新源 | `https://actnow.dev/updates/voxtnew` | `https://actnow.dev/updates/voxtnew/beta` |
| 清单里文件的地址 | `…/updates/voxtnew/<文件>` | `…/updates/voxtnew/beta/<文件>`（官网后台会检查） |
| 谁会收到 | 所有人 | 只有打开了 设置 → 诊断与关于 → **Beta 更新** 的人 |

- **Beta 更新开关**：打开时检查更新读 beta 通道的清单（官网保证它始终是 beta 和正式版中较新的那个），比当前版本新就提示；读不到时退回正式版的清单。关闭时只读正式版的清单。
- **默认值**：用户没动过开关时，当前运行的是 beta 就默认打开，是正式版就默认关闭。用户拨过一次就以用户为准。
- **版本号只能往上走**（官网后台的规则）：新版本必须高于这个通道发布过的所有版本，已发布或下架过的版本号不能再用。
- **不降级**：只会更新到比当前更新的版本。beta 用户关掉开关后，要等到比当前 beta 新的正式版才会更新。
- **还没有正式版时**（0.1.0-beta.1 是第一个版本），正式版通道的清单是 404，关着开关检查更新显示“已是最新”，不报错（`mygo.ErrNotPublished`）。

## 1. 一次性准备

都在发布用的 Mac（mac-air）上做。下面的命令都不含秘密值：要输入的地方由命令提示，或者从文件读。

### 1.1 Developer ID 证书

签名用 **Developer ID Application: wei guan (779M533DL5)**：york@actnow.dev 的个人账号，家族其他 app（Auralink 等）也用它（用户 10-19 决定，推翻了家族规范 §17 Q1 原来给 VoxtNew 定的 RY2TQTYSKQ / Xiaofeng Cao，§9）。mac-air 上 779M533DL5 只有**一张**有效证书，`security find-identity -v -p codesigning` 却列出 4 次：它同时在登录钥匙串和 `~/.cache/herenow-signing/herenow-signing.keychain-db` 里（10-19 查过）：

| SHA-1 | 生效 | 到期 |
|---|---|---|
| `EBCFF160A0270A61E62504F924CC65A4FF309AFB` | 2026-10-16 03:01:32 GMT | 2027-02-01 22:12:15 GMT |

它由 Developer ID Certification Authority（旧的 G1 中间证书）签发。codesign 遇到同名的不同证书会报 “ambiguous”，所以**按 SHA-1 签**，用 [scripts/release-identity.sh](../scripts/release-identity.sh) 选：它只看 `find-identity -v` 列出的（没过期、没吊销、有私钥），同一张证书出现几次只算一次；同一个 Team 有几张时取**最新**的（生效时间最晚，再比到期时间，再比 SHA-1），和钥匙串的顺序无关，现在是 `EBCFF160…`。以后证书续期了，脚本自动选新的。

```sh
scripts/release-identity.sh 779M533DL5          # 打印 <SHA-1><Tab>Developer ID Application: wei guan (779M533DL5)，几张时候选和日期在 stderr
export VOXTNEW_SIGN_IDENTITY="$(scripts/release-identity.sh 779M533DL5 | cut -f1)"
```

**这张证书 2027-02-01 到期**（G1 中间证书的到期日）。带时间戳签的版本到期后照样能装、能运行、能更新；到期前要在 developer.apple.com 用 779M533DL5 再申请一张（G2 中间证书），导入钥匙串，`release-identity.sh` 会选它。家族其他 app 用的是同一张，续期一次大家都用新的。同一个 Team 换证书，TCC 授权保留（§4）。

**还没用它签过 VoxtNew**（10-19）：同一张证书在 herenow-signing 钥匙串里的那一份排在搜索列表前面，ssh 里是锁着的。如果图形会话里 codesign 因此报 “ambiguous”、弹出解锁那个钥匙串的窗口或 `errSecInternalComponent`，`release.sh` 一开始的试签就会停下；那时按下面 “用不了时” 的办法在终端里跑，或者问用户 herenow-signing 钥匙串怎么处理（不要自己改钥匙串的搜索列表，别的项目在用）。

**ssh 里怎么签**：ssh 会话里登录钥匙串是锁着的，codesign 报 `errSecInternalComponent`（`release.sh` 一开始先试签一个文件，签不了就停下，不会白编 3 分钟）。不需要密码的办法：[scripts/mac-gui.sh](../scripts/mac-gui.sh) 把命令当成一次性的 LaunchAgent 放进**登录在屏幕上的那个用户的图形会话**（`launchctl bootstrap gui/<uid>`）里跑，那里钥匙串是解锁的，codesign 不弹窗（10-17 在 mac-air 上用当时 RY2TQTYSKQ 的两张证书都签成功，带 Apple 的时间戳）。输出实时传回 ssh，退出码照传，Ctrl-C 会停掉它；环境变量（签名身份、更新私钥的路径、`ACTNOW_INGEST_TOKEN` 等，脚本头部列着）经一个 0700 临时目录里的文件交给它，任务一开始就删掉。条件是 mac-air 上这个用户登录着（锁屏也可以）。

```sh
scripts/mac.sh 'scripts/mac-gui.sh scripts/release.sh --runtimes $HOME/voxtnew-runtime'
```

用不了时（没有人登录、换了机器）有两个办法：
- 在 Mac 的“终端”里运行 `scripts/release.sh`；第一次使用时系统可能会问是否允许 codesign 使用这把私钥，选“始终允许”。
- 或者一次性地在 ssh 里解锁钥匙串，并允许 codesign 不弹窗使用私钥（两条都会提示输入**登录密码**，由用户自己输入）：
  ```sh
  security unlock-keychain ~/Library/Keychains/login.keychain-db
  security set-key-partition-list -S apple-tool:,apple:,codesign: -s -t private ~/Library/Keychains/login.keychain-db
  ```
  之后 ssh 里直接跑 `scripts/release.sh`，直到钥匙串再次锁上（睡眠、超时）。

### 1.2 公证凭据（notarytool）

公证用 Apple ID **york@actnow.dev**、Team **779M533DL5** 和一个 **App 专用密码**（用户 10-19 决定，和 Auralink 一样；york@actnow.dev 就是 779M533DL5 这个个人账号本身，签名证书也是它的），存进登录钥匙串的 notarytool 配置 `voxtnew-notary`。**mac-air 上还没有**任何 notarytool 配置（10-17 查过：`voxtnew-notary`、`auralink-notary`、`notary`、`AC_PASSWORD` 都没有；10-19 登录钥匙串里仍没有 notarytool 的条目）。

york@actnow.dev 要开启双重认证（App 专用密码需要）。也可以改用 779M533DL5 的 App Store Connect API key（`store-credentials --key … --key-id … --issuer …`）。Auralink 的配置 `auralink-notary` 是同一个 Apple ID 和 Team，但 VoxtNew 用自己的配置名（可以用同一个 App 专用密码，也可以单独生成一个）。

1. 打开 [account.apple.com](https://account.apple.com)，用 york@actnow.dev 登录 → **登录与安全** → **App 专用密码** → 生成，名称写 `VoxtNew notarization`；复制显示的密码（只显示一次），存进密码管理器。
2. 在 Mac 的终端里存成配置（不加 `--password`，notarytool 会提示输入，密码不进 shell 历史）：
   ```sh
   xcrun notarytool store-credentials voxtnew-notary --apple-id york@actnow.dev --team-id 779M533DL5
   ```
   成功会显示 “Validating your credentials… Success”。检查（不提交任何东西）：`xcrun notarytool history --keychain-profile voxtnew-notary`（ssh 里经 `scripts/mac-gui.sh`）。
3. 以后改了 Apple ID 的密码，App 专用密码全部失效，重新生成，再执行一次第 2 步。

### 1.3 更新签名密钥

VoxtNew 有**自己的**一对密钥（10-17 在 mac-air 上用 `go tool mygo keygen` 生成，不再和 Auralink 共用）：

- 公钥 `MbROqANE8pK8IQ8hdtUtR/m5SqUY2nTgzHwrQt04Zps=`，写在 [release/mygo.release.json](../release/mygo.release.json) 的 `updates.publicKey`，官网面板 /panel/releases 的 voxtnew 设置里也要填同一个（由 ActNow 项目填）。
- 私钥在 mac-air 的 `~/.config/voxtnew-release/update-keys/mygo-update.key`（600，目录 700；同目录的 `mygo-update.pub` 是公钥），**不进仓库**、不进日志。

`release.sh` 有身份签名时自己找私钥：环境变量 `MYGO_UPDATER_PRIVATE_KEY`（密钥本身），否则 `VOXTNEW_UPDATE_KEY` 指的文件（默认就是上面那个路径），交给 MyGo 的是文件路径（`updates.privateKey`），密钥不进命令行和环境。开始前它核对私钥和 `release/mygo.release.json` 的公钥是一对，不是就停下。

**要做（用户）**：把 `mygo-update.key` 的内容存进密码管理器（例如 `cat` 后复制，别贴进聊天或日志）。已安装的 VoxtNew 只接受这把私钥签的更新，丢了就再也收不到自动更新（§7）。

### 1.4 官网的 ingest token

`ACTNOW_INGEST_TOKEN`：官网给 VoxtNew 的发布 token，只能操作 voxtnew 的版本，只在 `--publish` 时用。它的值是 `~/.config/actnow/ingest-tokens.env` 里 `RELEASE_INGEST_TOKEN_VOXTNEW` 那一行；这个文件在**开发用的 Linux 机器**上，mac-air 上没有。不写进仓库、命令行或日志：

- 从 Linux 发布时，[scripts/mac.sh](../scripts/mac.sh) 把本地环境里的 `ACTNOW_INGEST_TOKEN` 经 ssh 的标准输入交给远端命令（不出现在 Mac 的 `ps` 里），`mac-gui.sh` 再经 0700 的临时文件交给图形会话里的 `release.sh`：
  ```sh
  ACTNOW_INGEST_TOKEN="$(sed -n 's/^RELEASE_INGEST_TOKEN_VOXTNEW=//p' ~/.config/actnow/ingest-tokens.env)" \
    scripts/mac.sh 'scripts/mac-gui.sh scripts/release.sh --notarize voxtnew-notary --publish'
  ```
- 在 Mac 的终端里发布时，自己 `export ACTNOW_INGEST_TOKEN=…`（从密码管理器粘贴，命令前加空格不进历史）。

### 1.5 第一次试跑

先做一个不公证、不发布的签名构建，确认签名和检查都通过（用 `--runtimes ~/voxtnew-runtime` 约 2 分钟，编运行时约 10 分钟）。在 Linux 上：

```sh
MAC_DIR=code/VoxtNew-rel scripts/mac.sh 'export VOXTNEW_SIGN_IDENTITY="$(scripts/release-identity.sh 779M533DL5 | cut -f1)" https_proxy=http://127.0.0.1:7897;
  scripts/mac-gui.sh scripts/release.sh --runtimes $HOME/voxtnew-runtime'
```

（mac-air 上要 `https_proxy`：MyGo 读官网上已发布的清单来做增量，actnow-publish.sh 上传也要连官网；不设时增量那一步会超时，只打印警告。）

`release-verify.sh` 这时的模式是 `signed`：每段代码（app、辅助进程、qwen3_asr_server、每个 dylib、voxtnew-llm、llama.framework、libmygo-sqlite3）都要通过 `codesign --verify --strict`，带 hardened runtime、时间戳和同一个 Team（779M533DL5）；Omni 服务和 voxtnew-llm 要能从 app 里启动（hardened runtime 的库验证要求同一个 Team 签名的库）。Gatekeeper 这时会拒绝（“Unnotarized Developer ID”），脚本只把结论打印出来。最后 `actnow-publish.sh --dry-run` 检查更新文件、清单地址和更新说明。再加上 `--notarize voxtnew-notary` 跑一次，模式变成 `developer-id`，Gatekeeper 必须接受（`source=Notarized Developer ID`），app 和 dmg 都要 staple。

## 2. 发一个版本

### 2.1 步骤（beta 和正式版一样）

1. **改版本号**，三处要一致（`release.sh`、`build-helper.sh`、`build-llm.sh` 都会检查）。beta 用 `0.x.y-beta.n`，正式版用 `0.x.y`：
   ```sh
   v=0.1.0-beta.2        # 或 0.1.0
   jq --arg v "$v" '.version = $v' mygo.json > mygo.json.tmp && mv mygo.json.tmp mygo.json
   /usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $v" helper/Support/Info.plist   # macOS
   /usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $v" llm/Support/Info.plist
   ```
   app 的 `CFBundleShortVersionString` 是完整版本号（更新比较用它）；`CFBundleVersion` 由 MyGo 写成数字部分（`0.1.0`，补丁 0007）。
2. **写 CHANGELOG.md**：把最上面的 `## [Unreleased]` 改成 `## [0.1.0-beta.2] - 2026-11-02`（版本号和 mygo.json 完全一样），英文在前（`### Added`、`### Fixed`…），然后是 `### 中文`，下面用 `####` 小标题（`#### 新增`、`#### 修复`…）。更新窗口显示整节；官网用同一节：发布脚本从第一个含汉字的 `###` 标题（`### 中文`，这个标题本身去掉）分成 `notes`（英文）和 `notesZh`，后台还要求清单里的 notes（整节）包含英文部分。`--publish` 时没有这一节就失败。
3. 提交、推到 main，等 CI（[voxtnew-ci.yaml](../../.github/workflows/voxtnew-ci.yaml)）通过，在 mac-air 上跑一遍回归（`scripts/mac-regress.sh`，[manual-test.md](manual-test.md) §10）。
4. 从 Linux 同步到发布用的 checkout 并构建（§1.1 的 `mac-gui.sh`；更新私钥 `release.sh` 自己从 `~/.config/voxtnew-release/` 读，§1.3）：
   ```sh
   # 先不发布：构建、公证、检查、actnow-publish.sh --dry-run
   MAC_DIR=code/VoxtNew-rel scripts/mac.sh 'export VOXTNEW_SIGN_IDENTITY="$(scripts/release-identity.sh 779M533DL5 | cut -f1)" https_proxy=http://127.0.0.1:7897;
     scripts/mac-gui.sh scripts/release.sh --notarize voxtnew-notary'
   # 发布（会重新构建；或者见下面“只发布”）：token 经 ssh 的标准输入过去（§1.4）
   ACTNOW_INGEST_TOKEN="$(sed -n 's/^RELEASE_INGEST_TOKEN_VOXTNEW=//p' ~/.config/actnow/ingest-tokens.env)" \
     MAC_DIR=code/VoxtNew-rel scripts/mac.sh 'export VOXTNEW_SIGN_IDENTITY="$(scripts/release-identity.sh 779M533DL5 | cut -f1)" https_proxy=http://127.0.0.1:7897;
     scripts/mac-gui.sh scripts/release.sh --notarize voxtnew-notary --publish'
   ```
   在 Mac 的终端里也可以直接跑 `scripts/release.sh …`（`ACTNOW_INGEST_TOKEN` 自己 export）。构建时**不加 `-upload`**：文件都经 `actnow-publish.sh` 交给官网后台，它校验后写进 R2，最后才写公开清单。频道由版本号决定：带 `-` 的是 beta，清单和更新包的地址在 `…/updates/voxtnew/beta/` 下（MyGo 的 `updates.prereleases`：`updates.url` 保持 `…/updates/voxtnew`，beta 版本自动放进它的 `beta/`；后台会检查）。公证通常几分钟。增量来自官网两个通道上已发布的版本（§5.2）；第一个版本没有增量。
5. 打 tag 记下发布的提交：`git tag -a "voxtnew-v$v" -m "VoxtNew $v" && git push origin "voxtnew-v$v"`（加前缀，不和 sglang-omni 自己的 tag 混在一起）。
6. 检查：
   ```sh
   curl -s https://actnow.dev/updates/voxtnew/beta/update-darwin-arm64.json | jq '{version, deltas: [.deltas[]?.from]}'
   curl -s https://actnow.dev/updates/voxtnew/update-darwin-arm64.json | jq .version    # 正式版
   ```
   从官网下载页装一遍 dmg：`spctl -a -vv /Applications/VoxtNew.app` 应显示 `accepted` 和 `source=Notarized Developer ID`。给用户装在 mac-air 上时，第一次从 RY2TQTYSKQ 签的版本换过来要先 reset 授权（§4.1）。

**只发布**（构建已经做好、检查过，在同一个 checkout 里；不需要钥匙串，ssh 里直接跑）：
```sh
ACTNOW_INGEST_TOKEN="$(sed -n 's/^RELEASE_INGEST_TOKEN_VOXTNEW=//p' ~/.config/actnow/ingest-tokens.env)" \
  MAC_DIR=code/VoxtNew-rel scripts/mac.sh 'export https_proxy=http://127.0.0.1:7897;
  bash scripts/actnow-publish.sh --app voxtnew --dir build/darwin-arm64 --changelog CHANGELOG.md --min-macos 15.0'
```
（`MAC_DIR` 的同步不碰 `build/`。）

**发布失败**：脚本打印官网的错误（不含 token）并中止这次会话。常见的：

| 官网回复 | 意思 | 处理 |
|---|---|---|
| 401 / 403 | token 缺失、不对，或是别的 app 的 | 换 `ACTNOW_INGEST_TOKEN` |
| 404 / 422 未知 app | 官网不认识 voxtnew | 问 ActNow 项目（产品和 token 10-16 已经有了） |
| 409 `version_exists` | 这个版本号已经有了 | 发一个新版本号 |
| 422 `version_not_newer` | 不比这个通道发布过的版本高 | 改版本号 |
| 422 `invalid_artifacts` / `verification_failed` | 文件名、大小、sha256、清单地址、签名或 notes 不对（`details.errors` 逐条列出） | 看是哪一条；清单地址不对多半是没走 `release-config.sh` |

### 2.2 不签名的发布式构建（测试用）

不设 `VOXTNEW_SIGN_IDENTITY` 时，`release.sh` 做一个 ad hoc 签名、**没有更新源**的构建，其他步骤一样（运行时都在里面、检查通过）。用来在发版前确认打包没问题，也是 mac-air 上回归用的构建：

```sh
MAC_DIR=code/VoxtNew-rel scripts/mac.sh 'export PATH=/opt/homebrew/bin:$PATH; scripts/release.sh --runtimes $HOME/voxtnew-runtime'
```

签名这一路不用证书也能测：[scripts/mac-sign-e2e.sh](../scripts/mac-sign-e2e.sh) 在一个临时钥匙串里造一个自签的代码签名身份，用它跑 `release.sh`（`release-verify.sh signed`：每段代码都带 hardened runtime、时间戳），结束时删掉钥匙串、恢复钥匙串搜索列表。自签证书没有 Team ID，库验证不让运行时加载它们的库，所以“从 app 里启动”这两项只报告、不算失败：

```sh
MAC_DIR=code/VoxtNew-sign scripts/mac.sh scripts/mac-sign-e2e.sh
```

ad hoc 构建里 voxtnew-llm 和 llama.framework **不带** hardened runtime：hardened runtime 的库验证要求库和进程是同一个 Team 签的，ad hoc 签名没有 Team，llama.framework 会被拒绝加载（M2 在 mac-air 上见过）。真实身份签名时它们都带 hardened runtime。

### 2.3 发布脚本（actnow-publish.sh）

`scripts/actnow-publish.sh` 是官网脚本 `https://actnow.dev/actnow-publish.sh` 的**原样副本**，不在发布时下载最新的：发布用的代码在仓库里、经过审阅，而且 mac-air 连官网要走代理。10-17 对过一次：官网的版本只在注释里多了 `voxtnew`（`--app` 的取值），已同步。官网改了脚本时同步：

```sh
curl -fsSL https://actnow.dev/actnow-publish.sh -o scripts/actnow-publish.sh
git diff scripts/actnow-publish.sh        # 看改了什么，再提交
```

### 2.4 以后在 GitHub Actions 上发布（自托管的 Mac runner）

现在不用（§9）。以后在仓库的自托管 Apple 芯片 runner 上发布时，照 `release.sh` 写一个根目录的工作流（例如 `.github/workflows/voxtnew-release.yaml`，tag `voxtnew-v*` 触发）。Secret 只有这几个，**不要**在 sglang-omni 的 fork 上 `gh secret set`，等用户决定用哪个仓库：

| Secret | 内容 |
|---|---|
| `ACTNOW_INGEST_TOKEN` | `RELEASE_INGEST_TOKEN_VOXTNEW` 的值（§1.4） |
| `MYGO_UPDATER_PRIVATE_KEY` | `mygo-update.key` 的内容（§1.3） |
| `APPLE_TEAM_ID` | `779M533DL5` |
| Developer ID 证书 | 779M533DL5 的 .p12 和它的密码（Auralink 的 CI 也用这个 Team 的）：自托管 runner 上也可以直接用机器的钥匙串 |
| 公证凭据 | york@actnow.dev 的 App 专用密码（或 779M533DL5 的 App Store Connect API key） |

```yaml
      - name: Build, sign, notarize, check (no -upload)
        working-directory: VoxtNew
        env:
          MYGO_UPDATER_PRIVATE_KEY: ${{ secrets.MYGO_UPDATER_PRIVATE_KEY }}
        run: |
          export VOXTNEW_SIGN_IDENTITY="$(scripts/release-identity.sh 779M533DL5 | cut -f1)"
          scripts/release.sh --runtimes "$HOME/voxtnew-runtime" --notarize voxtnew-notary
      - name: Publish to actnow.dev
        if: github.event_name == 'push'          # a voxtnew-v* tag
        working-directory: VoxtNew
        env:
          ACTNOW_INGEST_TOKEN: ${{ secrets.ACTNOW_INGEST_TOKEN }}
        run: |
          bash scripts/actnow-publish.sh --app voxtnew --dir build/darwin-arm64 \
            --changelog CHANGELOG.md --min-macos 15.0
```

runner 以图形会话里的用户运行（LaunchAgent），钥匙串就是解锁的；以 LaunchDaemon 运行时要先 `security unlock-keychain`。

## 3. 签名细节

- **谁签什么**：`build-helper.sh` 和 `build-llm.sh` 用 `VOXTNEW_SIGN_IDENTITY`（没有就 ad hoc）签各自的产物，供 app 外单独使用；`build-omni.sh` 总是 ad hoc。打包时 `mygo build` 用真实身份把 `Contents/Resources` 里**每个** Mach-O 文件和 framework 重新签一遍（由内向外，hardened runtime + 时间戳；`codesign --deep` 碰不到 Resources，所以 MyGo 自己遍历：补丁之外的 `signNestedCode`），再签整个 app。`macos.helperEntitlements` 给辅助进程和 voxtnew-llm 指定 entitlement，其余代码保留签名时的 entitlement（都没有）。
- **ad hoc** 时 MyGo 只给没有有效签名的代码补签，已有签名的保持原样，指定了 entitlement 的（辅助进程、voxtnew-llm）重签。
- **mygo.json 要求 voxtnew-llm 在**：`helperEntitlements` 里有 `llm/voxtnew-llm`，没有它 `go tool mygo build` 和 `mygo dev` 都会失败（“the resources hold no executable … at llm/voxtnew-llm”）。开发时先 `scripts/build-llm.sh`，或者在 mac-air 上用现成的：`cp -Rc ~/voxtnew-runtime/llm resources/darwin-arm64/llm`。更新的 e2e 测试（§6）不带运行时，自己去掉这一项。
- **不开沙箱**：app 要启动子进程（辅助进程、qwen3_asr_server、voxtnew-llm）。
- **公证**由 MyGo 做：`macos.notarize = {keychainProfile}`（`release-config.sh --notary-profile`）时，`mygo build` 把签好的 dmg 交给 `notarytool submit --wait`，通过后 staple dmg 和 app（更新包里的 app 是 staple 过的）。被拒时看 `xcrun notarytool log <id> --keychain-profile voxtnew-notary`。

## 4. 从 ad-hoc 换到 Developer ID：授权（TCC）

macOS 按 bundle ID + 代码签名的要求记住麦克风和辅助功能授权。ad-hoc 签名的要求就是那次构建的哈希，Developer ID 的是“Apple 签发、Team ID 为 779M533DL5、标识为 dev.hehehai.voxtnew”。所以：

- **第一次从自己编的 ad-hoc 版本换到发布版**，旧授权对不上：麦克风变成“拒绝且不再弹窗”，辅助功能开关看着开着但不起作用。换之前执行一次，再打开发布版重新授权：
  ```sh
  tccutil reset Microphone dev.hehehai.voxtnew
  tccutil reset Accessibility dev.hehehai.voxtnew
  ```
- **之后的更新**（同一个 Team ID、同一个 bundle ID）授权保留，beta 和正式版之间也一样。`identifier` 发布后就不能再改。
- 开发时用 Apple Development 证书签名的构建，与发布版之间切换也要 reset 一次。
- **换 Team 也要 reset 一次**：10-19 起从 RY2TQTYSKQ 换成 779M533DL5（§1.1）。用户在 mac-air 上装着的版本（37a3063d，10-17 23:12 装的）是 RY2TQTYSKQ 签的，装上第一个 779M533DL5 签的版本后，麦克风和辅助功能的授权都对不上。还没有发布过版本（0.1.0-beta.1 没发出去），第一个公开版本就是 779M533DL5 签的，别的用户不受影响。
- 下载好的模型（`$HUB`，默认 `~/.cache/huggingface/hub`）不受签名影响，换版本、更新都保留。

### 4.1 给用户装新构建（mac-air，换 Team 那一次）

1. 退出 VoxtNew（菜单栏 → 退出），确认 `pgrep -fl /Applications/VoxtNew.app/` 没有输出。
2. 只在第一次装 779M533DL5 签的构建时：
   ```sh
   tccutil reset Microphone dev.hehehai.voxtnew
   tccutil reset Accessibility dev.hehehai.voxtnew
   ```
3. 换掉 `/Applications/VoxtNew.app`（只留一份这个 bundle ID 的 app），打开它，按提示重新允许麦克风，在 系统设置 → 隐私与安全性 → 辅助功能 里重新打开 VoxtNew：用户要动手授权一次。
4. 检查：`codesign -dvv /Applications/VoxtNew.app 2>&1 | grep -E 'Authority=Developer|TeamIdentifier'` 显示 wei guan (779M533DL5)；录一段、粘贴一次。

之后同一个 Team 的构建和更新都保留授权，不用再做第 2 步。

## 5. 更新怎么工作

### 5.1 检查和安装，以及“装好但还没重启”

- 插件每天在后台检查一次官网的更新源（启动后 10 秒起），有新版本时弹出更新窗口（不抢正在打字的 App 的焦点），显示更新说明，提供 **安装更新 / 稍后提醒 / 跳过这个版本**。窗口语言跟随 VoxtNew 的界面语言。
- 手动入口：应用菜单、菜单栏图标的菜单、⌘K、设置 → 诊断与关于 → 更新。
- 安装时先下载增量，校验签名，用已安装的 app 拼出新版本、逐个文件核对 SHA-256；任何一步不对就改下载完整包。app 里有约 260 MB 的运行时，增量只含变了的文件：只改 Go 代码的版本，增量只有几 MB。
- 装好后插件提示重启（`App.Relaunch`：先走 VoxtNew 的退出流程，停掉辅助进程和模型进程，再启动新版本）。
- **“自动安装更新”默认关**。开着时后台装好的更新不会自动重启；supervisor 发现磁盘上的版本变了，就不再启动磁盘上的新辅助进程（`helper.FailUpdated`），侧边栏出现“已安装更新 / 重启 VoxtNew 以完成更新”。

### 5.2 增量在两个通道之间

补丁 0006 的 `updates.prereleases`（[release/mygo.release.json](../release/mygo.release.json) 里打开）让构建同时读**两个通道的清单**：先放两个通道各自的版本，再按版本号从高到低放它们清单里列的旧版本，去重、只取比当前低的，最多 3 个。增量文件和这个版本的更新包放在同一个目录，清单里是完整的 URL，所以正式版的清单复制到 beta 通道以后照样指向 `voxtnew/` 下的文件。两个通道最新版本的用户总能拿到增量；更旧的、或者改过 app 的，下完整包。

## 6. 不用证书测试更新

两个脚本在 Mac 上用一把**测试**密钥和 ad-hoc 签名把整条更新链跑一遍：按发布配置构建（`url` 换成测试的），每个构建先用 `actnow-publish.sh --dry-run` 检查，再由 [scripts/update-server](../scripts/update-server/main.go) `-publish` 像官网后台那样摆进一个代表 R2 桶的目录，`-site` 在本机回环地址上用 HTTPS 模拟 actnow.dev 的 `/updates/voxtnew[/beta]/`。不开窗口、不抢焦点、不启动辅助进程、不碰钥匙串；构建里不带模型运行时（更新器不关心 bundle 里是什么）。

```sh
MAC_DIR=code/VoxtNew-upd scripts/mac.sh scripts/mac-update-e2e.sh   # 增量、完整包、增量失败后改完整包、已最新
MAC_DIR=code/VoxtNew-upd scripts/mac.sh scripts/mac-beta-e2e.sh     # 两个通道：0.1.0-beta.1 … 0.1.1，Beta 更新开 / 关 / 默认
```

## 7. 回滚、密钥和证书出问题时

- **发布后发现问题**：在官网面板下架这个版本，或者对上一个版本 Roll back。已经装了新版本的不会降级；修好后发一个**更高**的版本。急需回退的人可以从官网下载旧版本的 dmg 覆盖安装（同一 Developer ID，授权和模型都保留）。
- **更新私钥丢失**：已安装的 VoxtNew 再也收不到自动更新，只能换公钥发新版本，请所有用户手动安装一次。（密钥是 VoxtNew 自己的，Auralink 不受影响。）
- **更新私钥泄漏**：立即用旧私钥签发只换了公钥的新版本，官网面板里 voxtnew 的公钥也要换。
- **ingest token 泄漏**：请官网换掉 VoxtNew 的 token；文件仍要过后台的签名校验，推送不了恶意更新。
- **Developer ID 证书**：779M533DL5 现在的那张 2027-02-01 到期（§1.1），过期后带时间戳签的旧版本仍能运行；换新证书（同一 Team）后授权保留。换 Team 则授权全部失效（§4）。

## 8. 只能用真证书、真官网验证的

1. 公证通过；`release-verify.sh developer-id` 全部通过。如果公证因为 `Contents/Resources` 里的可执行文件被拒，看 `notarytool log`（MyGo 支持这种布局，预期能过）。
2. hardened runtime 下：Omni 服务和 voxtnew-llm 从 app 里启动、加载模型、出结果（10-17 用旧 Team RY2TQTYSKQ 签的构建做过 `--smoke`，见实测记录；779M533DL5 签的还要再做一次；还要按 [manual-test.md](manual-test.md) 各点一遍）。MLX 只用 Metal，不需要 JIT 一类的 entitlement；万一某个运行时被系统拒绝，`log show --predicate 'subsystem == "com.apple.securityd"' --last 5m` 里能看到原因。
3. reset TCC 后授权麦克风和辅助功能，录音、粘贴、读浏览器网址都正常。
4. 第一次真实发布：后台校验通过、两个通道的清单对、dmg 能从下载页下载、更新日志显示中英文说明。
5. 发下一个版本后：“安装更新 → 重启”，授权和下载好的模型都还在；增量和完整包各走一次。

## 9. 为什么这样做

- **在 Mac 上用脚本发布，不用 GitHub Actions**：发布要编 Omni 运行时（CMake、固定版本的 MLX）和 voxtnew-llm（xcodebuild、Metal 着色器），GitHub 的 macOS runner 上每次要 15 分钟以上，还要把 Developer ID、公证凭据、更新私钥和官网 token 放进 sglang-omni 这个公开仓库的 Secret。发布用的 Mac 上这些都在钥匙串和 `~/.config` 里。CI（`voxtnew-ci.yaml`）只做 Linux 上的测试。以后要在 CI 里发布，见 §2.4。
- **`updates` 不写进仓库的 mygo.json**，只在签名的发布构建里由 [scripts/release-config.sh](../scripts/release-config.sh) 合并：任何带 `updates` 的 `go tool mygo build`（包括 mac-air 上各线程的测试构建）都会链接更新源，放在可写目录里就会每天去检查，发布后甚至会用正式版替换掉测试构建。
- **自己的更新密钥**：家族规范要求每个 app 一对（M4 时曾和 Auralink 共用）；密钥出事只影响 VoxtNew（§7）。
- **签名 Team 是 779M533DL5，和家族其他 app 一样**：用户 10-19 决定（之前按家族规范 §17 Q1 用 RY2TQTYSKQ / Xiaofeng Cao，不再用）。结果：VoxtNew 可以加入家族第三期的钥匙串访问组、App Group 和共享 engine（家族规范 §8.5、§11.3），这些都要求同一个 Team；第一、二期不受影响。代价只有一次：用户已装的 RY2TQTYSKQ 版本换过来时要重新授权（§4）。公证也简单了：york@actnow.dev 就是这个 Team 的账号，不用再加入别人的团队。
- **ssh 里经 `mac-gui.sh` 签名**：不需要密码、不改钥匙串的设置（分区列表），只借用已经登录的图形会话。
- **两个更新源，beta 取较新的**：官网的约定，和 Auralink 一样；打开 Beta 更新时 app 读 beta 清单，失败时退回正式版。beta 的地址不单独配置：`updates.url` 是正式版的，MyGo（补丁 0006 的 `prereleases`）按版本号把 beta 放进它的 `beta/`，所以 `release/mygo.release.json` 只有一个 url。
- **按 SHA-1 选签名身份**：codesign 遇到两张同名证书会报 “ambiguous”，按哈希签不会。

## 实测记录

mac-air（M4 / 24 GB，macOS 26.7），10-16，运行时用 `~/voxtnew-runtime`（`--runtimes`）：

- **ad hoc**（`scripts/release.sh --runtimes ~/voxtnew-runtime`，约 3 分钟，大半是辅助进程的 `swift build`）：`VoxtNew.app` 280 MB，dmg 45 MB；`release-verify.sh ad-hoc` 全部通过：8 段代码（app、辅助进程、qwen3_asr_server、libmlx、libjaccl、voxtnew-llm、llama.framework、MyGo 的 libmygo-sqlite3）都签好，Omni 服务和 voxtnew-llm 能从 app 里启动。
- **签名**（`scripts/mac-sign-e2e.sh`：临时自签的代码签名身份代替 Developer ID）：8 段代码都带 hardened runtime 和 Apple 的时间戳，`codesign --verify --deep --strict` 通过；Gatekeeper 拒绝（自签，预期）。Omni 服务和 voxtnew-llm 从 app 里**启动不了**：`libmlx.dylib … not valid for use in process: mapping process and mapped file (non-platform) have different Team IDs`。这是库验证在比较 Team ID，自签证书没有 Team；Developer ID 签的进程和库是同一个 Team，能加载：10-17 用当时的 Team RY2TQTYSKQ 确认了（下面）。
- **启动**（10-17，合并后的 ad hoc 发布式构建）：`VoxtNew.app/Contents/MacOS/VoxtNew --smoke <dir>`（自己的数据目录，链接快速语音模型、VAD 和快速语言模型）：辅助进程就绪，asr.fast 0.8 s 回答，llm.fast 6.8 s（冷启动，机器上同时有其他线程的模型进程）回答“OK”，退出后没有留下 `qwen3_asr_server`、`voxtnew-llm`、`voxtnew-helper`。`open -g` 正常启动：窗口“VoxtNew”出现、辅助进程就绪；`quit` 后同样一个进程都不剩。
- **Developer ID**（10-16）：在 ssh 里签不了（`errSecInternalComponent`，钥匙串锁着）；10-17 起经 `scripts/mac-gui.sh` 在图形会话里签（§1.1）。
- **公证**：mac-air 上没有 notarytool 的配置（§1.2，10-17 再查过），没有提交过。
- **RY2TQTYSKQ 签名**（10-17，用的是旧 Team，10-19 起改用 779M533DL5，还没用它签过；`scripts/mac-gui.sh scripts/release.sh --runtimes ~/voxtnew-runtime`，身份 `DE35317B…`，更新私钥从 `~/.config/voxtnew-release/` 读）：约 2 分钟；app 280 MB，dmg 45 MB，更新包 87 MB；`release-verify.sh signed` 全部通过：8 段代码都带 hardened runtime、Apple 的时间戳、`TeamIdentifier=RY2TQTYSKQ`，**Omni 服务和 voxtnew-llm 从 app 里启动了**（库验证按 Team 比对，同一 Team 能加载，M4 自签身份时的 “different Team IDs” 没有了）。`codesign -dvv` 显示 `Authority=Developer ID Application: Xiaofeng Cao (RY2TQTYSKQ)`、`TeamIdentifier=RY2TQTYSKQ`、`flags=0x10000(runtime)`。Gatekeeper 拒绝（Unnotarized Developer ID，预期）。`--smoke`（真实的快速语音模型、VAD、快速语言模型）：辅助进程就绪，asr.fast 0.72 s、llm.fast 1.9 s 回答 “OK”，退出后没有留下进程。清单 `url` 是 `https://actnow.dev/updates/voxtnew/beta/voxtnew-0.1.0-beta.1-darwin-arm64.tar.gz`；`actnow-publish.sh --dry-run` 通过（beta，3 个文件，notes 和 notesZh 分开），没有发送。增量那一步没有代理时超时（只是警告：还没有已发布的版本）。同一次在 ssh 里直接跑：一开始的试签报 `errSecInternalComponent`，立刻停下。
- **与 Auralink 同时运行**（10-17，`scripts/mac-hotkey-coexist.sh`，Auralink 在跑、用 fn）：见 [manual-test.md](manual-test.md) §2.14。
