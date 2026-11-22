# 把录音转成字幕文件

[English](en/file_to_srt.md)

手头有一段会议或讲座的录音，想要一份带时间戳的字幕文件，或者一份纯文字稿。Voxt 自带的命令行可以做到，用的是**转录**选用的同一个语音识别模型。

1. 让 Voxt 保持运行，确认**转录**页选好了语音识别模型。
2. 打开终端，运行：

   ```sh
   VOXT=/Applications/Voxt.app/Contents/MacOS/Voxt
   $VOXT transcribe talk.m4a -o talk.srt
   ```

3. 得到的 `talk.srt` 是带开始和结束时间的字幕文件。要纯文字、不要时间戳时，加 `--format txt`；录音不是系统主语言时，加 `--lang en` 一类的语言提示。

Voxt 按停顿把录音切成最长七秒左右的片段，没听到内容的片段会略过。常见格式（m4a、mp3、wav、mp4、mov 等）直接支持；mkv、webm 需要先用 ffmpeg 转成 m4a。完整的参数、退出码和隐私说明见[命令行转录文档](https://github.com/sgl-project/sglang-omni/blob/main/Voxt/docs/transcribe.md)。

## 交给 Agent

> 按本页确认 Voxt 在运行且转录已选好语音识别模型。对我指定的录音文件运行命令，输出到我指定的位置。把生成的文件和命令的退出结果告诉我。
