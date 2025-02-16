# 音频录制与识别工具
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


项目简介
本项目包含两个Python脚本，分别用于音频录制、实时语音识别以及音频文件处理。项目使用了`SenseVoiceSmall`模型进行语音识别，并结合`pyaudio`、`pydub`等库实现音频录制和处理功能。


功能介绍

• 音频录制：

• 实时录制音频并保存为WAV格式文件。

• 支持指定录音设备和保存路径。

• 自动增益处理，提升音频音量。


• 语音识别：

• 使用`SenseVoiceSmall`模型进行实时语音识别。

• 支持自动语言检测和文本后处理。

• 替换表情符号为文本描述，去除其他无关表情符号。


• 音频文件处理：

• 合并多个音频文件为一个文件。

• 保存识别结果到文本文件。


• 图形界面（`GUI2.py`）：

• 提供基于`PyQt5`的图形用户界面，方便用户操作。


项目结构

• `GUI2.py`：包含音频录制、实时识别和图形用户界面的完整实现。

• `录音识别转换.py`：包含音频录制、识别和文件处理的命令行版本。


使用方法

环境依赖

• Python 3.6+

• 以下Python库（脚本会自动检查并安装缺失的库）：

• `pyaudio`

• `wave`

• `pydub`

• `funasr`

• `numpy`

• `emoji`

• `torch`

• `torchaudio`

• `PyQt5`（仅`GUI2.py`需要）


运行

• 确保已安装所有依赖库。

• 根据需要选择运行以下脚本：

• 图形界面版本：

```bash
     python GUI2.py
```


• 命令行版本：

```bash
     python 录音识别转换.py
```



致谢
本项目使用了以下开源项目和资源，特此感谢：

• ModelScope社区：感谢通义实验室提供的`SenseVoiceSmall`模型，为本项目提供了强大的语音识别能力。

• PyQt5：为图形界面版本提供了友好的用户交互支持。

• PyAudio：用于音频录制功能的实现。

• Pydub：用于音频文件的处理和增益调整。

• FFmpeg：用于音频文件的格式转换和处理。

• emoji：用于处理和替换文本中的表情符号。

特别感谢ModelScope社区的贡献和支持，使得本项目能够高效地实现语音识别功能。


注意事项

• 请确保已正确安装并配置`ffmpeg`路径。

• 根据实际需求调整音频录制参数（如采样率、录制时长等）。

• 命令行版本可通过手动中断（如`Ctrl+C`）结束程序运行。


联系方式
如有任何问题或建议，请随时联系项目维护者。

许可证
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

