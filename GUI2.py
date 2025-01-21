# -*- coding: utf-8 -*-
import importlib.util
import subprocess
import sys
import os
# 定义需要的库
required_packages = ['pyaudio', 'wave', 'pydub', 'funasr', 'numpy', 'emoji', "torch", "torchaudio", "PyQt5"]

# 检查并安装缺失的库
for package in required_packages:
    if importlib.util.find_spec(package) is None:
        print(f"{package} 未安装，正在安装...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--index-url", "https://mirrors.aliyun.com/pypi/simple/"])
    else:
        # print(f"{package} 已安装")
        pass
print("所有依赖库均已安装")
# 导入需要的库
import threading
from datetime import datetime
import time
import pyaudio
import wave
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import numpy as np
import emoji
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QTextEdit,
    QMessageBox
)
from PyQt5.QtCore import QTimer

# 设置ffmpeg路径（如果需要）
ffmpeg_path = r"C:\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"
os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)

# 配置模型
model_dir = r"C:\Users\admin\.cache\modelscope\hub\iic\SenseVoiceSmall"
vad_model_dir = r"C:\Users\admin\.cache\modelscope\hub\iic\speech_fsmn_vad_zh-cn-16k-common-pytorch"
model = AutoModel(
    model=model_dir,
    vad_model=vad_model_dir,
    trust_remote_code=False,
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
    disable_update=True
)

# 音频录制参数
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5  # 每次识别的音频片段时长（秒）

# 定义表情符号映射表
emoji_map = {
    "😊": "(高兴)",
    "😡": "(愤怒/兴奋)",
    "😔": "(悲伤)",
    "😀": "(笑声)",
    "🎼": "(音乐)",
    "👏": "(掌声)",
    "🤧": "(咳嗽和打喷嚏)",
    "😭": "(哭泣)"
}

all_audio_dir = []

class AudioRecorder:
    def __init__(self, input_device_index):
        self.input_device_index = input_device_index
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.recording = False
        self.frames = []

    def start_recording(self):
        self.recording = True
        self.frames = []
        self.stream = self.audio.open(
            format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK,
            input_device_index=self.input_device_index
        )
        print("开始录制...")

    def stop_recording(self):
        self.recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
            print("录制结束...")

    def record_chunk(self):
        if self.recording:
            data = self.stream.read(CHUNK)
            self.frames.append(data)
            return data
        return None

    def save_audio(self, audio_dir):
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        audio_file_name = f"recorded_audio_{timestamp}.wav"
        wave_file = os.path.join(audio_dir, audio_file_name)

        wf = wave.open(wave_file, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        all_audio_dir.append(str(wave_file))
        return wave_file

def recognize_audio(audio_data):
    try:
        # 将音频数据保存为临时文件
        temp_audio_file = "temp_audio.wav"
        wf = wave.open(temp_audio_file, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(RATE)
        wf.writeframes(audio_data)
        wf.close()

        res = model.generate(
            input=temp_audio_file,
            cache={},
            language="auto",
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )
        if res and isinstance(res, list) and len(res) > 0:
            full_text = ' '.join([segment["text"] for segment in res])
            full_text = rich_transcription_postprocess(full_text)

            # 替换指定的表情符号
            for emoji_char, description in emoji_map.items():
                full_text = full_text.replace(emoji_char, description)

            # 去除其他表情符号
            full_text = ''.join(char for char in full_text if not emoji.is_emoji(char))

            return full_text
        else:
            return ""
    except Exception as e:
        print(f"音频识别错误: {e}")
        return ""

class AudioApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.input_device_index = 1  # 选择正确的设备索引
        self.audio_dir = None
        self.recorder = AudioRecorder(self.input_device_index)
        self.recording_thread = None
        self.recording = False
        self.text_area.append("secret_dumplings出品\n欢迎使用\n鸣谢：modelscope社区@通义实验室提供的SenseVoice模型")

    def initUI(self):
        self.setWindowTitle("音频录制与实时识别")
        self.setGeometry(100, 100, 600, 400)

        layout = QVBoxLayout()

        self.label = QLabel("音频录制与实时识别工具")
        layout.addWidget(self.label)

        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        layout.addWidget(self.text_area)

        self.record_button = QPushButton("开始录制")
        self.record_button.clicked.connect(self.toggle_recording)
        layout.addWidget(self.record_button)
        self.audio_dir = None  # 如果创建失败，将 audio_dir 设置为 None
        self.setLayout(layout)

    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
            # 合并音频文件
            self.concatenate_audios(os.listdir(self.audio_dir), os.path.join(self.audio_dir, "recorded_audio.wav"))

    def start_recording(self):
        if self.audio_dir is None:
            # 自动创建目录
            now = datetime.now()
            self.audio_dir = now.strftime("./output/AUDIO_DIR_%Y%m%d_%H%M%S")
            try:
                os.makedirs(r"./output", exist_ok=True)
                os.makedirs(self.audio_dir, exist_ok=True)
                self.text_area.append(f"音频保存目录已创建: {self.audio_dir}")
            except Exception as e:
                self.text_area.append(f"创建音频保存目录失败: {e}")

        self.recording = True
        self.record_button.setText("停止录制")
        self.text_area.append("正在录制音频...")
        self.recording_thread = threading.Thread(target=self.record_audio_thread)
        self.recording_thread.start()

    def stop_recording(self):
        self.recording = False
        self.record_button.setText("开始录制")
        self.text_area.append("录制结束")
        self.recorder.stop_recording()

    def record_audio_thread(self):
        print(f"音频保存目录: {self.audio_dir}")
        if self.audio_dir is None:
            print("音频保存目录未初始化，请检查程序逻辑！")
            return

        self.recorder.start_recording()
        while self.recording:
            for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
                self.recorder.record_chunk()
            audio_data = b''.join(self.recorder.frames)
            text = recognize_audio(audio_data)
            if text:
                self.text_area.append(f"实时识别结果: {text}")
                # 保存音频到文件
                self.recorder.save_audio(self.audio_dir)
                # 保存识别结果到文件
                text_file_path = os.path.join(all_audio_dir, "recognized_text.txt")
                with open(text_file_path, "a", encoding="utf-8-sig", errors="replace") as f:
                    f.write(f"{text}\n")
            self.recorder.frames = []  # 清空已处理的音频帧
            time.sleep(0.1)  # 控制识别频率

    def concatenate_audios(self, file_list, output_file):
        """合并多个音频文件到一个文件"""
        from pydub import AudioSegment
        combined = AudioSegment.empty()
        for file in file_list:
            try:
                audio = AudioSegment.from_wav(file)
                combined += audio
            except Exception as e:
                pass
                # print(f"无法处理文件 {file}: {e}")
        try:
            combined.export(output_file, format="wav")
            print("合并完成。")
        except Exception as e:
            print(f"导出错误: {e}")

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = AudioApp()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"程序启动失败，错误信息: {e}")
