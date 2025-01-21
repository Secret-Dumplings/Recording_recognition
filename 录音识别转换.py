# -*- coding: utf-8 -*-
import sys
import subprocess
import importlib.util

# 定义需要的库
required_packages = ['pyaudio', 'wave', 'pydub', 'funasr', 'numpy', 'emoji', "torch", "torchaudio"]

# 检查并安装缺失的库
for package in required_packages:
    if importlib.util.find_spec(package) is None:
        print(f"{package} 未安装，正在安装...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--index-url", "https://mirrors.aliyun.com/pypi/simple/"])
    else:
        print(f"{package} 已安装")

# 导入所需库
import pyaudio
import wave
import pydub
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import os
from datetime import datetime
import numpy as np
import emoji

# 设置ffmpeg路径
ffmpeg_path = r"C:\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"  # 替换为实际路径
os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)
pydub.AudioSegment.ffmpeg = ffmpeg_path

# 配置模型
model_dir = "iic/SenseVoiceSmall"
model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
)

# 音频录制参数
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 10  # 每次录制的时间（秒）

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

def record_audio(input_device_index, audio_dir):
    """录制音频并保存到唯一文件名"""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK,
                        input_device_index=input_device_index)
    print("开始录制...")
    frames = []

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("录制结束...")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # 生成唯一文件名
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    audio_file_name = f"recorded_audio_{timestamp}.wav"
    wave_file = os.path.join(audio_dir, audio_file_name)

    # 保存音频文件
    wf = wave.open(wave_file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    # 应用增益提高音量
    audio_segment = pydub.AudioSegment.from_wav(wave_file)
    audio_segment = audio_segment.apply_gain(30)  # 增加30分贝
    audio_segment.export(wave_file, format="wav")

    # 重新读取增益处理后的音频数据
    audio_segment_processed = pydub.AudioSegment.from_wav(wave_file)
    audio_data_processed = audio_segment_processed.raw_data
    audio_array_processed = np.frombuffer(audio_data_processed, dtype=np.int16)

    # 检查是否有NaN值
    if np.isnan(audio_array_processed).any():
        print("音频数据包含NaN值，进行处理...")
        audio_array_processed = np.nan_to_num(audio_array_processed)

    # 检查音频是否太安静
    if np.max(np.abs(audio_array_processed)) < 10:
        print("Recording is too quiet. Skipping.")
        return None

    rms = np.sqrt(np.mean(audio_array_processed ** 2))
    print(f"RMS value after gain: {rms}")
    return wave_file

def recognize_audio(audio_file):
    try:
        res = model.generate(
            input=audio_file,
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

def concatenate_audios(file_list, output_file):
    """合并多个音频文件到一个文件"""
    from pydub import AudioSegment
    combined = AudioSegment.empty()
    for file in file_list:
        try:
            audio = AudioSegment.from_wav(file)
            combined += audio
        except Exception as e:
            print(f"无法处理文件 {file}: {e}")
    try:
        combined.export(output_file, format="wav")
        print("合并完成。")
    except Exception as e:
        print(f"导出错误: {e}")

def main():
    recorded_files = []
    input_device_index = 1  # 选择正确的设备索引

    # 生成基于时间戳的目录名
    now = datetime.now()
    timestamp_dir = now.strftime("AUDIO_DIR_%Y%m%d_%H%M%S")
    os.makedirs(timestamp_dir, exist_ok=True)

    try:
        while True:
            print("准备录制音频...")
            audio_file = record_audio(input_device_index, timestamp_dir)
            if audio_file:
                print(f"录制成功，文件路径: {audio_file}")
                recorded_files.append(audio_file)
                print("开始识别音频...")
                text = recognize_audio(audio_file)
                print(f"识别结果: {text}")

                # 保存识别结果到文件，指定encoding="utf-8-sig"
                text_file_path = os.path.join(timestamp_dir, "recognized_text.txt")
                with open(text_file_path, "a", encoding="utf-8-sig", errors="replace") as f:
                    f.write(f"{text}\n")
            else:
                print("录音太安静，跳过...")
    except KeyboardInterrupt:
        print("程序已手动中断，正在退出...")
    except Exception as e:
        print(f"发生错误: {e}")

    # 录制结束后，合并所有录音文件
    if recorded_files:
        print(f"Recorded files: {recorded_files}")
        output_file = os.path.join(timestamp_dir, "final_audio.wav")
        concatenate_audios(recorded_files, output_file)
        print(f"所有录音已合并到 {output_file}")

if __name__ == "__main__":
    main()