import pyaudio
import tempfile
import wave
import time
import threading
import numpy as np
from scipy.signal import resample
from collections import deque

class RecAudioDB:
    def __init__(self, sld=1, min_db=2000, max_time=5, channels=1, rate=48000, device_index=0):
        """
        Args:
            vad_mode: vad 的模式
            sld: 静音多少 s 停止录音
            max_time: 最多录音多少秒
            channels: 声道数
            rate: 采样率
            device_index: 输入的设备索引
        """
        self._sld = sld
        self.max_time_record = max_time
        self.frame_is_append = True
        self.time_start = time.time()

        self.MIN_DB = min_db

        # 参数配置
        self.FORMAT = pyaudio.paInt16  # 16-bit 位深
        self.CHANNELS = channels              # 单声道
        self.RATE = rate              # 16kHz 采样率
        FRAME_DURATION = 30       # 每帧时长（ms）
        # self.FRAME_SIZE = int(self.RATE * FRAME_DURATION / 3000)  # 每帧采样数
        self.FRAME_SIZE = 1024

        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.FRAME_SIZE,
            input_device_index=device_index
        )

        self.exit_mode = 0

    def vad_audio(self):
        self.frame_is_append = False
        frames = []
        speech_detected = False
        last_speech_time = time.time()
        self.time_start = time.time()

        pre_speech_frames = deque(maxlen=10)  # 缓冲最近10帧，大约300ms

        try:
            while True:
                try:
                    frame = self.stream.read(self.FRAME_SIZE, exception_on_overflow=False)
                except Exception as e:
                    print(f"音频读取失败: {e}")
                    break

                pre_speech_frames.append(frame)  # 每帧都先存入缓冲区
                audio_data = np.frombuffer(frame, dtype=np.short)
                current_max = np.max(audio_data)
                # print('current_max: ', current_max)

                if current_max > self.MIN_DB:
                    last_speech_time = time.time()
                    if not speech_detected:
                        self.frame_is_append = True
                        speech_detected = True
                        print("检测到语音，开始录制...")
                        frames.extend(pre_speech_frames)  # 把前面几帧也补进去

                if self.frame_is_append:
                    frames.append(frame)

                current_time = time.time()
                if (speech_detected and
                    current_time - last_speech_time > self._sld):
                    print(f"静音超过 {self._sld} 秒，停止录制。")
                    self.exit_mode = 0
                    break

                if  (current_time - self.time_start) >= self.max_time_record:
                    print(f"录音时间超过 {self.max_time_record} 秒，停止录制。")
                    self.exit_mode = 1
                    break


        except Exception as e:
            print(e)
            print("手动中断录制。")

        finally:
            print("关闭音频流")
            self.stream.stop_stream()

            if len(frames) > 0:
                # 转换为 numpy ndarray（int16 类型）
                audio_data = b"".join(frames)
                audio_np = np.frombuffer(audio_data, dtype=np.int16)

                if self.RATE != 16000:
                    num_samples = int(len(audio_np) * float(16000) / self.RATE)
                    waveform = resample(audio_np, num=num_samples)
                    return waveform
                else:
                    return audio_np
            else:
                return None

    def record_audio(self):
        self.stream.start_stream()
        audio_ndarray = self.vad_audio()
        return audio_ndarray


class RecAudioDBThread(RecAudioDB):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.thread = None
        self.is_recording = False
        self.audio_ndarray = None  # 录音文件路径

    def start_recording(self):
        """启动录音线程"""
        if self.thread is None or not self.thread.is_alive():
            self.is_recording = True
            self.thread = threading.Thread(target=self._record_audio_thread)
            self.thread.start()

    def _record_audio_thread(self):
        """录音线程执行的方法"""
        self.audio_ndarray = self.record_audio()
        self.is_recording = False

    def stop_recording(self):
        """停止录音"""
        self.is_recording = False
        if self.thread and self.thread.is_alive():
            self.thread.join()

    def get_audio(self):
        """获取录音后的文件路径"""
        return self.audio_ndarray


# ---------------- 测试 ---------------- #
if __name__ == "__main__":
    rec = RecAudioDB(sld=1, max_time=5, rate=48000, device_index=1)
    print("按 Enter 开始录音 ...")
    input()
    wav = rec.record_audio()
    print("录音完成，采样点数:", None if wav is None else wav.shape[0])