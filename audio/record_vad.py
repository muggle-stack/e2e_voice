import time
import os
import threading
import numpy as np
from collections import deque
from scipy.signal import resample
import subprocess
from queue import Queue
from threading import Lock

import pyaudio            # 录音
import onnxruntime as ort  # VAD

model_url = "https://archive.spacemit.com/spacemit-ai/openwebui/sensevoice.tar.gz"
cache_dir = os.path.expanduser("~/.cache")
vad_model_dir = os.path.join(cache_dir, "sensevoice")
vad_model_path = os.path.join(vad_model_dir, "silero_vad.onnx")
tar_path = os.path.join(cache_dir, "sensevoice.tar.gz")

def resample_audio(frame_bytes, original_rate, target_rate=16000):
    # 先转为 int16 PCM
    audio = np.frombuffer(frame_bytes, dtype=np.int16)
    # 计算新采样点数
    new_len = int(len(audio) * target_rate / original_rate)
    # 重采样
    resampled = resample(audio, new_len)
    # 再转回 bytes（float32->int16）
    resampled_int16 = np.clip(resampled, -1.0, 1.0) * 32767
    return resampled_int16.astype(np.int16).tobytes()

# -------- Silero VAD 封装（无 torch） -------- #
WIN_SAMPLES  = 512             # 32 ms
CTX_SAMPLES  = 64
RATE_VAD = 16000

class SileroVAD:
    """NumPy 封装，输入 bytes(512*2) -> 概率 float"""
    def __init__(self, record_rate=16000):
        if not os.path.exists(vad_model_path):
            print("模型文件不存在，正在下载模型文件")
            subprocess.run(["wget", "-O", tar_path, model_url], check=True)
            subprocess.run(["tar", "-xvzf", tar_path, "-C", cache_dir], check=True)
            subprocess.run(["rm", "-rf", tar_path], check=True)
            print("Models Download successfully")

        self._model_path = vad_model_path
        self.sess  = ort.InferenceSession(self._model_path, providers=['CPUExecutionProvider'])
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.ctx   = np.zeros((1, CTX_SAMPLES), dtype=np.float32)
        self.sr    = np.array(RATE_VAD, dtype=np.int64)

        self.record_rate = record_rate

    def reset(self):
        self.state.fill(0)
        self.ctx.fill(0)

    def __call__(self, pcm_bytes: bytes) -> float:

        if self.record_rate != 16000:
            pcm_bytes = resample_audio(pcm_bytes, original_rate=self.record_rate, target_rate=16000)

        wav = (np.frombuffer(pcm_bytes, dtype=np.int16)
                 .astype(np.float32) / 32768.0)[np.newaxis, :]      # (1,512)

        x = np.concatenate((self.ctx, wav), axis=1)                # (1,576)
        self.ctx = x[:, -CTX_SAMPLES:]

        prob, self.state = self.sess.run(
            None,
            {"input": x.astype(np.float32),
             "state": self.state,
             "sr":    self.sr}
        )
        return float(prob)

# ============ 录音管线，改用 SileroVAD ============ #
class RecAudioVad:
    def __init__(self, sld=1, max_time=5, channels=1, rate=16000, device_index=0, trig_on=0.60, trig_off=0.35):
        """
        Args:
            sld: 静音多少 s 停止录音
            max_time: 最多录音多少秒
            channels: 声道数
            rate: 采样率
            device_index: 输入的设备索引
            TRIG_ON: Vad触发阈值
            TRIG_OFF: Vad结束阈值
        """
        self._sld            = sld
        self.max_time_record = max_time
        self.trig_on = trig_on
        self.trig_off = trig_off
        self.frame_is_append = True

        # ---- 录音参数固定为 16k / 512samples 与 VAD 对齐 ---- #
        self.RATE       = rate # Silero v5 固定
        self.FRAME_SIZE = WIN_SAMPLES
        self.FORMAT     = pyaudio.paInt16
        self.CHANNELS   = channels

        self.pa     = pyaudio.PyAudio()
        self.stream = self.pa.open(format=self.FORMAT,
                                   channels=self.CHANNELS,
                                   rate=self.RATE,
                                   input=True,
                                   frames_per_buffer=self.FRAME_SIZE,
                                   input_device_index=device_index)

        self.vad = SileroVAD(record_rate=self.RATE)
        self._frame_q = Queue(maxsize=20)
        self._vad_lock  = Lock()
        self.prob_avg   = 0.0
        self._vad_thread = threading.Thread(target=self._vad_worker,
                                            daemon=True)
        self._vad_thread.start()
        self.hist = deque(maxlen=10)   # 300 ms 平滑阈值
        self.time_start = 0

        self.exit_mode = 0

    def _vad_worker(self):
        """不断从 _frame_q 取帧，算概率放回 _prob_q"""
        while True:
            frame = self._frame_q.get()
            p  = self.vad(frame)
            self.hist.append(p)
            with self._vad_lock:
                self.prob_avg = float(np.mean(self.hist))

    # ------------- 录音 + VAD ---------------- #
    def vad_audio(self):
        frames = []
        speech_detected = False
        self.frame_is_append = False
        last_speech_time = time.time()
        self.time_start  = time.time()

        pre_speech_frames = deque(maxlen=10)

        try:
            while True:
                frame = self.stream.read(self.FRAME_SIZE, exception_on_overflow=False)

                # --- Silero 推理 ---
                self._frame_q.put(frame)
                # print(f'conf: {p}')
                with self._vad_lock:
                    prob_avg = self.prob_avg
                # t0 = time.perf_counter()          # ① 记录开始等待时间
                # self._vad_lock.acquire()          # ② 阻塞拿锁
                # wait = time.perf_counter() - t0   # ③ 计算耗时

                # prob_avg = self.prob_avg          # 临界区
                # self._vad_lock.release()          # ④ 释放锁
                # print(f"[lock wait] {wait*1e6:.1f} µs")   # 打印等待时间（单位: 微秒）
                pre_speech_frames.append(frame)
                is_speech = prob_avg > self.trig_on if not speech_detected else prob_avg > self.trig_off

                if is_speech:
                    last_speech_time = time.time()
                    if not speech_detected:
                        self.frame_is_append = True
                        speech_detected = True
                        print("▶ 检测到语音，开始录制...")
                        frames.extend(pre_speech_frames)
                        continue

                if self.frame_is_append:
                    frames.append(frame)

                # ----- 停止条件 -----
                now = time.time()
                if (speech_detected and
                    now - last_speech_time > self._sld):
                    print(f"⏹ 静音超过 {self._sld}s，停止录制")
                    self.exit_mode = 0
                    break

                if (now - self.time_start) >= self.max_time_record:
                    print(f"⏹ 录音超过 {self.max_time_record}s，停止录制")
                    self.exit_mode = 1
                    break

        finally:
            self.stream.stop_stream()

        # --------- 返回 numpy int16 波形 16 k --------- #
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
            print("音频数组为空！！！")
            return None

    def record_audio(self):
        self.stream.start_stream()
        return self.vad_audio()


class RecAudioVadThread(RecAudioVad):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.thread = None
        self.audio_np = None      # numpy int16

    def start_recording(self):
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self._record_audio_thread, daemon=True)
            self.thread.start()

    def _record_audio_thread(self):
        self.audio_np = self.record_audio()

    def stop_recording(self):
        if self.thread and self.thread.is_alive():
            self.thread.join()

    def get_audio(self):
        return self.audio_np

# ---------------- 测试 ---------------- #
if __name__ == "__main__":
    rec = RecAudioVad(sld=1, max_time=5)
    print("按 Enter 开始录音 ...")
    input()

    wav = rec.record_audio()
    print("录音完成，采样点数:", None if wav is None else wav.shape[0])
