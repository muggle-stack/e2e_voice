import wave
import os
import time
import subprocess
import warnings
import numpy as np

from .models.sensevoice_bin import SenseVoiceSmall
from .models.postprocess_utils import rich_transcription_postprocess

model_url = "https://archive.spacemit.com/spacemit-ai/openwebui/sensevoice.tar.gz"
cache_dir = os.path.expanduser("~/.cache")
asr_model_dir = os.path.join(cache_dir, "sensevoice")
asr_model_path = os.path.join(asr_model_dir, "model_quant_optimized.onnx")
tar_path = os.path.join(cache_dir, "sensevoice.tar.gz")

class ASRModel:
    def __init__(self):
        if not os.path.exists(asr_model_path):
            print("模型文件不存在，正在下载模型文件")
            subprocess.run(["wget", "-O", tar_path, model_url], check=True)
            subprocess.run(["tar", "-xvzf", tar_path, "-C", cache_dir], check=True)
            subprocess.run(["rm", "-rf", tar_path], check=True)
            print("Models Download successfully")

        self._model_path = asr_model_dir
        self._model = SenseVoiceSmall(self._model_path, batch_size=10, quantize=True)

    def generate(self, audio_file, sr=16000):
        if isinstance(audio_file, np.ndarray):
            audio_path = audio_file
            audio_dur = len(audio_file) / sr
        elif isinstance(audio_file, str):
            audio_path = [audio_file]
            audio_dur = wave.open(audio_file).getnframes() / sr
        else:
            warnings.warn(
                f"[ASR] Unsupported type {type(audio_file).__name__}; "
                "expect str or np.ndarray. Skip this turn."
            )
            return None
            audio_dur = len(audio_file) / 16000

        t0 = time.perf_counter()
        try:
            asr_res = self._model(audio_path, language='zh', use_itn=True)
        except Exception as e:
            warnings.warn(f"[ASR] Inference error: {e}. Skip this turn.")
            return None
        infer_time = time.perf_counter() - t0
        rtf = infer_time / audio_dur if audio_dur > 0 else float("inf")
        print(f"infer_time: {infer_time:.3f}s, audio_dur: {audio_dur:.3f}s, RTF: {rtf:.2f}")
        # 后处理
        asr_res = asr_res[0][0].tolist()
        text = rich_transcription_postprocess(asr_res[0])
        return text
