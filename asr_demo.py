import os
import time
import argparse # 导入 argparse 模块

# 假设这些是您的模块
from asr import ASRModel
from audio import RecAudioVadThread

def main(args):
    """
    主函数，处理语音识别逻辑
    """
    # 根据传入的参数初始化 ASRModel 和 RecAudioVadThread
    # ASRModel 看起来没有直接的参数，所以这里保持不变
    asr_model = ASRModel()

    # RecAudioVadThread 的参数化
    rec_audio = RecAudioVadThread(
        sld=args.sld,
        max_time=args.max_recording_time,
        channels=args.channels,
        rate=args.sample_rate,
        device_index=args.device_index,
        trig_on=args.trigger_on,
        trig_off=args.trigger_off
    )

    try:
        while True:
            print("Press enter to start!")
            input() # enter 触发

            # 开始录制用户声音
            rec_audio.max_time_record = args.max_recording_time # 确保这里也使用参数
            rec_audio.start_recording()

            rec_audio.stop_recording() # 等待录音完成
            audio_ret = rec_audio.get_audio() # 获取录音数据或路径

            # 检查 audio_ret 是否有效
            if audio_ret is None:
                print("No audio recorded or detected. Please try again.")
                continue # 跳过本次循环，重新等待输入

            text = asr_model.generate(audio_ret)
            print('user: ', text)

    except KeyboardInterrupt:
        print("Process was interrupted by user.")
    finally:
        # 确保在程序退出时清理资源，如果 RecAudioVadThread 有 close 或 cleanup 方法的话
        if hasattr(rec_audio, 'close'):
            rec_audio.close()
        elif hasattr(rec_audio, 'cleanup'):
            rec_audio.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ASR Demo with Voice Activity Detection (VAD)")

    # 添加 RecAudioVadThread 相关的参数
    parser.add_argument('--sld', type=int, default=1,
                        help='Speech segment length in seconds for VAD (default: 1)')
    parser.add_argument('--max_recording_time', type=int, default=5,
                        help='Maximum recording time in seconds (default: 5)')
    parser.add_argument('--channels', type=int, default=1,
                        help='Number of audio channels (default: 1)')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Audio sample rate in Hz (default: 16000)')
    parser.add_argument('--device_index', type=int, default=3,
                        help='Audio input device index (default: 3)')
    parser.add_argument('--trigger_on', type=float, default=0.60,
                        help='VAD trigger-on threshold (0.0-1.0, default: 0.60)')
    parser.add_argument('--trigger_off', type=float, default=0.35,
                        help='VAD trigger-off threshold (0.0-1.0, default: 0.35)')

    args = parser.parse_args()
    main(args)
