import contextlib
import wave

import librosa
import librosa.display
import matplotlib.pyplot as plt
import webrtcvad


def read_wave(path):
    with contextlib.closing(wave.open(path, "rb")) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())

        return pcm_data, sample_rate


class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset : offset + n], timestamp, duration)
        timestamp += duration
        offset += n


if __name__ == "__main__":
    filepath = "../data/BASIC5000_0113.wav"

    audio, sample_rate = read_wave(filepath)
    vad = webrtcvad.Vad(3)

    plt.figure(figsize=(16, 4))
    y, sr = librosa.load(filepath, sr=16000)
    plt.plot(y)

    # フレームは10ミリ秒、20ミリ秒、30ミリ秒のいずれかでなければならない
    frames = frame_generator(30, audio, sample_rate)
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        print(frame.timestamp, frame.duration, is_speech)
        if is_speech:
            plt.plot(int(frame.timestamp * sample_rate), 0, "ro")
    plt.show()
