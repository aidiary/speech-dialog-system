import queue
import sys

import pyaudio
from google.cloud import speech

RATE = 16000
CHUNK = int(RATE / 10)
LANGUAGE_CODE = "ja-JP"


class MicrophoneStream:
    """録音ストリームを開き、オーディオチャンクを生成するジェネレータを返す"""

    def __init__(self, rate: int = RATE, chunk: int = CHUNK) -> None:
        self._rate = rate
        self._chunk = chunk

        # オーディオデータのスレッドセーフなバッファを作成
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            # オーディオストリームを非同期で実行してバッファオブジェクトを埋める
            # 呼び出しスレッドがネットワークリクエストを行う間に入力デバイスの
            # バッファがオーバーフローしないようにするために必要
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type: object, value: object, traceback: object) -> None:
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(
        self, in_data: object, frame_count: int, time_info: object, status_flags: object
    ) -> object:
        """オーディオストリームからデータを連続的に収集し、バッファに格納"""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self) -> object:
        """オーディオデータのストリームからオーディオチャンクを生成"""
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)


if __name__ == "__main__":
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=LANGUAGE_CODE,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in audio_generator
        )

        responses = client.streaming_recognize(streaming_config, requests)
        print(responses)

        num_chars_printed = 0
        for response in responses:
            if not response.results:
                continue

            result = response.results[0]
            if not result.alternatives:
                continue

            transcript = result.alternatives[0].transcript

            # 認識の途中結果を表示
            overwrite_chars = " " * (num_chars_printed - len(transcript))

            if not result.is_final:
                sys.stdout.write(transcript + overwrite_chars + "\r")
                sys.stdout.flush()

                num_chars_printed = len(transcript)
            else:  # final result
                print("***", transcript + overwrite_chars)
