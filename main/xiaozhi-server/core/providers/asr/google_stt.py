import os
import subprocess
import tempfile
from typing import List, Optional, Tuple

from google.cloud import speech

from config.logger import setup_logging
from core.providers.asr.base import ASRProviderBase
from core.providers.asr.dto.dto import InterfaceType

TAG = __name__
logger = setup_logging()


class ASRProvider(ASRProviderBase):
    def __init__(self, config: dict, delete_audio_file: bool):
        super().__init__()
        logger.bind(tag=TAG).info("Initializing Google ASR Provider")
        self.interface_type = InterfaceType.NON_STREAM
        self.delete_audio_file = delete_audio_file
        self.output_dir = config.get("output_dir", "/tmp")
        self.language_code = config.get("language_code", "vi-VN")
        self.client = speech.SpeechClient()

    async def speech_to_text(
        self, opus_data: List[bytes], session_id: str, audio_format="opus"
    ) -> Tuple[Optional[str], Optional[str]]:
        file_path = None
        if audio_format == "pcm":
            pcm_data = opus_data
        else:
            pcm_data = self.decode_opus(opus_data)

        combined_data = b"".join(pcm_data)
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pcm", dir=self.output_dir)
        tmp_file.write(combined_data)
        tmp_file.flush()
        file_path = tmp_file.name
        tmp_file.close()

        logger.bind(tag=TAG).info(f"Temporary audio file created: {file_path}")
        file_wav_path = file_path.replace(".pcm", ".wav")
        logger.bind(tag=TAG).info(f"Temporary audio file converted to WAV: {file_wav_path}")

        try:
            subprocess.run([
                'ffmpeg', '-f', 's16le', '-ar', '16000', '-ac', '1',
                '-i', file_path, file_wav_path
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.bind(tag=TAG).error(f"FFmpeg conversion error: {e.stderr.decode()}")
            return "", (None if self.delete_audio_file else file_path)

        try:
            with open(file_wav_path, "rb") as audio_file:
                content = audio_file.read()
            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code=self.language_code,
            )
            response = self.client.recognize(config=config, audio=audio)
            result = ""
            for res in response.results:
                result += res.alternatives[0].transcript.strip() + " "
            result = result.strip()
            logger.bind(tag=TAG).info(f"Transcription result: {result}")
            return result, (None if self.delete_audio_file else file_path)
        except Exception as e:
            logger.bind(tag=TAG).error(f"Transcription error: {e}", exc_info=True)
            return "", (None if self.delete_audio_file else file_path)
        finally:
            if self.delete_audio_file and os.path.exists(file_path):
                os.remove(file_path)
            if self.delete_audio_file and os.path.exists(file_wav_path):
                os.remove(file_wav_path)

   