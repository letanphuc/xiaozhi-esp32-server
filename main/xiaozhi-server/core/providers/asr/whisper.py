import os
import tempfile
from typing import List, Optional, Tuple

from openai import OpenAI

from config.logger import setup_logging
from core.providers.asr.base import ASRProviderBase
from core.providers.asr.dto.dto import InterfaceType
import subprocess

TAG = __name__
logger = setup_logging()


class ASRProvider(ASRProviderBase):
    def __init__(self, config: dict, delete_audio_file: bool):
        super().__init__()
        logger.bind(tag=TAG).info("Initializing OpenAI ASR Provider")
        self.interface_type = InterfaceType.NON_STREAM
        self.api_key = config.get("api_key", "")
        self.model_name = config.get("model_name", "gpt-4o-mini-transcribe")
        self.delete_audio_file = delete_audio_file
        self.output_dir = config.get("output_dir", "/tmp")
        self.client = OpenAI(api_key=self.api_key)

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
        file_mp3_path = file_path.replace(".pcm", ".mp3")
        logger.bind(tag=TAG).info(f"Temporary audio file converted to MP3: {file_mp3_path}")

        try:
            subprocess.run([
                'ffmpeg', '-f', 's16le', '-ar', '16000', '-ac', '1',
                '-i', file_path, file_mp3_path
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.bind(tag=TAG).error(f"FFmpeg conversion error: {e.stderr.decode()}")
            return "", (None if self.delete_audio_file else file_path)

        try:
            with open(file_mp3_path, "rb") as f:
                transcription = self.client.audio.transcriptions.create(
                    model=self.model_name,
                    file=f,
                )
            result = transcription.text.strip()
            return result, (None if self.delete_audio_file else file_path)
        except Exception as e:
            logger.bind(tag=TAG).error(f"Transcription error: {e}", exc_info=True)
            return "", (None if self.delete_audio_file else file_path)
        finally:
            if self.delete_audio_file and os.path.exists(file_path):
                os.remove(file_path)
