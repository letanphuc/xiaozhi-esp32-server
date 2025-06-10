import os
import subprocess
import time

from google.cloud import texttospeech

from config.logger import setup_logging
from core.providers.tts.base import TTSProviderBase
from core.utils.util import check_model_key

TAG = __name__
logger = setup_logging()


class TTSProvider(TTSProviderBase):
    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)
        logger.bind(tag=TAG).info("Initializing Google TTS Provider")
        self.language_code = config.get("language_code", "en-US")
        self.voice_name = config.get("voice", None)  # Optional: specific Google voice name
        self.ssml_gender = config.get("ssml_gender", "NEUTRAL").upper()
        self.response_format = config.get("format", "mp3").upper()  # MP3, LINEAR16, OGG_OPUS, etc.
        self.audio_file_type = self.response_format.lower()
        self.output_dir = config.get("output_dir", "tmp/")
        self.client = texttospeech.TextToSpeechClient()

    async def text_to_speak(self, text, output_file):
        logger.bind(tag=TAG).info(f"output path {output_file}, text: {text}")
        synthesis_input = texttospeech.SynthesisInput(text=text)

        # Set up voice params
        gender = getattr(texttospeech.SsmlVoiceGender, self.ssml_gender, texttospeech.SsmlVoiceGender.NEUTRAL)
        voice_params = {
            "language_code": self.language_code,
            "ssml_gender": gender
        }
        voice = texttospeech.VoiceSelectionParams(**voice_params)

        # Set up audio config
        encoding = getattr(texttospeech.AudioEncoding, self.response_format, texttospeech.AudioEncoding.MP3)
        audio_config = texttospeech.AudioConfig(audio_encoding=encoding)

        response = self.client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        if output_file:
            with open(output_file, "wb") as f:
                f.write(response.audio_content)
        else:
            return response.audio_content
