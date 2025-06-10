from openai import OpenAI
from core.utils.util import check_model_key
from core.providers.tts.base import TTSProviderBase
from config.logger import setup_logging

TAG = __name__
logger = setup_logging()


class TTSProvider(TTSProviderBase):
    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)
        self.api_key = config.get("api_key")
        self.model = config.get("model", "tts-1")
        self.voice = config.get("private_voice") or config.get("voice", "alloy")
        self.response_format = config.get("format", "wav")
        self.audio_file_type = self.response_format

        speed = config.get("speed", "1.0")
        self.speed = float(speed) if speed else 1.0

        self.output_dir = config.get("output_dir", "tmp/")
        check_model_key("TTS", self.api_key)

        self.client = OpenAI(api_key=self.api_key)

    async def text_to_speak(self, text, output_file):
        logger.bind(tag=TAG).info(f"output path {output_file}, text: {text}")

        response = self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text,
            response_format=self.response_format,
            speed=self.speed,
        )

        if output_file:
            with open(output_file, "wb") as f:
                f.write(response.content)
        else:
            return response.content
