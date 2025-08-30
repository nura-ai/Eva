import os
from dotenv import load_dotenv
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

def main():
    # loading API key
    load_dotenv()
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("API key is not set. Please check your .env file.")

    # speech initialization
    client = ElevenLabs(api_key=api_key)

    # EVA's text
    text = "Hello, I am EVA. I remember you, Diana. Let's build something beautiful together."

    # voice setting 
    voice_id = "EXAVITQu4vr4xnSDxMaL"  # rachel (you can change it)
    voice_settings = VoiceSettings(
        stability=0.3,
        similarity_boost=0.8,
        style=0.5,
        use_speaker_boost=True,
        speed=1.0
    )

    try:
        # generating speech
        response = client.text_to_speech.convert(
            voice_id=voice_id,
            output_format="mp3_22050_32",
            text=text,
            model_id="eleven_turbo_v2_5",
            voice_settings=voice_settings
        )

        # saving audio
        with open("eva_voice.mp3", "wb") as f:
            for chunk in response:
                if chunk:
                    f.write(chunk)

        print("✅ EVA's voice has been generated and saved as eva_voice.mp3")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
