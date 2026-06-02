import asyncio
import re
import traceback

# Try to import sounddevice natively
try:
    import sounddevice as sd
    import numpy as np
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("WARNING: sounddevice or numpy is not installed. Audio playback will be disabled.")

# Try to import kokoro_onnx
try:
    from kokoro_onnx import Kokoro
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    print("WARNING: kokoro_onnx is not installed. VoiceSpeaker will fallback to silent mode.")


class VoiceSpeaker:
    def __init__(self, model_path: str = "kokoro-v1.0.onnx", voices_path: str = "voices-v1.0.bin"):
        """
        Initializes the VoiceSpeaker engine with zero-disk I/O Kokoro ONNX model.

        """
        self.model_path = model_path
        self.voices_path = voices_path
        self.kokoro = None
        self.sample_rate = 24000  # Default for Kokoro v0.19

        self._initialize_model()

    def _initialize_model(self):
        """loads the model into memory at boot time."""
        if not KOKORO_AVAILABLE or not AUDIO_AVAILABLE:
            return

        try:
            print("[VoiceSpeaker] Initializing Kokoro TTS engine into memory...")
            # Load the model into memory once at boot time
            self.kokoro = Kokoro(self.model_path, self.voices_path)
            print("[VoiceSpeaker] Kokoro TTS Engine initialized successfully.")
        except Exception as e:
            print(f"[WARNING] VoiceSpeaker failed to initialize Kokoro model: {e}")
            print("[WARNING] Audio generation will be silently skipped to prevent crashing.")
            self.kokoro = None

    def _sanitize_text(self, text: str) -> str:
        """
        Dynamic Token Sanitization:
        Strips out raw formatting notation like backticks, clean markdown punctuation, 
        and edge-case characters to prevent reading aloud code syntax.
        """
        # Remove markdown code blocks (```...```)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        # Remove inline code backticks (`...`)
        text = re.sub(r'`[^`]*`', '', text)
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        # Remove asterisks, underscores, tildes, and hashes
        text = re.sub(r'[*_~#]', '', text)
        # Replace newlines with spaces for a smoother continuous read
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _generate_audio_blocking(self, text: str, voice: str, speed: float = 1.0) -> 'Any':
        """
        Heavy ONNX matrix math generation pass (Blocking).
        This must be run in an executor to avoid freezing the main thread.
        """
        if not self.kokoro:
            return None

        try:
            # Kokoro create() generates text-to-speech entirely in memory
            # Returns: audio_array (float32 numpy array), sample_rate (int)
            samples, sample_rate = self.kokoro.create(
                text, voice=voice, speed=speed, lang="en-us"
            )
            self.sample_rate = sample_rate
            return samples
        except Exception as e:
            print(f"[VoiceSpeaker] Error during ONNX generation: {e}")
            traceback.print_exc()
            return None

    def _play_audio_blocking(self, audio_array: 'Any'):
        """
        Plays back raw numpy arrays natively using sounddevice (Blocking).
        """
        if audio_array is None or len(audio_array) == 0:
            return
        
        if not AUDIO_AVAILABLE:
            return

        try:
            # Hand off the generated float32 or int16 numpy array natively to the sound hardware driver
            # This directly communicates with the host OS audio output (e.g. WASAPI, CoreAudio, ALSA)
            sd.play(audio_array, samplerate=self.sample_rate)
            
            # Wait for the playback to finish to ensure non-overlapping speech 
            # and to block the offloaded thread appropriately until speaking is done.
            sd.wait()
        except Exception as e:
            print(f"[VoiceSpeaker] Error during sounddevice playback: {e}")

    async def say(self, text: str, voice: str = "am_echo", speed: float = 1.0) -> None:
        """
        Non-Blocking Async Design:
        Offloads generation to background thread, then streams to hardware natively.
        """
        if not text:
            return

        sanitized_text = self._sanitize_text(text)
        if not sanitized_text:
            return

        loop = asyncio.get_running_loop()

        # Thread-Safe Offloading: 
        # Offload the heavy ONNX matrix math generation pass to a background thread pool executor
        audio_array = await loop.run_in_executor(
            None, self._generate_audio_blocking, sanitized_text, voice, speed
        )

        if audio_array is not None:
            # Play the generated audio array via sounddevice without blocking the async event loop
            await loop.run_in_executor(None, self._play_audio_blocking, audio_array)


# Global instance for easy import across the project
speaker = VoiceSpeaker()


# -------------------------------------------------------------------------
# Legacy Adapters (Preserved to prevent breaking other modules)
# -------------------------------------------------------------------------

async def respond(text: str, play_animation: bool = True, language: str = "en"):
    """
    Unified system response coordinator (Updated to use local Kokoro TTS).
    Always prints to the terminal with beautiful styling.
    """
    # Print response text directly in blue color with no prefix
    print(f"\n\033[94m{text}\033[0m\n")
    
    # Bypass language translation networks to stay offline, route immediately to local Kokoro.
    await speaker.say(text, voice="am_echo")
        
    return {"status": "success", "message": "Response processed via local VoiceSpeaker."}


async def speak_text(text: str, play_animation: bool = True, language: str = "en"):
    """Legacy wrapper for speak_text"""
    return await respond(text, play_animation, language)