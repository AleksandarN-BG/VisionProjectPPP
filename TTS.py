import os
import threading
import time

from gtts import gTTS


class TTSManager:
    def __init__(self):
        self.tts_lock = threading.Lock()
        self.is_busy = False
        self.current_thread = None
        self.queue = []
        self.queue_lock = threading.Lock()
        self.running = True  # Set this BEFORE starting the thread
        self.queue_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.queue_thread.start()

    def _process_queue(self):
        """Worker thread that processes the audio queue"""
        while self.running:
            task = None
            with self.queue_lock:
                if self.queue:
                    task = self.queue.pop(0)

            if task:
                task_type, content = task
                if task_type == "tts":
                    self._speak(content)
                elif task_type == "file":
                    self._play_sound_file(content)
            else:
                time.sleep(0.1)  # Sleep when queue is empty

    def stop(self):
        """Stop any ongoing speech and clear queue"""
        with self.queue_lock:
            self.queue = []
        try:
            if self.current_thread and self.current_thread.is_alive():
                self.is_busy = False  # Signal the thread to terminate early
        except Exception as e:
            print(f"Error stopping TTS: {e}")

    def _speak(self, text):
        """Internal method to speak text (called by queue processor)"""
        try:
            self.is_busy = True
            with self.tts_lock:
                print(f"[TTS] Speaking: '{text}'")

                # Create a unique filename in project directory
                import uuid
                temp_filename = f"speech_{uuid.uuid4().hex[:8]}.mp3"
                temp_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), temp_filename)

                # Generate speech file
                tts = gTTS(text=text, lang='en')
                tts.save(temp_file)

                # Play using pygame
                try:
                    import pygame
                    pygame.mixer.init()
                    pygame.mixer.music.load(temp_file)
                    pygame.mixer.music.play()

                    # Wait for playback to complete
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)

                    pygame.mixer.quit()
                except Exception as e:
                    print(f"Error playing audio: {e}")

                # Clean up after playback
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception as e:
                    print(f"Error removing temp file: {e}")

        except Exception as e:
            print(f"TTS error: {e}")
        finally:
            self.is_busy = False

    def speak(self, text, block=False):
        """Add TTS speech to the queue"""
        if block:
            self._speak(text)
        else:
            with self.queue_lock:
                self.queue.append(("tts", text))
        return True

    def _play_sound_file(self, sound_file):
        """Internal method to play sound file (called by queue processor)"""
        try:
            if not os.path.exists(sound_file):
                print(f"Sound file not found: {sound_file}")
                return False

            self.is_busy = True
            try:
                import pygame

                # Only initialize once and don't quit between plays
                if not hasattr(pygame.mixer, 'init') or not pygame.mixer.get_init():
                    pygame.mixer.init()
                    print(f"Initialized pygame mixer for: {sound_file}")

                # Print debug info
                print(f"Playing sound file: {sound_file}")

                # Load and play sound
                sound = pygame.mixer.Sound(sound_file)
                channel = sound.play()

                # Wait for playback to complete
                while channel.get_busy():
                    time.sleep(0.1)

            except Exception as e:
                print(f"Error playing sound file: {sound_file}")
                print(f"Error details: {e}")

                # Try alternative with playsound as fallback
                try:
                    from playsound import playsound
                    playsound(sound_file, block=True)
                except Exception as e2:
                    print(f"Fallback playsound also failed: {e2}")

            finally:
                self.is_busy = False

        except Exception as e:
            print(f"Error setting up sound playback: {e}")
            return False

    def play_file(self, sound_file):
        """Add sound file to the queue"""
        with self.queue_lock:
            self.queue.append(("file", sound_file))
        return True
