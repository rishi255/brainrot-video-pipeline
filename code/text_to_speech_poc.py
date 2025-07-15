# POC for TTS library
# Finally works!

# time the tts_to_file function
import time

from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

import torch

print("is available?", torch.cuda.is_available())

start_time = time.monotonic()

# Generate audio for Stewie
stewie = tts.tts_to_file(
    text="Ah there it is, we did it Brian, we made 9/11 happen, HIGH FIVE! \n Wow. That probably wouldn’t look good out of context.",
    file_path="outputs/stewie_output.mp3",
    speaker_wav=["assets/audio/stewie_30sec.mp3"],
    language="en",
    split_sentences=True,
)

stewie_end_time = time.monotonic()
print("Stewie output file path:", stewie)
print("Stewie generation time:", stewie_end_time - start_time, "seconds")

# Generate audio for Peter
peter = tts.tts_to_file(
    text="Hey Lois, I just got a new job at the airport. I’m going to be a pilot! \n I’m going to fly all over the world and see all kinds of amazing things! A boat’s a boat but the mystery box could be anything. It could even be a boat! You know how much we’ve wanted one of those",
    file_path="outputs/peter_output.mp3",
    speaker_wav=["assets/audio/peter_10_minutes.mp3"],
    language="en",
    split_sentences=True,
)
final_end_time = time.monotonic()

print("Peter output file path:", peter)
print("Peter generation time:", final_end_time - stewie_end_time, "seconds")

print("Total generation time:", final_end_time - start_time, "seconds")
