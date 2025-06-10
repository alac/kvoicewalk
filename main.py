import datetime

from utilities.kvoicewalk import KVoiceWalk
import argparse
import soundfile as sf
from utilities.speech_generator import SpeechGenerator
import os
import numpy as np

from utilities.pytorch_sanitizer import load_multiple_voices
from utilities.transcriber import transcribe


def main():
    parser = argparse.ArgumentParser(description="A random walk Kokoro voice cloner.")

    # Common required arguments
    parser.add_argument("--target_text", type=str, help="The words contained in the target audio file. Should be around 100-200 tokens (two sentences).")

    # Optional arguments
    parser.add_argument("--other_text", type=str,
                      help="A segment of text used to compare self similarity. Should be around 100-200 tokens.",
                      default="If you mix vinegar, baking soda, and a bit of dish soap in a tall cylinder, the resulting eruption is both a visual and tactile delight, often used in classrooms to simulate volcanic activity on a miniature scale.")
    parser.add_argument("--voice_folder", type=str,
                      help="Path to the voices you want to use as part of the random walk.",
                      default="./voices")
    parser.add_argument("--interpolate_start",
                      help="Goes through an interpolation search step before random walking",
                      action='store_true')
    parser.add_argument("--population_limit", type=int,
                      help="Limits the amount of voices used as part of the random walk",
                      default=10)
    parser.add_argument("--step_limit", type=int,
                      help="Limits the amount of steps in the random walk",
                      default=10000)
    parser.add_argument("--output", type=str,
                      help="Filename for the generated output audio",
                      default="out.wav")

    # Arguments for random walk mode
    group_walk = parser.add_argument_group('Random Walk Mode')
    group_walk.add_argument("--target_audio", type=str,
                          help="Path to the target audio file. Must be 24000 Hz mono wav file.")
    group_walk.add_argument("--starting_voice", type=str,
                          help="Path to the starting voice tensor")

    # Arguments for test mode
    group_test = parser.add_argument_group('Test Mode')
    group_test.add_argument("--test_voice", type=str,
                          help="Path to the voice tensor you want to test")

    # Arguments for util mode
    group_util = parser.add_argument_group('Utility Mode')
    # TODO: Add ffmpeg file prep
    group_util.add_argument("--export_bin",
                      help='Exports target voices in the --voice_folder directory',
                      action='store_true')
    group_util.add_argument("--transcribe",
                      help='Uses Faster-Whisper to quickly transcribe your target audio. Replaces --target_text')
    args = parser.parse_args()

    # Export Utility
    if args.export_bin:
        if not args.voice_folder:
            parser.error("--voice_folder is required to export a voices bin file")

        # Collect all .pt file paths
        file_paths = [os.path.join(args.voice_folder, f) for f in os.listdir(args.voice_folder) if f.endswith('.pt')]
        voices = load_multiple_voices(file_paths, auto_allow_unsafe=False) # Set True if you prefer to bypass Allow/Repair/Reject voice file menu

        with open("voices.bin", "wb") as f:
            np.savez(f,**voices)

        return

    if args.transcribe:
        args.target_text = transcribe(args.output)
        return

    # Validate arguments based on mode
    if args.test_voice:
        # Test mode
        if not args.target_text:
            parser.error("--target_text is required when using --test_voice")

        speech_generator = SpeechGenerator()
        audio = speech_generator.generate_audio(args.target_text, args.test_voice)
        sf.write(args.output, audio, 24000)
    else:
        # Random walk mode
        if not args.target_audio:
            parser.error("--target_audio is required for random walk mode")
        if not args.target_text:
            parser.error("--target_text is required for random walk mode")

        ktb = KVoiceWalk(args.target_audio,
                        args.target_text,
                        args.other_text,
                        args.voice_folder,
                        args.interpolate_start,
                        args.population_limit,
                        args.starting_voice)
        ktb.random_walk(args.step_limit)

if __name__ == "__main__":
    main()
