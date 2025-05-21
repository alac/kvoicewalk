# KVoiceWalk
KVoiceWalk tries to create new [Kokoro](https://github.com/hexgrad/kokoro) voice style tensors that clones target voices by using a random walk algorithm and a hybrid scoring method that combines Resemblyzer similarity, feature extraction, and self similarity. This is meant to be a step towards a more advanced genetic algorithm and prove out the scoring function and general concept.

This project is only possible because of the incredible work of projects like [Kokoro](https://github.com/hexgrad/kokoro) and [Resemblyzer](https://github.com/resemble-ai/Resemblyzer). I was struck by how small the Kokoro style tensors were and wondered if it would be possible to "evolve" new voice tensors more similar to target audio. The results are promising and this scoring method could be a valid option for a future genetic algorithm. I wanted more voice options for Kokoro, and now I have them.

## Example Audio
#### Target Audio File (Generated Using A different text to speech library)

https://github.com/user-attachments/assets/ecafe828-4fee-41dd-8fe6-1f766ad19f56

#### The Most Similar Stock Trained Voice From Kokoro, af_heart.pt. Similarity score of 71%

https://github.com/user-attachments/assets/0d693c60-a9f4-43bc-bb36-409bd7391d79

#### KVoiceWalk Generated Voice Tensor After 10,000 steps. Similarity score of 93% (From Resemblyzer)

https://github.com/user-attachments/assets/b19559dd-712c-427d-8ec4-93ff26daaa92

## Usage
1. Clone this repository, change directory into it
```bash
git clone https://github.com/RobViren/kvoicewalk.git
cd kvoicewalk
```
2. Convert your target audio into the proper format (or use example). KVoiceWalk expects 24000 Hz sample rate wav file for target audio. The target audio is ideally 20-30 seconds long and is a single speaker.

```bash
ffmpeg -i input_file.wav -ar 24000 target.wav
```

3. Use [uv](https://docs.astral.sh/uv/) to run the application with arguments

```bash
uv run main.py --target_text "The old lighthouse keeper never imagined that one day he'd be guiding ships from the comfort of his living room, but with modern technology and an array of cameras, he did just that, sipping tea while the storm raged outside and gulls shrieked overhead." --target_audio ./example/target.wav
```

4. KVoiceWalk will now go through each voice in the voices folder to find the closest matches to the target file. After narrowing that down using the **population_limit** argument, it will begin to randomly guess and check voices keeping the best voice as the source for the random walk. It will log the progress and save audio and voice tensors to the **out** folder. You can then use these voice tensors in your other projects or generate some audio using the following command.

```bash
uv run main.py --test_voice /path/to/voice.pt --target_text "Your really awesome text you want spoken"
```

This will generate an audio file called out.wav using the supplied *.pt file you give it. This way you can easily test a variety of voice tensors and input text.

Play with the command line arguments and find what works for you. This is a fairly random process and processing for a long time could suddenly result in a better outcome. You can create a folder of your favorite sounding voices from the other random walks and use that as the basis for interpolation or just use that as the source for the next random walk. You can pass **starting_voice** argument to tell the system exactly what to use as a base if you want. Playing around with the options can get you a voice closer to the stlye of the target.

## Interpolated Start
KVoiceWalk has a function to interpolate around the trained voices and determine the best possible starting population of tensors to act as a guide for the random walk function to clone the target voice. Simply run the application as follows to run interpolation first. This does take awhile and having a beefy GPU will help with processing time.

```bash
uv run main.py --target_text "The works the speaker says in the audio clip" --target_audio /path/to/target.wav --interpolate_start
```

This will run an interpolation search for the best voices and put them in a folder labeled **interpolated** which you can use as the basis for a new random walk later. It will also continue a random walk afterwards.

## Example Outputs
The closest voice in the trained models for the example/target.wav was af_heart.pt with the following stats.
```
af_heart.pt          Target Sim: 0.709, Self Sim: 0.978, Feature Sim: 0.47, Score: 81.22
```
Interpolation search gave a voice that had the following stats.
```
af_jessica.pt_if_sara.pt_0.10.pt Target Sim: 0.780, Self Sim: 0.973, Feature Sim: 0.34, Score: 84.20
```
The interpolation showed a big improvement. The population of interpolated voices is then used as the basis for standard deviation mutation of a supplied voice tensor. After 10,000 steps of random walking and replacing with the best, we get this.
```
Step:9371, Target Sim:0.917, Self Sim:0.971, Feature Sim:0.54, Score:92.99, Diversity:0.01
```
An improvement of 13.7% in similarity while still maintaining model stability and voice quality.

## Design
By far the hardest thing to get right was the scoring function. Earlier attempts using Resemblyzer only resulted in overfitted garbage. Self similarity was important in keeping the model producing the same sounding input despite different inputs. Self similarity represented stability in the model and was critical in evaluation.

But even with self similarity and similarity presented by Resemblyzer it was not enough. I had to add an audio feature similarity comparison in order to prevent audio quality getting poor. What happened without this is the audio would pass similarity and self similarity checks but again sound like a metal basket of tools being thrown down stairs. The feature comparison made the difference and prevented over fitting to a random sound that apparently sounded similar to the target wav file.

The other secret sauce was the harmonic mean calculation that controls the scoring. The harmonic mean allows for some backsliding on self similarity, feature similarity, and target similarity so long as the improvement goes the right way. This made exploring the space easier for the system instead of requiring that all three only improve, which led to quick and sad stagnation. I lowered the weighting on the feature similarity. I mainly need that to prevent the voice from going completely out of bounds.

## Notes
This does not run in parallel, but does adopt early returning on bad tensors. You can run multiple instances assuming you have the GPU/CPU for it. I can run about 2 in parallel on my 3070 laptop. The results are random. You can have some that led to incredible sounding results after stagnating for a long time, and others can crash and burn right away. Totally random. This is where a future genetic algorithm would be better. But the random walk proves out the theory.

Other things you could do:
- Populate a database with results from this and train a model to predict similarity and see if you can use that to more tightly guide voice creation
- Use different methods for voice generation than my simple method, though PCA had some challenges
- Implement your own genetic algorithm and evolve voice tensors instead of random walk
