# Trigger-Word-Detection
Trigger word detection is the technology that allows devices like Amazon Alexa, Google Home, Apple Siri, and Baidu DuerOS to wake up upon hearing a certain word.

Our trigger word will be "Activate." Every time it hears you say "activate," it will make a "chiming" sound.

## Data synthesis: Creating a speech dataset
- A speech dataset should ideally be as close as possible to the application you will want to run it on.
- In this case, we'd like to detect the word "activate" in working environments (library, home, offices, open-spaces ...).

In the raw_data directory, subset of the raw audio files of the positive words, negative words, and background noise can be found. We will use these audio files to synthesize a dataset to train the model.
- The "activate" directory contains positive examples of people saying the word "activate".
- The "negatives" directory contains negative examples of people saying random words other than "activate".
- There is one word per audio recording.
- The "backgrounds" directory contains 10 second clips of background noise in different environments.
## From audio recordings to spectrograms
Audio recording is a long list of numbers measuring the little air pressure changes detected by the microphone.

We will use audio sampled at 44100 Hz (or 44100 Hertz).
- This means the microphone gives us 44,100 numbers per second.
- Thus, a 10 second audio clip is represented by 441,000 numbers (=10×44100).
### Spectrogram
A spectrogram is computed by sliding a window over the raw audio signal, and calculating the most active frequencies in each window using a Fourier transform.

- It is quite difficult to figure out from this "raw" representation of audio whether the word "activate" was said.
- In order to help your sequence model more easily learn to detect trigger words, we will compute a spectrogram of the audio.
- The spectrogram tells us how much different frequencies are present in an audio clip at any moment in time.
Eg:
<p align = 'center'>
  <img src = '/images/train_reference.png'>
</p>
The graph above represents how active each frequency is (y axis) over a number of time-steps (x axis).

- The color in the spectrogram shows the degree to which different frequencies are present (loud) in the audio at different points in time.
- Green means a certain frequency is more active or more present in the audio clip (louder).
- Blue denote less active frequencies.
- The dimension of the output spectrogram depends upon the hyperparameters of the spectrogram software and the length of the input.
- We will be working with 10 second audio clips as the "standard length" for our training examples.
  - The number of timesteps of the spectrogram will be 5511.
  - You'll see later that the spectrogram will be the input x into the network, and so T<sub>x</sub>=5511.
- The output of our model will divide 10 seconds into 1,375 units.
  - T<sub>y</sub>=1375.
  - For each of the 1375 time steps, the model predicts whether someone recently finished saying the trigger word "activate."

