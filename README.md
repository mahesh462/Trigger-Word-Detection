# Trigger-Word-Detection
Trigger word detection is the technology that allows devices like Amazon Alexa, Google Home, Apple Siri, and Baidu DuerOS to wake up upon hearing a certain word.

Our trigger word will be "Activate." Every time it hears you say "activate," it will make a "chiming" sound.

## Data synthesis: Creating a speech dataset
- A speech dataset should ideally be as close as possible to the application you will want to run it on.
- In this case, we'd like to detect the word "activate" in working environments (library, home, offices, open-spaces etc).

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

## Generating a single training example
### Benefits of synthesizing data
Because speech data is hard to acquire and label, you will synthesize your training data using the audio clips of activates, negatives, and backgrounds.

- It is quite slow to record lots of 10 second audio clips with random "activates" in it.
- Instead, it is easier to record lots of positives and negative words, and record background noise separately (or download background noise from free online sources).

### Process for Synthesizing an audio clip
- To synthesize a single training example:
  - Pick a random 10 second background audio clip.
  - Randomly insert 0-4 audio clips of "activate" into this 10sec clip.
  - Randomly insert 0-2 audio clips of negative words into this 10sec clip.

### Label the positive/negative words
- The labels y<sup>⟨t⟩</sup> represent whether or not someone has just finished saying "activate."
  - y<sup>⟨t⟩</sup>=1  when that that clip has finished saying "activate".
  - Given a background clip, we can initialize  y<sup>⟨t⟩</sup>=0  for all  t , since the clip doesn't contain any "activates."
- When an "activate" clip is inserted or overlaid, update labels for y<sup>⟨t⟩</sup>.
  - Rather than updating the label of a single time step, update 50 steps of the output to have target label 1.
  - Updating several consecutive time steps can make the training data more balanced.
- We will train a GRU (Gated Recurrent Unit) to detect when someone has finished saying "activate".


Another reason for synthesizing the training data is it's relatively straightforward to generate these labels  y<sup>⟨t⟩</sup>  as described above.In contrast, if you have 10sec of audio recorded on a microphone, it's quite time consuming for a person to listen to it and mark manually exactly when "activate" finished.

### Visualizing the labels
- Here's a figure illustrating the labels  y⟨t⟩  in a clip.
  - We have inserted "activate", "innocent", activate", "baby."
  - Note that the positive labels "1" are associated only with the positive words.
<p align = 'center'>
  <img src = '/images/label_diagram.png'>
</p>

### Development set
- To test our model, we recorded a development set of 25 examples.
- While our training data is synthesized, we want to create a development set using the same distribution as the real inputs.
- Thus, we recorded 25 10-second audio clips of people saying "activate" and other random words, and labeled them by hand.
- This follows the principle - dev set has to be as similar as possible as to the test set distribution.

## Model
Our goal is to build a network that will ingest a spectrogram and output a signal when it detects the trigger word. This network will use 4 layers:
* One convolutional layer
* Two GRU layers
* A dense layer
<p align = 'center'>
  <img src = '/images/model.png'>
</p>

### 1D convolutional layer
One key layer of this model is the 1D convolutional step (near the bottom of above figure).
- It inputs the 5511 step spectrogram. Each step is a vector of 101 units.
- It outputs a 1375 step output.
- This output is further processed by multiple layers to get the final  T<sub>y</sub>=1375  step output.
- This 1D convolutional layer plays a role of extracting low-level features and then possibly generating an output of a smaller dimension.
- Computationally, the 1-D conv layer also helps speed up the model because now the GRU can process only 1375 timesteps rather than 5511 timesteps.
### GRU, dense and sigmoid
- The two GRU layers read the sequence of inputs from left to right.
- A dense plus sigmoid layer makes a prediction for  y<sup>⟨t⟩</sup>.
-  Because  y  is a binary value (0 or 1), we use a sigmoid output at the last layer to estimate the chance of the output being 1, corresponding to the user having just said "activate.".
### Unidirectional RNN
- Note that we use a unidirectional RNN rather than a bidirectional RNN.
- This is really important for trigger word detection, since we want to be able to detect the trigger word almost immediately after it is said.
- If we used a bidirectional RNN, we would have to wait for the whole 10sec of audio to be recorded before we could tell if "activate" was said in the first second of the audio clip.

