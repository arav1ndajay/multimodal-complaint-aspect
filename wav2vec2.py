import soundfile as sf
import audiofile
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

class GenerateAudio():

  def __init__(self):
    self.tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")    
  
  def get_audio_features(self, input_file):
 
    # read audio file
    # speech, samplerate = sf.read(input_file)
    speech, samplerate = audiofile.read(
          input_file,
          offset= 0,
          duration= 10,
          always_2d=True,
    )

    print(speech.shape)
    print(speech)

    # make it 1-D
    if len(speech.shape) > 1: 
        speech = speech[0,:] + speech[1,:]
    
    print(speech.shape)

    # resample to 16khz
    print("resampling...")
    if samplerate != 16000:
        speech = librosa.resample(speech, samplerate, 16000)

    print(speech)
    print("tokenizing...")
    # tokenize
    input_values = self.tokenizer(speech, return_tensors="pt").input_values
    
    print("running model...")
    # run the model
    logits = self.model(input_values).logits

    print(logits)
    print(logits.shape)

    # #take argmax (find most probable word id)
    # predicted_ids = torch.argmax(logits, dim=-1)
    # #get the words from the predicted word ids
    # transcription = tokenizer.decode(predicted_ids[0])
    # #output is all uppercase, make only the first letter in first word capitalized
    # transcription = correct_uppercase_sentence(transcription.lower())
    return

input_file = "downloads/tmp_0.mp4"
gau = GenerateAudio()
gau.get_audio_features(input_file)

