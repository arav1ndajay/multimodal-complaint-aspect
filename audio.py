import opensmile
import audiofile
import os
import pandas as pd

class GenerateAudio():
  def __init__(self):
  
    self.smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
    )
    
    self.audio_df: pd.DataFrame = None
    self.audio_file_path = "audio_embeddings.csv"

    if os.path.exists(self.audio_file_path):
      self.audio_df = pd.read_csv(self.audio_file_path, index_col=False)

  def generateAudioEmbeddings(self, file_path: str, timestamps: list[list[int, int]], idx: int):

    for tsmp in timestamps:

      start, end = tsmp
      signal, sampling_rate = audiofile.read(
          file_path,
          offset= start,
          duration= end - start,
          always_2d=True,
      )

      result = self.smile.process_signal(
        signal,
        sampling_rate
      )

      result.insert(0, column='idx', value=idx)
      result.insert(1, column='start', value=start)
      result.insert(2, column='end', value=end)
      
      # if self.audio_df is not None:
      self.audio_df = pd.concat([self.audio_df, result]).drop_duplicates(subset=['idx', 'start', 'end'])
      # else:
      #   result.set_index(['link', 'start', 'end'])
      #   result.to_csv(self.audio_file_path, index=False)
    
    self.audio_df.to_csv(self.audio_file_path, index=False)