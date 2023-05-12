import os
import sys
import pandas as pd
from tqdm import tqdm
import torch
from pytube import YouTube
import whisper
from audio import GenerateAudio
from keyframes import GenerateKeyframes
from utils import get_time_range_in_format, get_time_range_in_sec

class GenerateTranscript():
  def __init__(self):
    self.sizes = list(whisper._MODELS.keys())
    self.langs = ["none"] + sorted(list(whisper.tokenizer.LANGUAGES.values()))
    self.current_size = "base"
    self.loaded_model = whisper.load_model(self.current_size)

    # for audio embeddings
    self.gau = GenerateAudio()
    # for keyframes
    self.gfk = GenerateKeyframes()
    

    # change dataset path here
    self.dataset = pd.read_csv("complaint3.csv")
    self.dataset = self.dataset.loc[self.dataset["Video Link"].notnull()]
   
    self.yt = None
    self.filesize = None

    self.use_cuda = True
    self.cuda_device = -1
    if self.use_cuda:
        if torch.cuda.is_available():
            if self.cuda_device == -1:
                self.device = torch.device("cuda")
            else:
                self.device = torch.device(f"cuda:{self.cuda_device}")
        else:
            raise ValueError(
                "'use_cuda' set to True when cuda is unavailable."
                " Make sure CUDA is available or set use_cuda=False.")
    else:
        self.device = "cpu"
  
  def progress_function(self, stream, chunk, bytes_remaining):
    current = ((self.filesize - bytes_remaining)/self.filesize)
    percent = ('{0:.1f}').format(current*100)
    progress = int(50*current)
    status = '█' * progress + '-' * (50 - progress)
    sys.stdout.write(' ↳ |{bar}| {percent}%\r'.format(bar=status, percent=percent))
    sys.stdout.flush()
  
  def get_transcripts(self, links: dict[int, str], lang: str, size: str, subs: str):
    transcripts = {}

    if size != self.current_size:
      self.loaded_model = whisper.load_model(size)
      self.current_size = size
    
    if lang == "none":
      lang = None

    if not os.path.exists("downloads"):
      os.makedirs("downloads")
      
    for idx, link in tqdm(links.items()):
      
      self.yt = YouTube(link)
      self.yt.register_on_progress_callback(self.progress_function)
      # path = self.yt.streams.filter(only_audio=True)[0].download(filename="tmp.mp4")
      
      if not os.path.exists(f"downloads/tmp_{idx}.mp4"):
        try:
          video = self.yt.streams.filter(res="360p")[0]
          self.filesize = video.filesize
          path = video.download(filename=f"downloads/tmp_{idx}.mp4")
        except:
          print(f"Error occurred while downloading video {idx}. Please try for this video again later. Skipping...")
          continue
      
      else:
        print(f"Video {idx} exists. Skipping download...")
        path = f"downloads/tmp_{idx}.mp4"
      
      # get entire transcript from audio
      results = self.loaded_model.transcribe(path, language=lang)
      
      if subs == "None":
        transcripts[idx] = results["text"]
      elif subs == ".srt":
        transcripts[idx] = self.srt(results["segments"])
      elif ".csv" == ".csv":
        transcripts[idx] = self.csv(results["segments"])
      
    return transcripts
   
  def srt(self, segments):
    output = ""
    for i, segment in enumerate(segments):
      output += f"{i+1}\n"
      output += f"{self.format_time(segment['start'])} --> {self.format_time(segment['end'])}\n"
      output += f"{segment['text']}\n\n"
    return output
  
  def csv(self, segments):
    output = ""
    for segment in segments:
      output += f"{segment['start']},{segment['end']},{segment['text']}\n"
    return output

  def format_time(self, time):
    hours = time//3600
    minutes = (time - hours*3600)//60
    seconds = time - hours*3600 - minutes*60
    milliseconds = (time - int(time))*1000
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{int(milliseconds):03d}"
    
  def populate_metadata(self, link):
    self.yt = YouTube(link)
    return self.yt.thumbnail_url, self.yt.title
  
  def get_timestamps(self, data):
    # dictionary of video index as key and array of timestamps (in seconds) as value
    timestamps = {}

    for key, value in data.items():
      if "time stamp" in key.lower():
        for idx, tsp in value.items():
          if isinstance(tsp, str):
            tsp = get_time_range_in_sec(tsp)
            if idx not in timestamps:
              timestamps[idx] = [tsp]
            else:
              timestamps[idx].append(tsp)

    return timestamps
  
  def process_videos(self):
    data = self.dataset.to_dict()
    data = {k.lower(): v for k, v in data.items()}

    links = data['video link']

    timestamps = self.get_timestamps(data)
    print(timestamps)

    transcripts = self.get_transcripts(links, "english", "base", ".csv")

    cherrypicked_ts = {}

    
    # pick out sentences for specific timestamps
    for idx, trs in transcripts.items():
      
      # get all sentences from a transcript
      sentences = trs.split("\n")

      # pointer to go through marked timestamps as outer loop goes over entire transcript
      tsp_ptr = 0

      # if idx not in timestamps:
      #   print(f"Timestamps not present in video {idx}. Skipping this video...")
      #   continue

      for sent in sentences:
        if tsp_ptr == len(timestamps[idx]):
          break

        if sent != '':
          # start time, end time and sentence
          st, et, words = sent.split(',', 2)
          st = int(float(st))
          et = int(float(et))

          ts_st, ts_et = timestamps[idx][tsp_ptr]
          time_key = tuple(timestamps[idx][tsp_ptr])

          # if there is overlap in ranges, add that sentence to timestamp
          if et >= ts_st and ts_et >= st:
            if idx not in cherrypicked_ts:
              cherrypicked_ts[idx] = {time_key : sent }
            else:
              if time_key not in cherrypicked_ts[idx]:
                cherrypicked_ts[idx][time_key] = sent

              else:
                start, end, old_words =  cherrypicked_ts[idx][time_key].split(',', 2)
                new_sentence = str(start)+ ',' + str(et) + ',' + old_words + ' ' + words
                
                cherrypicked_ts[idx][time_key] = new_sentence
          
          if st > ts_et:
            tsp_ptr += 1


    # print("cherrypicked: ", cherrypicked_ts)
    # print("timestamps: ", timestamps)

    new_timestamps = {}
    # return

    # updating dataset with transcripts
    for idx, tstamps in cherrypicked_ts.items():
      
      # adding indices to uniquely identify videos in new data file
      # if idx not in data:
      #   data["idx"] = {idx: idx}
      # else: data["idx"][idx] = idx

      timestamp_number = 1

      for tmp, sentence in tstamps.items():
        start, end, words = sentence.split(',', 2)
        new_ts = get_time_range_in_format([start, end])

        if idx not in new_timestamps:
          new_timestamps[idx] = [[int(float(start)), int(float(end))]]
        
        else: new_timestamps[idx].append([int(float(start)), int(float(end))])

        transcript_name = f"transcript {timestamp_number}"
        
        if transcript_name not in data:
          data[transcript_name] = {idx: words}
        
        else:
          data[transcript_name][idx] = words
        # updating timestamp
        data[f"time stamp {timestamp_number}"][idx] = new_ts
      
        timestamp_number += 1
    
    # print("timestamps after: ", new_timestamps)

    new_dataset = pd.DataFrame(data)
    new_dataset.index.name = "video index"
    new_dataset.to_csv('new_data.csv')


    print("Generating audio embeddings...")
    for idx, tsmp in tqdm(new_timestamps.items()):

      file_path = f"downloads/tmp_{idx}.mp4"
      # self.gau.generateAudioEmbeddings(file_path, tsmp, data["video link"][idx])
      self.gau.generateAudioEmbeddings(file_path, tsmp, idx)
    

    print("Capturing keyframes...")
    for idx, tsmp in tqdm(new_timestamps.items()):

      file_path = f"downloads/tmp_{idx}.mp4"
      self.gfk.generateKeyframes(file_path, tsmp, idx)  
      
      # os.remove(file_path)
    
gtr = GenerateTranscript()

gtr.process_videos()

print("Processing completed.")