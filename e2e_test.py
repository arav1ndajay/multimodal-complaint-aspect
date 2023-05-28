import os
import sys
import cv2
import opensmile
import audiofile
import pandas as pd
from tqdm import tqdm
from pytube import YouTube
import whisper
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import AutoTokenizer
from PIL import Image
from models.maf_model import MAF
from audio import GenerateAudio
from keyframes import GenerateKeyframes
from utils import get_time_range_in_format, get_time_range_in_sec
import numpy as np


class GenerateTranscript():
  def __init__(self):
    self.sizes = list(whisper._MODELS.keys())
    self.langs = ["none"] + sorted(list(whisper.tokenizer.LANGUAGES.values()))
    self.current_size = "base"
    self.loaded_model = whisper.load_model(self.current_size)

   
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

        # for text input creation
    self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    # aspects for final predictions
    self.aspects = ["design", "camera", "os", "battery life", "speakers"]
      
    # for audio embeddings
    self.smile = opensmile.Smile(
      feature_set=opensmile.FeatureSet.eGeMAPSv02,
      feature_level=opensmile.FeatureLevel.Functionals,
    )

    # for video embeddings
    # model for video embeddings
    self.resnet= models.resnet50(weights='ResNet50_Weights.DEFAULT')
    self.resnet = self.resnet.to(self.device)

    self.transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    ])
  
  def progress_function(self, stream, chunk, bytes_remaining):
    current = ((self.filesize - bytes_remaining)/self.filesize)
    percent = ('{0:.1f}').format(current*100)
    progress = int(50*current)
    status = '█' * progress + '-' * (50 - progress)
    sys.stdout.write(' ↳ |{bar}| {percent}%\r'.format(bar=status, percent=percent))
    sys.stdout.flush()
  
  def get_transcript(self, link: str, lang: str, size: str, subs: str):
    transcripts = {}

    if size != self.current_size:
      self.loaded_model = whisper.load_model(size)
      self.current_size = size
    
    if lang == "none":
      lang = None

    if not os.path.exists("test_downloads"):
      os.makedirs("test_downloads")

    # begin by getting video from YouTube
    self.yt = YouTube(link, use_oauth=True, allow_oauth_cache=True)
    self.yt.register_on_progress_callback(self.progress_function)

      
    if not os.path.exists(f"test_downloads/video.mp4"):
      try:
          video = self.yt.streams.filter(res="360p")[0]
          self.filesize = video.filesize
          path = video.download(filename=f"test_downloads/video.mp4")
      except Exception as error:
          print(error)
    
    else:
        print(f"Video exists. Skipping download...")
        path = f"test_downloads/video.mp4"
      
    # get entire transcript from audio
    print("Generating transcripts...")
    results = self.loaded_model.transcribe(path, language=lang)
      
    if subs == "None":
        transcripts = results["text"]
    elif subs == ".srt":
        transcripts = self.srt(results["segments"])
    elif ".csv" == ".csv":
        transcripts = self.csv(results["segments"])
    
    print(transcripts)
      
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
  
  def process_video(self, link: str):

    transcript = self.get_transcript(link, "english", "base", ".csv")

    self.timestamps = []
    self.sents = []


    # get all sentences from a transcript
    sentences = transcript.split("\n")

    for sent in sentences:

      if sent != '':
        # start time, end time and sentence
        st, et, words = sent.split(',', 2)
        st = int(float(st))
        et = int(float(et))

        self.timestamps.append([int(float(st)), int(float(et))])
        self.sents.append(words)

    print(f"Num. of instances: {len(self.timestamps)}")
    
    print("Generating audio embeddings...")
    file_path = f"test_downloads/video.mp4"

    audio_embeddings = []

    # get audio embeddings
    for tsmp in tqdm(self.timestamps):

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

      result = result.values.tolist()[0]

      audio_embeddings.append(result)

    # print("Capturing keyframes...")

    # if not os.path.exists("test_keyframes"):
    #     os.makedirs("test_keyframes")

    # for tsmp in tqdm(timestamps):
    #     vidcap = cv2.VideoCapture(file_path)
    #     fps = vidcap.get(cv2.CAP_PROP_FPS)

    #     start, end = tsmp

    #     success,image = vidcap.read()
    #     count = 0
    #     success = True

    #     timestamp_runner = -1

    #     while success:
    #       success,frame = vidcap.read()
    #       count+=1
    #       timestamp = int(count/fps)

    #       if timestamp >= start and timestamp <= end:
    #         if timestamp_runner == -1:
    #           timestamp_runner = timestamp
          
    #         # save a frame every 5 seconds in a timestamp interval
    #         if timestamp == timestamp_runner:
    #           cv2.imwrite(f"test_keyframes/{timestamp_runner}-{min(timestamp_runner + 5, end)}.jpg", frame)
    #           timestamp_runner += 5
    #           if timestamp_runner >= end: break
          
    #       elif timestamp > end: break

    self.testing_examples = []

    print("Generating testing instances...")
    for idx, tsmp in enumerate(tqdm(self.timestamps)):

      # tokenized text
      tokenized_output = self.tokenizer(self.sents[idx], max_length = 130, padding='max_length', truncation= True)
      input_ids = tokenized_output.input_ids
      attention_masks = tokenized_output.attention_mask

      # get video embeddings and take average
      video_embeddings = []
      image_list = os.listdir("test_keyframes")

      for image in image_list:
          name_to_tstamp = image[:-4].split("-")
          name_to_tstamp = [int(name_to_tstamp[0]), int(name_to_tstamp[1])]
          # check if there is overlap in the two timestamps
          if name_to_tstamp[1] >= tsmp[0] and tsmp[1] >= name_to_tstamp[0]:
            # print(name_to_tstamp, timestamp)
            image_to_process = Image.open("test_keyframes/" + image)
            image_tensor = self.transform(image_to_process).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)

            with torch.no_grad():
              features = self.resnet(image_tensor)

            # print(features.shape)
            video_embeddings.append(features.detach().cpu())
      
      mean_video_embedding = torch.mean(torch.stack(video_embeddings), dim=0)
      
      # print(len(mean_video_embedding), mean_video_embedding[0].shape)
      self.testing_examples.append([input_ids, attention_masks, audio_embeddings[idx], mean_video_embedding])
    
  def run_model(self):

    # loading model generated by models/main.py
    model = MAF()
    model.load_state_dict(torch.load("final_model.pt"))
    model = model.to(self.device)
    model.eval()

    dataset = self.testing_examples

    all_input_ids = torch.tensor([f[0] for f in dataset], dtype=torch.long)
    all_input_masks = torch.tensor([f[1] for f in dataset], dtype=torch.long)
    all_audio_embeddings = torch.tensor([f[2] for f in dataset], dtype=torch.float)
    all_video_embeddings = torch.stack([torch.squeeze(f[3]) for f in dataset])

    all_input_ids, all_input_masks, all_audio_embeddings, all_video_embeddings = all_input_ids.to(self.device), all_input_masks.to(self.device), all_audio_embeddings.to(self.device), all_video_embeddings.to(self.device)


    # preparing dataset
    # test_dataset = TensorDataset(
    #     all_input_ids,
    #     all_input_mask,
    #     all_audio_embedding,
    #     all_video_embedding,
    # )


    # dataloader = DataLoader(
    #     test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=1
    # )

    # get output from the model
    complaint_logits, aspect_logits = model(all_input_ids, all_input_masks, all_audio_embeddings, all_video_embeddings)

    # print(complaint_logits.shape, aspect_logits.shape)

    complaint_preds = torch.argmax(complaint_logits, dim=1).tolist()

    aspect_preds = torch.eq(torch.max(aspect_logits, dim=1)[0][:, None], aspect_logits).to(torch.int).tolist()

    pred_indices = []
    start = None
    for i, pred in enumerate(complaint_preds):
      # indicates complaint
      if pred == 0:
        if start is None:
         start = i
      elif start is not None:
         pred_indices.append([start, i-1])
         start = None
    
    if start is not None:
       pred_indices.append([start, len(complaint_preds)-1])

    for indices in pred_indices:
        start = indices[0]
        end = indices[1]

        print(f"{self.timestamps[start][0]} : {self.timestamps[end][1]} - ", end="")

        possible_aspects = []

        for idx in range (start, end+1):
          possible_aspects.append(self.aspects[aspect_preds[idx].index(1)])

        print(possible_aspects)


    
gtr = GenerateTranscript()

link = "https://www.youtube.com/watch?v=Imsw1jeGt9o"
gtr.process_video(link)

print("Processing completed. Running model...")
gtr.run_model()