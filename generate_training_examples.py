# ensure that you first run preprocess_data.py first and obtain new_data.csv, audio_embeddings.csv and test

# text embeddings: SBERT
# audio embeddings: OpenSMILE (already present in audio_embeddings.csv)
# video embeddings: ResNet18

import os
import pandas as pd
import torch
import math
import pickle
from utils import get_time_range_in_sec
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# class SBERTModel():
#   def __init__(self) -> None:
#     # for getting sentence embeddings
#     self.sbert = SentenceTransformer("all-MiniLM-L6-v2")
  
#   def forward(self, features):

#     sbert_output = self.sbert(features)

#     output = sbert_output['sentence_embeddings']

#     print(output)

class GenerateTrainingExamples():

  def __init__(self) -> None:

    # each training example will have a timestamp, its transcript, audio embedding and average of keyframe embeddings
    
    # read the updated dataset
    self.df = pd.read_csv('new_data.csv')

    if not os.path.exists("testframes"):
      os.makedirs("testframes")

    self.df = self.df.filter(regex='index|product|time stamp|aspect|complaint|transcript')
    self.df = self.df.drop(['video transcript', 'product name'], axis=1, errors='ignore')
    
    self.audio_df = pd.read_csv('audio_embeddings.csv')
    self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')


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

    # # initializing language model
    # self.sbert_model = SBERTModel()
    # self.sbert_model = self.sbert_model.to(self.device)

    self.cols = self.df.columns.to_list()

    self.label_cols = [x for x in self.cols if "complaint" in x]
    # self.transcript_cols = [x for x in self.cols if "transcript" in x]
    # self.aspect_cols = [x for x in self.cols if "aspect" in x]

    # # has video index + transcripts
    # transcripts = self.df[["video index"] + self.transcript_cols]

    # labels = self.df[["video index"] + self.label_cols]

    # print(labels)

    # print(transcripts)
    self.training_examples = []
    nan_count = 0

    # dict of aspect labels
    self.aspects = ["design", "camera", "os", "battery life", "speakers"]

    for idx, row in tqdm(self.df.iterrows()):
      video_id = row["video index"]
      video_base_path = f"testframes/{video_id}"
      image_list = os.listdir(video_base_path)
      category = row["product"]

      for label in self.label_cols:
        if not math.isnan(row[label]):
          num = label.strip()[-1]

          timestamp = row["time stamp "+num]
          timestamp = get_time_range_in_sec(timestamp)

          aspect = row["aspect " + num].lower()
          # print("Aspect: ", aspect)

          if aspect not in self.aspects:
            continue

          # get one-hot label of aspect
          aspect_vec = [int(i == self.aspects.index(aspect)) for i in range(len(self.aspects))]
          # print("Aspect vec: ", aspect_vec)


          # sentence to get the embedding of
          if "transcript " + num not in row:
             continue
          transcript = row["transcript "+num]

          if not isinstance(transcript, str):
             nan_count+=1
             continue

          # tokenize sentence
          tokenized_output = self.tokenizer(transcript, max_length = 130, padding='max_length', truncation= True)
          input_ids = tokenized_output.input_ids
          attention_masks = tokenized_output.attention_mask

          # audio embedding from file
          audio_embedding = self.audio_df.loc[(self.audio_df["idx"] == video_id) & (self.audio_df["start"] == timestamp[0]) & (self.audio_df["end"] == timestamp[1])]
          audio_embedding = audio_embedding.drop(['idx', 'start', 'end'], axis=1)

          # final audio embedding list to be attached to training example
          audio_embedding = audio_embedding.values.tolist()[0]
          # print(video_id, num, timestamp, transcript, aspect)

          # get video embeddings and take average
          video_embeddings = []

          for image in image_list:
             name_to_tstamp = image[:-4].split("-")
             name_to_tstamp = [int(name_to_tstamp[0]), int(name_to_tstamp[1])]
             # check if there is overlap in the two timestamps
             if name_to_tstamp[1] >= timestamp[0] and timestamp[1] >= name_to_tstamp[0]:
              # print(name_to_tstamp, timestamp)
              image_to_process = Image.open(video_base_path + "/" + image)
              image_tensor = self.transform(image_to_process).unsqueeze(0)
              image_tensor = image_tensor.to(self.device)

              with torch.no_grad():
                features = self.resnet(image_tensor)

              # print(features.shape)
              video_embeddings.append(features.detach().cpu())
          
          mean_video_embedding = torch.mean(torch.stack(video_embeddings), dim=0)
          
          # print(len(mean_video_embedding), mean_video_embedding[0].shape)
          self.training_examples.append([category, aspect_vec, int(row[label]), input_ids, attention_masks, audio_embedding, mean_video_embedding])

    print("Nan data count: ", nan_count)
    print("No. of training examples: ", len(self.training_examples))

    with open('dataset2.pickle', 'wb') as handle:
      pickle.dump(self.training_examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    



gte = GenerateTrainingExamples()