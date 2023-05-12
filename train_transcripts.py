from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import pandas as pd
import torch

from torch.utils.data import DataLoader

class TrainTranscripts():
    def __init__(self):
  
      #Define the model. Either from scratch of by loading a pre-trained model
      self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

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
      
      self.model = self.model.to(self.device)

      # reading dataset for transcripts
      self.dataset = pd.read_csv("new_data.csv")

      self.dataset = self.dataset.filter(like='transcript')
      self.dataset = self.dataset.values.tolist()

      new_dataset = []

      for list in self.dataset:
         new_dataset.append([x for x in list if str(x) != 'nan'])
   
    #   train_examples = [InputExample(texts=sentences, label=0) for sentences in new_dataset]

      train_examples = [InputExample(texts=['My first sentence', 'My second sentence', 'Third'], label=0.8),
      InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]

      # # defining dataloader
      self.train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

      # # decide on loss function
      self.train_loss = losses.CosineSimilarityLoss(self.model)

    def train(self):
      # tuning the model
      self.model.fit(train_objectives=[(self.train_dataloader, self.train_loss)], epochs=5, warmup_steps=100, show_progress_bar=True)
  
ttr = TrainTranscripts()
ttr.train()