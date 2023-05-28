import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm, trange
from sklearn.metrics import classification_report
from typing import Optional
from maf_model import MAF

def get_dataset():

    with open('../dataset2.pickle', 'rb') as handle:
        dataset = pickle.load(handle)

    all_aspect_labels = torch.tensor([f[1] for f in dataset], dtype=torch.float)
    all_labels = torch.tensor([f[2] for f in dataset], dtype=torch.long)
    all_input_ids = torch.tensor([f[3] for f in dataset], dtype=torch.long)
    all_input_mask = torch.tensor([f[4] for f in dataset], dtype=torch.long)
    all_audio_embedding = torch.tensor([f[5] for f in dataset], dtype=torch.float)
    all_video_embedding = torch.stack([torch.squeeze(f[6]) for f in dataset])

    # preparing TensorDataset
    dataset = TensorDataset(
        all_input_ids,
        all_input_mask,
        all_audio_embedding,
        all_video_embedding,
        all_labels,
        all_aspect_labels
    )

    # splitting into train and test datasets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    print("Training samples: ", len(train_dataset), "\nTest samples: ", len(test_dataset))

    return train_dataset, test_dataset

# getting datasets and dataloaders
batch_size = 8

train_dataset, test_dataset = get_dataset()
train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=1
    )

test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=1
    )

text_embedding_size = 384
audio_embedding_size = 88
video_embedding_size = 1000

model = MAF()
# Configure GPU device
device = torch.device("cuda:0")
model = model.to(device)

# Define the binary cross entropy loss
criterion = nn.CrossEntropyLoss()
# Define the optimizer
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

num_epochs = 20
train_iterator = trange(int(num_epochs),
                        desc="Epoch",
                        mininterval=0)
global_step = 0


# Define the training loop
def train(model, dataloader, criterion, optimizer):
    model.train()

    for epoch in train_iterator:
        train_loss = 0.0
        train_correct = 0

        for i, (input_ids, input_mask, audio_embeddings, video_embeddings, complaint_labels, aspect_labels) in enumerate(tqdm(dataloader, desc=f"iteration{global_step}")):
            # Move the inputs to the device
            
            input_ids, input_mask, audio_embeddings, video_embeddings, complaint_labels, aspect_labels = input_ids.to(device), input_mask.to(device), audio_embeddings.to(device), video_embeddings.to(device), complaint_labels.to(device), aspect_labels.to(device)

            # Forward pass
            logits_complaint, logits_aspect = model(input_ids, input_mask, audio_embeddings, video_embeddings)
            # print(logits, labels)
            complaint_loss = criterion(logits_complaint, complaint_labels)
            aspect_loss = criterion(logits_aspect, aspect_labels)

            loss = complaint_loss + aspect_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"\rloss: {loss:.4f}", end="")

            # Compute the training loss and accuracy
            # train_loss += loss.item() * text_embeddings.size(0)
            train_loss += loss.item()
            # train_correct += (torch.argmax(logits_complaint, dim=-1) == complaint_labels).sum().item()

        train_loss /= len(dataloader.dataset)
        # train_accuracy = train_correct / len(dataloader.dataset)

        print('Epoch [{}/{}], Train Loss: {:.4f}'
          .format(epoch+1, num_epochs, train_loss))
    
    # return train_loss, train_accuracy

def test(model, dataloader, criterion):
    model.eval()
    test_loss = 0.0
    # test_correct = 0.0

    complaint_preds = []
    aspect_preds = []
    all_complaint_labels = []
    all_aspect_labels = []

    with torch.no_grad():
        for i, (input_ids, input_mask, audio_embeddings, video_embeddings, complaint_labels, aspect_labels) in enumerate(tqdm(dataloader, desc=f"iteration{global_step}")):
            # Move the inputs to the device
            input_ids, input_mask, audio_embeddings, video_embeddings, complaint_labels, aspect_labels = input_ids.to(device), input_mask.to(device), audio_embeddings.to(device), video_embeddings.to(device), complaint_labels.to(device), aspect_labels.to(device)


            # Forward pass
            complaint_logits, aspect_logits = model(input_ids, input_mask, audio_embeddings, video_embeddings)
            
            complaint_loss = criterion(complaint_logits, complaint_labels)
            aspect_loss = criterion(aspect_logits, aspect_labels)

            complaint_preds.append(torch.argmax(complaint_logits, dim=1))
            aspect_preds.append(torch.eq(torch.max(aspect_logits, dim=1)[0][:, None], aspect_logits).to(torch.int))
            all_complaint_labels.append(complaint_labels)
            all_aspect_labels.append(aspect_labels)

            loss = complaint_loss + aspect_loss


            # # Compute the testing loss and accuracy
            test_loss += loss.item()
            # test_correct += (torch.argmax(logits, dim=-1) == labels).sum().item()

    test_loss /= len(dataloader.dataset)
    # test_accuracy = test_correct / len(dataloader.dataset)

    print("Final test loss: ", test_loss)

    # Concatenate predictions and labels across all batches
    complaint_preds = torch.cat(complaint_preds, dim=0)
    aspect_preds = torch.cat(aspect_preds, dim=0)
    all_complaint_labels = torch.cat(all_complaint_labels, dim=0)
    all_aspect_labels = torch.cat(all_aspect_labels, dim=0)

    # Compute and print classification report for each task
    print('Complaint classification report:')
    print(classification_report(all_complaint_labels.cpu(), complaint_preds.cpu()))
    print('Aspect classification report:')
    print(classification_report(all_aspect_labels.cpu(), aspect_preds.cpu()))
    # print("Final test accuracy: ", test_accuracy)

test(model, test_dataloader, criterion)
train(model, train_dataloader, criterion, optimizer)
test(model, test_dataloader, criterion)

# saving model for e2e testing with new videos
torch.save(model.state_dict(), "../final_model.pt")
print("Final model saved in final_model.pt in root directory.")