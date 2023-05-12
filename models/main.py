import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm, trange
from sklearn.metrics import classification_report


# Define the bimodal cross attention model
# class BimodalCrossAttention(nn.Module):
#     def __init__(self, text_embedding_size, audio_embedding_size, video_embedding_size):
#         super().__init__()
        
#         # Text cross attention layer
#         self.text_cross_attention = nn.MultiheadAttention(text_embedding_size, num_heads=8)
#         self.text_linear = nn.Linear(text_embedding_size, text_embedding_size)
#         self.text_pool = nn.AdaptiveMaxPool1d(1)
        
#         # Audio-visual cross attention layer
#         self.av_cross_attention = nn.MultiheadAttention(audio_embedding_size + video_embedding_size, num_heads=8)
#         self.av_linear = nn.Linear(audio_embedding_size + video_embedding_size, audio_embedding_size + video_embedding_size)
#         self.av_pool = nn.AdaptiveMaxPool1d(1)

#         # Classification layer
#         self.classifier = nn.Linear(text_embedding_size + audio_embedding_size + video_embedding_size, 6)
        
#     def forward(self, text_embeddings, audio_embeddings, video_embeddings):
#         # Text cross attention
#         text_output, _ = self.text_cross_attention(text_embeddings.permute(1, 0, 2), text_embeddings.permute(1, 0, 2), text_embeddings.permute(1, 0, 2))
#         text_output = self.text_linear(text_output)
#         text_output = text_output.permute(1, 2, 0)
#         text_output = self.text_pool(text_output).squeeze()
        
#         # Audio-visual cross attention
#         av_input = torch.cat([audio_embeddings, video_embeddings], dim=-1)
#         av_output, _ = self.av_cross_attention(av_input.permute(1, 0, 2), av_input.permute(1, 0, 2), av_input.permute(1, 0, 2))
#         av_output = self.av_linear(av_output)
#         av_output = av_output.permute(1, 2, 0)
#         av_output = self.av_pool(av_output).squeeze()
        
#         # Concatenate text and audio-visual features
#         features = torch.cat([text_output, av_output], dim=-1)
        
#         # Classification
#         logits = self.classifier(features)
#         probabilities = nn.functional.softmax(logits, dim=-1)
#         return logits, probabilities

class ComplaintIdentificationModel(nn.Module):
    def __init__(self, text_embedding_size=768, audio_embedding_size=88, video_embedding_size=1000, num_complaint_labels=2, num_aspect_labels=5):
        super().__init__()

        # Text input layers
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.text_fc = nn.Linear(text_embedding_size, text_embedding_size)
        self.text_cross_attention = nn.MultiheadAttention(text_embedding_size, 4)

        # Audio input layers
        self.audio_fc = nn.Linear(audio_embedding_size, text_embedding_size)
        self.audio_cross_attention = nn.MultiheadAttention(text_embedding_size, 4)

        # Video input layers
        self.video_fc = nn.Linear(video_embedding_size, text_embedding_size)
        self.video_cross_attention = nn.MultiheadAttention(text_embedding_size, 4)

        self.fusion_fc = nn.Linear(text_embedding_size*3, text_embedding_size)
        # self.fusion_fc = nn.Linear(text_embedding_size*2, text_embedding_size)
        # Output layers
        self.output_complaint = nn.Linear(text_embedding_size, num_complaint_labels)

        self.output_aspect = nn.Linear(text_embedding_size, num_aspect_labels)

        # Softmax function
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, audio_embeddings, video_embeddings):
        # Text input
        features = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

        sbert_output = self.sbert_model(features)
        text_embeddings = sbert_output['sentence_embedding']
        text_embeddings = self.text_fc(text_embeddings)
        # text_embeddings = text_embeddings.transpose(0,1)


        # Audio input
        audio_embeddings = self.audio_fc(audio_embeddings)
        # audio_embeddings, _ = self.audio_cross_attention(audio_embeddings.transpose(0, 1), text_embeddings.transpose(0, 1), text_embeddings.transpose(0, 1))
        audio_embeddings, _ = self.audio_cross_attention(audio_embeddings, text_embeddings, text_embeddings)
        
        # audio_embeddings = audio_embeddings.transpose(0, 1)

        # Video input
        video_embeddings = self.video_fc(video_embeddings)
        # video_embeddings, _ = self.video_cross_attention(video_embeddings.transpose(0, 1), text_embeddings.transpose(0, 1), text_embeddings.transpose(0, 1))
        video_embeddings, _ = self.video_cross_attention(video_embeddings, text_embeddings, text_embeddings)
        # video_embeddings = video_embeddings.transpose(0, 1)

        # Concatenate and pool text and AV embeddings
        # av_embeddings = torch.cat((audio_embeddings, video_embeddings), dim=1)
        # print(av_embeddings.shape)
        # pooled_av_embeddings, _ = torch.max(av_embeddings, dim=1)
        # print(pooled_av_embeddings.shape)
        # pooled_text_embeddings, _ = torch.max(text_embeddings, dim=1)
        # print(pooled_av_embeddings.shape, pooled_text_embeddings.shape)
        # pooled_embeddings = torch.cat((pooled_av_embeddings, pooled_text_embeddings), dim=1)

        fusion_embeddings = torch.cat([text_embeddings, audio_embeddings, video_embeddings], dim=1)
        # fusion_embeddings = torch.cat([text_embeddings, audio_embeddings], dim=1)

        fusion_embeddings = self.fusion_fc(fusion_embeddings)


        # Output layer for complaint
        # outputs = self.output_fc(pooled_av_embeddings)
        complaint_outputs = self.output_complaint(fusion_embeddings)
        # Softmax
        complaint_outputs = self.softmax(complaint_outputs)
        # print("outputs: ", outputs)

        # output layer for aspect identification
        aspect_outputs = self.output_aspect(fusion_embeddings)
        aspect_outputs = self.softmax(aspect_outputs)

        return complaint_outputs, aspect_outputs


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

model = ComplaintIdentificationModel(text_embedding_size, audio_embedding_size, video_embedding_size)
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