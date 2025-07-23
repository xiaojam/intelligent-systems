# sentiment_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix
import os
import pandas as pd

class IndoBERT_Classifier(nn.Module):
    def __init__(self, bert_model_name):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=2)
    def forward(self, input_ids, attention_mask, labels=None):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

class IndoBERT_CNN_Classifier(nn.Module):
    def __init__(self, bert_model_name, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        embedding_dim = self.bert.config.to_dict()['hidden_size']
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=n_filters, kernel_size=fs) for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, input_ids, attention_mask, labels=None):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embedded = bert_output[0].permute(0, 2, 1)
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

class IndoBERT_LSTM_Classifier(nn.Module):
    def __init__(self, bert_model_name, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(
            self.bert.config.hidden_size, hidden_dim, num_layers=n_layers, 
            bidirectional=bidirectional, batch_first=True, dropout=dropout if n_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    def forward(self, input_ids, attention_mask, labels=None):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output[0]
        _, (hidden, _) = self.lstm(sequence_output)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden)

class SentimentAnalyzer:
    def __init__(self, model_type='bert', model_name='indobenchmark/indobert-base-p1', max_len=128, batch_size=64):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Model akan menggunakan device: {self.device} | Arsitektur: {model_type.upper()}")
        self.model_type = model_type
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_len = max_len
        self.batch_size = batch_size
        self.model = self._build_model(model_name).to(self.device)

    def _build_model(self, model_name):
        if self.model_type.lower() == 'bert':
            return IndoBERT_Classifier(bert_model_name=model_name)
        elif self.model_type.lower() == 'cnn':
            return IndoBERT_CNN_Classifier(
                bert_model_name=model_name, n_filters=100, filter_sizes=[2, 3, 4], output_dim=2, dropout=0.5
            )
        elif self.model_type.lower() == 'lstm':
            return IndoBERT_LSTM_Classifier(
                bert_model_name=model_name, hidden_dim=256, output_dim=2,
                n_layers=2, bidirectional=True, dropout=0.25
            )
        else:
            raise ValueError("Tipe model tidak dikenali. Pilih 'bert', 'cnn', atau 'lstm'.")

    def prepare_dataloaders(self, df: pd.DataFrame):
        encoded_inputs = self.tokenizer.batch_encode_plus(df['cleaned_review'].values, add_special_tokens=True, return_attention_mask=True, padding='max_length', max_length=self.max_len, truncation=True, return_tensors='pt')
        labels = torch.tensor(df['label'].values)
        dataset = TensorDataset(encoded_inputs['input_ids'], encoded_inputs['attention_mask'], labels)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, self.test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        self.train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=self.batch_size)
        self.test_dataloader = DataLoader(self.test_dataset, sampler=SequentialSampler(self.test_dataset), batch_size=self.batch_size)
        print(f"DataLoaders siap (Train: {train_size}, Test: {test_size}).")

    def train(self, epochs=3):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        criterion = nn.CrossEntropyLoss().to(self.device)
        history = {'loss': [], 'val_loss': []}
        print(f"🏋️ Memulai training untuk {epochs} epoch...")
        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0
            for batch in self.train_dataloader:
                b_input_ids, b_input_mask, b_labels = [b.to(self.device) for b in batch]
                self.model.zero_grad()
                if self.model_type.lower() == 'bert':
                    outputs = self.model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                    loss = outputs.loss
                else:
                    predictions = self.model(b_input_ids, attention_mask=b_input_mask)
                    loss = criterion(predictions, b_labels)
                total_train_loss += loss.item()
                loss.backward()
                optimizer.step()
            avg_train_loss = total_train_loss / len(self.train_dataloader)
            history['loss'].append(avg_train_loss)
            
            # Validation loop
            self.model.eval()
            total_val_loss = 0
            for batch in self.test_dataloader:
                b_input_ids, b_input_mask, b_labels = [b.to(self.device) for b in batch]
                with torch.no_grad():
                    if self.model_type.lower() == 'bert':
                        outputs = self.model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                        loss = outputs.loss
                    else:
                        predictions = self.model(b_input_ids, attention_mask=b_input_mask)
                        loss = criterion(predictions, b_labels)
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(self.test_dataloader)
            history['val_loss'].append(avg_val_loss)
            
            print(f"Epoch {epoch + 1}/{epochs} | Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")
        print("Training selesai.")
        return history

    def evaluate(self):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in self.test_dataloader:
                b_input_ids, b_input_mask, b_labels = [b.to(self.device) for b in batch]
                if self.model_type.lower() == 'bert':
                    outputs = self.model(b_input_ids, attention_mask=b_input_mask)
                    logits = outputs.logits
                else:
                    logits = self.model(b_input_ids, attention_mask=b_input_mask)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(b_labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
        report_dict = classification_report(all_labels, all_preds, target_names=['Negatif', 'Positif'], output_dict=True)
        cm = confusion_matrix(all_labels, all_preds)
        print(f"\n--- Hasil Evaluasi --- \nAkurasi: {accuracy:.4f} | F1-Score (Macro): {f1:.4f}\n")
        return {'accuracy': accuracy, 'precision_macro': precision, 'recall_macro': recall, 'f1_macro': f1, 'report_dict': report_dict, 'confusion_matrix': cm}
    
    def save_model(self, file_path='model.pt'):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(self.model.state_dict(), file_path)
        print(f"Model berhasil disimpan di: {file_path}")

    def load_model(self, file_path='model.pt'):
        if os.path.exists(file_path):
            self.model.load_state_dict(torch.load(file_path, map_location=self.device))
            self.model.eval()
            print(f"Model berhasil dimuat dari: {file_path}")
        else:
            raise FileNotFoundError(f"File model tidak ditemukan di {file_path}. Lakukan training terlebih dahulu.")