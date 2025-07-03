import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
import os

class SentimentAnalyzer:
    def __init__(self, model_name='indobenchmark/indobert-base-p1', max_len=128, batch_size=16):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Model akan menggunakan device: {self.device}")
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_len = max_len
        self.batch_size = batch_size
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        return IndoBERT_LSTM_Classifier(
            bert_model_name=self.model_name, hidden_dim=256, output_dim=2,
            n_layers=2, bidirectional=True, dropout=0.25
        )

    def prepare_dataloaders(self, df: pd.DataFrame):
        encoded_inputs = self.tokenizer.batch_encode_plus(
            df['cleaned_review'].values, add_special_tokens=True, return_attention_mask=True,
            padding='max_length', max_length=self.max_len, truncation=True, return_tensors='pt'
        )
        input_ids, attention_masks, labels = encoded_inputs['input_ids'], encoded_inputs['attention_mask'], torch.tensor(df['label'].values)
        train_inputs, test_inputs, train_labels, test_labels = train_test_split(input_ids, labels, random_state=42, test_size=0.2, stratify=labels)
        train_masks, test_masks, _, _ = train_test_split(attention_masks, labels, random_state=42, test_size=0.2, stratify=labels)
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        self.train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=self.batch_size)
        test_data = TensorDataset(test_inputs, test_masks, test_labels)
        self.test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=self.batch_size)
        print("DataLoaders siap.")

    def train(self, epochs=3):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        criterion = nn.CrossEntropyLoss().to(self.device)
        print(f"Memulai training untuk {epochs} epoch...")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in self.train_dataloader:
                b_input_ids, b_input_mask, b_labels = [b.to(self.device) for b in batch]
                optimizer.zero_grad()
                predictions = self.model(b_input_ids, b_input_mask)
                loss = criterion(predictions, b_labels)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            avg_train_loss = total_loss / len(self.train_dataloader)
            print(f"Epoch {epoch + 1}/{epochs} | Training Loss: {avg_train_loss:.4f}")
        print("Training selesai.")

    def evaluate(self):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in self.test_dataloader:
                b_input_ids, b_input_mask, b_labels = [b.to(self.device) for b in batch]
                predictions = self.model(b_input_ids, b_input_mask)
                preds = torch.argmax(predictions, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(b_labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
        report_dict = classification_report(all_labels, all_preds, target_names=['Negatif', 'Positif'], output_dict=True)
        cm = confusion_matrix(all_labels, all_preds)
        tn, fp, fn, tp = cm.ravel()

        print("\n--- Hasil Evaluasi Model ---")
        print(f"Akurasi: {accuracy:.4f} | Presisi (Macro): {precision:.4f} | Recall (Macro): {recall:.4f} | F1-Score (Macro): {f1:.4f}")
        print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
        print("---------------------------\n")
        
        return {
            'accuracy': accuracy, 'precision_macro': precision, 'recall_macro': recall,
            'f1_macro': f1, 'report_dict': report_dict, 'confusion_matrix': cm
        }

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
            print(f"Peringatan: File model tidak ditemukan di {file_path}. Model tidak dimuat.")

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
        
    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output[0]
        _, (hidden, _) = self.lstm(sequence_output)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden)