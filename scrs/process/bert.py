import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertModel, BertConfig
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys, os

# Sauvgarder le modèle
def save_model(model, save_dir="../../bin", filename="bert_cnn_lstm.pt"):
    """
    Sauvegarde les poids du modèle dans un fichier local.
    """
    model_path = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), model_path)
    print(f"Modèle sauvegardé dans : {model_path}")

# Charger le modèle
def load_model(model_class, model_kwargs, save_dir="../../bin", filename="bert_cnn_lstm.pt", device="cpu"):
    """
    Charge les poids d’un modèle à partir d’un fichier local.
    
    model_class : la classe du modèle (ex: BERT_CNN_LSTM_Classifier)
    model_kwargs : un dictionnaire des paramètres d'initialisation (ex: {"hidden_dim": 768, "num_classes": 2})
    """
    model = model_class(**model_kwargs)
    model_path = os.path.join(save_dir, filename)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Modèle chargé depuis : {model_path}")
    return model

# Hyperparamètres
SEQ_LEN = 256
BATCH_SIZE = 32
EPOCHS = 4
PRETRAINED_MODEL = "hfl/chinese-roberta-wwm-ext"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pré-traitement
def map_to_categorical(df):
    df['label'] = pd.Categorical(df.Tag, ordered=True).codes
    label2Index = {row['Tag']: row['label'] for idx, row in df.iterrows()}
    index2label = {row['label']: row['Tag'] for idx, row in df.iterrows()}
    df.rename(columns={'label': 'labels', 'Content': 'text'}, inplace=True)
    return df[['text', 'labels']], label2Index, index2label

df = pd.read_csv('../../data/clean/steam_reviews.csv')
df, label2Index, index2label = map_to_categorical(df)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)

# Dataset
class BERTDataset(Dataset):
    def __init__(self, df, tokenizer, seq_len):
        self.texts = df['text'].tolist()
        self.labels = df['labels'].tolist()
        self.seq_len = seq_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.seq_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Créer Dataset et DataLoader

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
total_size = len(df)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

df_train = df.iloc[:train_size]
df_val = df.iloc[train_size:train_size+val_size]
df_test = df.iloc[train_size+val_size:]

# data augmentation (optionnel)
from aug import augment_df
df_train_aug = augment_df(df_train)

train_dataset = BERTDataset(df_train_aug, tokenizer, SEQ_LEN)
val_dataset = BERTDataset(df_val, tokenizer, SEQ_LEN)
test_dataset = BERTDataset(df_test, tokenizer, SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# BERT → CNN → BiLSTM → Dense
class BERT_CNN_LSTM_Classifier(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained(PRETRAINED_MODEL)

        for name, param in self.bert.named_parameters():
            if any([name.startswith(f'encoder.layer.{i}') for i in [10, 11]]) or name.startswith('pooler'):
                param.requires_grad = True
            else:
                param.requires_grad = False

        # CNN
        self.conv1d = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # BiLSTM on top of CNN output
        self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_dim,
                            bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # shape: (B, L, 768)

        x = x.transpose(1, 2)  # shape: (B, 768, L) → adapter Conv1d
        x = self.relu(self.conv1d(x))  # shape: (B, 256, L)
        x = x.transpose(1, 2)  # shape: (B, L, 256) → transformer en LSTM 

        lstm_out, _ = self.lstm(x)
        pooled = lstm_out[:, -1, :]
        x = self.dropout(pooled)
        logits = self.classifier(x)
        return logits

model = BERT_CNN_LSTM_Classifier(hidden_dim=768, num_classes=len(label2Index)).to(DEVICE)

from sklearn.utils.class_weight import compute_class_weight

labels_np = df['labels'].to_numpy()
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels_np), y=labels_np)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR

def train(model, train_loader, val_loader, epochs):
    # Définition de la fonction de perte pondérée selon les classes
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # Optimiseur Adam avec un taux d’apprentissage initial
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # Planificateur de taux d’apprentissage : réduction du LR toutes les 2 époques
    scheduler = StepLR(optimizer, step_size=2, gamma=0.5) 

    # Listes pour stocker l’exactitude à l'entraînement et en validation
    train_accs, val_accs = [], []
    train_losses = []
    best_val_acc = 0
    patience = 2  # Nombre d'époques à attendre avant l'arrêt anticipé
    patience_counter = 0

    for epoch in range(epochs):
        model.train()  # Mode entraînement
        total_correct, total = 0, 0
        epoch_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in loop:
            # Chargement des données sur le bon appareil (CPU/GPU)
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            optimizer.zero_grad()  # Réinitialiser les gradients
            outputs = model(input_ids, attention_mask)  # Prédictions du modèle
            loss = criterion(outputs, labels)  # Calcul de la perte
            loss.backward()  # Rétropropagation du gradient

            # Éviter les gradients explosifs
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()  # Mise à jour des poids

            # Calcul de la précision sur le lot actuel
            preds = torch.argmax(outputs, dim=1)
            correct = (preds == labels).sum().item()
            acc = correct / labels.size(0)
            loop.set_postfix({
                'Batch Loss': f"{loss.item():.4f}",
                'Batch Acc': f"{acc:.4f}"
            })

            total_correct += correct
            total += labels.size(0)
            epoch_loss += loss.item()

        scheduler.step()  # Mise à jour du taux d’apprentissage
        train_acc = total_correct / total
        train_accs.append(train_acc)
        train_losses.append(epoch_loss)

        # Validation
        model.eval()
        total_correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)

                outputs = model(input_ids, attention_mask)
                preds = torch.argmax(outputs, dim=1)
                total_correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = total_correct / total
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1} Summary — Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # EarlyStopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_model(model, save_dir="../../bin", filename="best_model.pt")  # Sauvegarde du meilleur modèle
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")  # Arrêt si pas d’amélioration suffisante
                break

    return train_accs, val_accs, train_losses  # Retourne les listes d’exactitudes

# Entraînement
train_accs, val_accs, train_losses = train(model, train_loader, val_loader, EPOCHS)

# Plot
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from plot.visual import plot_train_val_accuracy, plot_train_loss

BASE_DIR = os.getcwd()
PLOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'plot'))

plot_train_loss(train_losses, title="Train Loss over Epochs", save_dir=PLOT_DIR)
plot_train_val_accuracy(train_accs, val_accs, title="Train/Val Accuracy over Epochs", save_dir=PLOT_DIR)

# Evaluation

# Charger le modèle (si besion)
'''
model_kwargs = {
    "hidden_dim": 768,
    "num_classes": len(label2Index)
}
model = load_model(BERT_CNN_LSTM_Classifier, model_kwargs, device=DEVICE)
'''

def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    
    return y_true, y_pred

y_true, y_pred = evaluate(model, test_loader)

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from plot.visual import plot_evaluation

plot_evaluation(
    y_true, y_pred,
    labels=list(index2label.values()),
    save_dir=PLOT_DIR,
    cm_filename="test_confusion.png",
    report_filename="test_classification_report.txt"
)
