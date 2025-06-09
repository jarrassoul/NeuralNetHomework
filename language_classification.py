import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import random
import os

# Create output directory
if not os.path.exists('output'):
    os.makedirs('output')

# Set random seed
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed()
device = torch.device('cpu')

class LanguageDataset(Dataset):
    def __init__(self):
        # Carefully selected distinctive words for each language
        self.english_words = [
            "the", "and", "that", "have", "for", "not", "with", "you", "this", "but",
            "his", "from", "they", "say", "her", "she", "will", "one", "all", "would",
            "there", "their", "what", "out", "about", "who", "get", "which", "when", "make",
            "can", "like", "time", "just", "him", "know", "take", "people", "into", "year",
            "your", "good", "some", "could", "them", "see", "other", "than", "then", "now",
            "look", "only", "come", "its", "over", "think", "also", "back", "after", "use",
            "two", "how", "our", "work", "first", "well", "way", "even", "new", "want",
            "because", "any", "these", "give", "day", "most", "us", "world", "life", "hand",
            "through", "while", "should", "never", "still", "find", "again", "much", "own", "must"
        ]

        self.spanish_words = [
            "el", "la", "de", "que", "y", "en", "un", "ser", "se", "no",
            "haber", "por", "con", "su", "para", "como", "estar", "tener", "más", "pero",
            "hacer", "cuando", "poder", "decir", "este", "ir", "otro", "ese", "dar", "muy",
            "deber", "bien", "donde", "todo", "parte", "vida", "ver", "tiempo", "cada", "día",
            "casa", "cosa", "año", "hombre", "forma", "mundo", "mujer", "amor", "mismo", "tres",
            "país", "ciudad", "hora", "noche", "agua", "tierra", "padre", "madre", "trabajo", "gente",
            "mano", "hijo", "amigo", "lugar", "manera", "pueblo", "luz", "nuevo", "así", "dejar",
            "poco", "grande", "tanto", "llegar", "pasar", "bueno", "desde", "saber", "sobre", "entre",
            "sentir", "pensar", "vivir", "hasta", "siempre", "momento", "algo", "volver", "poner", "ningún"
        ]

        self.french_words = [
            "le", "la", "de", "et", "un", "être", "avoir", "que", "pour", "dans",
            "ce", "qui", "ne", "sur", "se", "pas", "plus", "pouvoir", "par", "je",
            "avec", "tout", "faire", "mettre", "autre", "on", "mais", "nous", "comme", "ou",
            "si", "leur", "elle", "devoir", "avant", "deux", "même", "prendre", "aussi", "celui",
            "donner", "bien", "où", "fois", "vous", "encore", "nouveau", "aller", "cela", "entre",
            "chaque", "parler", "état", "après", "très", "dire", "jour", "sans", "homme", "femme",
            "grand", "monde", "enfant", "pays", "dernier", "main", "yeux", "lieu", "moment", "petit",
            "vie", "savoir", "rien", "voir", "rester", "toujours", "pendant", "depuis", "eau", "temps",
            "venir", "quand", "alors", "votre", "notre", "maintenant", "toute", "année", "heure", "sous"
        ]

        # Combine words and create labels
        self.words = (self.english_words + self.spanish_words + self.french_words)
        self.labels = ([0] * len(self.english_words) +
                      [1] * len(self.spanish_words) +
                      [2] * len(self.french_words))
        
        # Create character to index mapping
        all_chars = set(''.join(self.words))
        self.char_to_idx = {char: idx+1 for idx, char in enumerate(sorted(all_chars))}
        self.char_to_idx['<pad>'] = 0
        self.vocab_size = len(self.char_to_idx)
        
        # Data augmentation: add variations
        self.augment_data()
        
        # Shuffle dataset
        combined = list(zip(self.words, self.labels))
        random.shuffle(combined)
        self.words, self.labels = zip(*combined)

    def augment_data(self):
        # Add common variations and misspellings
        augmented_words = []
        augmented_labels = []
        
        for word, label in zip(self.words, self.labels):
            # Add original word
            augmented_words.append(word)
            augmented_labels.append(label)
            
            # Add capitalized version
            augmented_words.append(word.capitalize())
            augmented_labels.append(label)
            
            # Add uppercase version
            augmented_words.append(word.upper())
            augmented_labels.append(label)
            
            # Add common variations if word is long enough
            if len(word) > 3:
                # Remove last letter
                augmented_words.append(word[:-1])
                augmented_labels.append(label)
                
                # Remove first letter
                augmented_words.append(word[1:])
                augmented_labels.append(label)
        
        self.words = augmented_words
        self.labels = augmented_labels

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word = self.words[idx].lower()
        label = self.labels[idx]
        char_indices = [self.char_to_idx[c] for c in word]
        return torch.tensor(char_indices), torch.tensor(label), len(char_indices)

class ImprovedBiGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=2, dropout=0.5):
        super(ImprovedBiGRU, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, 
                         num_layers=n_layers,
                         bidirectional=True,
                         dropout=dropout if n_layers > 1 else 0,
                         batch_first=True)
        
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        
        packed_embedded = pack_padded_sequence(embedded, text_lengths,
                                             batch_first=True, enforce_sorted=False)
        
        _, hidden = self.gru(packed_embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        
        dense1 = self.fc1(hidden)
        dense1 = self.batch_norm(dense1)
        dense1 = self.relu(dense1)
        dense1 = self.dropout(dense1)
        
        return self.fc2(dense1)

def train_model():
    # Create dataset and split into train/val/test
    dataset = LanguageDataset()
    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create data loaders
    def collate_fn(batch):
        texts, labels, lengths = zip(*batch)
        texts = pad_sequence(texts, batch_first=True, padding_value=0)
        return texts, torch.tensor(labels), torch.tensor(lengths)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    # Initialize model with improved architecture
    model = ImprovedBiGRU(
        vocab_size=dataset.vocab_size,
        embedding_dim=256,
        hidden_dim=256,
        output_dim=3,
        n_layers=2,
        dropout=0.5
    ).to(device)

    # Training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                    factor=0.1, patience=5)

    # Training loop
    num_epochs = 50
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    print("Starting training...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for texts, labels, lengths in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            output = model(texts, lengths)
            loss = criterion(output, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for texts, labels, lengths in val_loader:
                texts, labels = texts.to(device), labels.to(device)
                output = model(texts, lengths)
                loss = criterion(output, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step(val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')
        print('-' * 50)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'output/best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    # Evaluate on test set
    model.load_state_dict(torch.load('output/best_model.pth'))
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for texts, labels, lengths in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts, lengths)
            _, predicted = torch.max(outputs, 1)
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_accuracy = 100 * test_correct / test_total
    print(f'\nFinal Test Accuracy: {test_accuracy:.2f}%')

    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('output/training_curves.png')
    plt.close()

    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('output/confusion_matrix.png')
    plt.close()

    # Save results
    with open('output/results.txt', 'w') as f:
        f.write("Language Classification Results\n")
        f.write("============================\n\n")
        f.write(f"Test Accuracy: {test_accuracy:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(all_labels, all_preds,
                                    target_names=['English', 'Spanish', 'French']))

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()