#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import re
from sklearn.model_selection import train_test_split
import seaborn as sns

# デバイス設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"📱 使用デバイス: {device}")

#%%

class TextPreprocessor:
    """テキスト前処理クラス（書籍の手法に準拠）"""
    
    def __init__(self, vocab_size=2000, max_length=100):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.word_to_index = {}
        self.index_to_word = {}
        
    def clean_text(self, text):
        """テキストクリーニング"""
        # 小文字化
        text = text.lower()
        # 特殊文字を除去
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # 複数スペースを単一スペースに
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def build_vocab(self, texts):
        """語彙辞書を構築（書籍の方法）"""
        # 全テキストをクリーニングしてトークン化
        all_words = []
        for text in texts:
            cleaned = self.clean_text(text)
            words = cleaned.split()
            all_words.extend(words)
        
        # 頻度カウント
        word_counts = Counter(all_words)
        
        # 特殊トークン
        self.word_to_index = {
            '<PAD>': 0,
            '<UNK>': 1,
        }
        self.index_to_word = {0: '<PAD>', 1: '<UNK>'}
        
        # 頻度の高い単語を選択
        most_common = word_counts.most_common(self.vocab_size - 2)
        
        for i, (word, count) in enumerate(most_common, 2):
            self.word_to_index[word] = i
            self.index_to_word[i] = word
        
        print(f"📚 語彙サイズ: {len(self.word_to_index)} 語")
        return self.word_to_index
    
    def texts_to_sequences(self, texts):
        """テキストを数値シーケンスに変換"""
        sequences = []
        for text in texts:
            cleaned = self.clean_text(text)
            words = cleaned.split()
            sequence = []
            for word in words:
                if word in self.word_to_index:
                    sequence.append(self.word_to_index[word])
                else:
                    sequence.append(self.word_to_index['<UNK>'])
            sequences.append(sequence)
        return sequences
    
    def pad_sequences(self, sequences):
        """シーケンスをパディング"""
        padded = []
        for seq in sequences:
            if len(seq) > self.max_length:
                # 長すぎる場合は切り詰め
                padded.append(seq[:self.max_length])
            else:
                # 短い場合はパディング
                padded.append(seq + [0] * (self.max_length - len(seq)))
        return np.array(padded)

# 書籍準拠のLSTMモデル
class TextClassificationModel(nn.Module):
    """書籍のコード例をベースにしたLSTM分類モデル"""
    
    def __init__(self, vocab_size, embedding_dim=50, hidden_dim=24, 
                 lstm_layers=1, dropout_rate=0.2, bidirectional=True):
        super(TextClassificationModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        
        # 埋め込み層
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # ドロップアウト（書籍で推奨）
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # LSTM層（双方向）
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if lstm_layers > 1 else 0
        )
        
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # グローバルプーリング（書籍のアプローチ）
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全結合層
        lstm_output_size = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc1 = nn.Linear(lstm_output_size, hidden_dim)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        # 活性化関数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 埋め込み
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = self.dropout1(embedded)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim * 2)
        lstm_out = self.dropout2(lstm_out)
        
        # グローバルプーリングのために転置
        lstm_out = lstm_out.transpose(1, 2)  # (batch_size, hidden_dim * 2, seq_len)
        
        # グローバル平均プーリング
        pooled = self.global_pool(lstm_out)  # (batch_size, hidden_dim * 2, 1)
        pooled = pooled.squeeze(-1)  # (batch_size, hidden_dim * 2)
        
        # 分類層
        x = self.relu(self.fc1(pooled))
        x = self.dropout3(x)
        x = self.sigmoid(self.fc2(x))
        
        return x

# スタック（積層）LSTM
class StackedLSTMModel(nn.Module):
    """書籍で説明されている積層LSTMモデル"""
    
    def __init__(self, vocab_size, embedding_dim=50, hidden_dim=24, 
                 lstm_layers=2, dropout_rate=0.2):
        super(StackedLSTMModel, self).__init__()
        
        # 埋め込み層
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # 第1層LSTM（双方向）
        self.lstm1 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # 第2層LSTM（双方向）
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim * 2,  # 前の層が双方向なので2倍
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # グローバルプーリング
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分類層
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 埋め込み
        embedded = self.embedding(x)
        embedded = self.dropout1(embedded)
        
        # 第1層LSTM
        lstm1_out, _ = self.lstm1(embedded)
        
        # 第2層LSTM
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.dropout2(lstm2_out)
        
        # グローバルプーリング
        lstm2_out = lstm2_out.transpose(1, 2)
        pooled = self.global_pool(lstm2_out)
        pooled = pooled.squeeze(-1)
        
        # 分類
        x = self.relu(self.fc1(pooled))
        x = self.dropout3(x)
        x = self.sigmoid(self.fc2(x))
        
        return x

class ExperimentRunner:
    """実験実行クラス"""
    
    def __init__(self):
        self.device = device
        self.results = {}
    
    def create_sample_data(self, num_samples=1000):
        """サンプルデータ作成（皮肉検出っぽいデータ）"""
        
        # ポジティブ（皮肉でない）文章例
        positive_texts = [
            "I love this beautiful sunny day",
            "This movie is absolutely amazing",
            "Great job on the presentation",
            "The food was delicious and service was excellent",
            "I'm so happy to be here with you",
            "This book is incredibly inspiring",
            "Thank you for your wonderful help",
            "The weather is perfect for a picnic",
            "I enjoy spending time with family",
            "This music makes me feel peaceful"
        ]
        
        # ネガティブ（皮肉）文章例
        negative_texts = [
            "Oh great another meeting that could have been an email",
            "I just love waiting in line for hours",
            "Perfect another traffic jam on Monday morning",
            "Wonderful my computer crashed right before the deadline",
            "Fantastic it's raining on my vacation day",
            "Amazing the elevator is broken again",
            "Brilliant forgot my umbrella on the rainiest day",
            "Excellent my phone battery died at the worst time",
            "Superb the WiFi is down during an important call",
            "Marvelous getting a parking ticket on my birthday"
        ]
        
        # データ生成
        texts = []
        labels = []
        
        for _ in range(num_samples // 2):
            # ポジティブサンプル
            base_text = np.random.choice(positive_texts)
            # 少しバリエーションを加える
            variations = ["really ", "quite ", "very ", "extremely ", ""]
            text = np.random.choice(variations) + base_text
            texts.append(text)
            labels.append(0)  # 非皮肉
            
            # ネガティブサンプル
            base_text = np.random.choice(negative_texts)
            text = base_text
            texts.append(text)
            labels.append(1)  # 皮肉
        
        return texts, labels
    
    def train_model(self, model, train_loader, val_loader, epochs=50, lr=0.001):
        """モデル訓練（書籍のアプローチ）"""
        
        model.to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            # 訓練フェーズ
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y.float())
                loss.backward()
                
                # 勾配クリッピング（書籍で言及）
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                train_correct += (predicted == batch_y.float()).sum().item()
                train_total += batch_y.size(0)
            
            # 検証フェーズ
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_x).squeeze()
                    loss = criterion(outputs, batch_y.float())
                    
                    val_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    val_correct += (predicted == batch_y.float()).sum().item()
                    val_total += batch_y.size(0)
            
            # メトリクス計算
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            # 早期停止判定
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}:')
                print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}')
                print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            if patience_counter >= patience:
                print(f"早期停止 at epoch {epoch+1}")
                break
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies
        }
    
    def run_experiments(self):
        """複数の実験を実行"""
        
        # データ準備
        print("📊 データ準備中...")
        texts, labels = self.create_sample_data(2000)
        
        # 前処理
        preprocessor = TextPreprocessor(vocab_size=2000, max_length=50)
        preprocessor.build_vocab(texts)
        sequences = preprocessor.texts_to_sequences(texts)
        padded_sequences = preprocessor.pad_sequences(sequences)
        
        # 訓練・検証分割
        X_train, X_val, y_train, y_val = train_test_split(
            padded_sequences, labels, test_size=0.2, random_state=42
        )
        
        # データローダー
        train_dataset = TensorDataset(
            torch.LongTensor(X_train), 
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.LongTensor(X_val), 
            torch.LongTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # 実験設定
        vocab_size = len(preprocessor.word_to_index)
        experiments = {
            'Basic LSTM': {
                'model': TextClassificationModel(vocab_size, hidden_dim=24),
                'lr': 0.001
            },
            'Stacked LSTM': {
                'model': StackedLSTMModel(vocab_size, hidden_dim=24),
                'lr': 0.001
            },
            'Low LR LSTM': {
                'model': TextClassificationModel(vocab_size, hidden_dim=24),
                'lr': 0.0003  # 書籍で推奨される低学習率
            },
            'High Dropout LSTM': {
                'model': TextClassificationModel(vocab_size, hidden_dim=24, dropout_rate=0.5),
                'lr': 0.001
            }
        }
        
        # 実験実行
        print("\n🚀 実験開始！")
        for name, config in experiments.items():
            print(f"\n--- {name} の訓練 ---")
            
            results = self.train_model(
                config['model'], 
                train_loader, 
                val_loader, 
                epochs=50,
                lr=config['lr']
            )
            
            self.results[name] = results
            
            # 最終性能
            final_val_acc = results['val_accuracies'][-1]
            final_val_loss = results['val_losses'][-1]
            print(f"✅ {name}: Val Acc={final_val_acc:.4f}, Val Loss={final_val_loss:.4f}")
        
        return self.results
    
    def plot_results(self):
        """結果をプロット（書籍風）"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 訓練ロス
        ax1 = axes[0, 0]
        for name, results in self.results.items():
            ax1.plot(results['train_losses'], label=name, linewidth=2)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 検証ロス
        ax2 = axes[0, 1]
        for name, results in self.results.items():
            ax2.plot(results['val_losses'], label=name, linewidth=2)
        ax2.set_title('Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 訓練精度
        ax3 = axes[1, 0]
        for name, results in self.results.items():
            ax3.plot(results['train_accuracies'], label=name, linewidth=2)
        ax3.set_title('Training Accuracy')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 検証精度
        ax4 = axes[1, 1]
        for name, results in self.results.items():
            ax4.plot(results['val_accuracies'], label=name, linewidth=2)
        ax4.set_title('Validation Accuracy')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 最終性能比較
        self.plot_final_comparison()
    
    def plot_final_comparison(self):
        """最終性能の比較プロット"""
        
        names = list(self.results.keys())
        final_val_acc = [self.results[name]['val_accuracies'][-1] for name in names]
        final_val_loss = [self.results[name]['val_losses'][-1] for name in names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 精度比較
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightyellow']
        bars1 = ax1.bar(names, final_val_acc, color=colors)
        ax1.set_title('Final Validation Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xticklabels(names, rotation=45)
        
        for bar, acc in zip(bars1, final_val_acc):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # ロス比較
        bars2 = ax2.bar(names, final_val_loss, color=colors)
        ax2.set_title('Final Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xticklabels(names, rotation=45)
        
        for bar, loss in zip(bars2, final_val_loss):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{loss:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

#%%

# サンプル推論関数
def test_model_predictions(model, preprocessor):
    """モデルのテスト推論（書籍風）"""
    
    model.eval()
    
    # テスト文章（書籍の例を参考）
    test_sentences = [
        "I love this beautiful sunny day",  # 非皮肉
        "Oh great another meeting",         # 皮肉
        "Thank you for your help",          # 非皮肉
        "Perfect it's raining on my vacation",  # 皮肉
        "This book is amazing",             # 非皮肉
        "Wonderful my computer crashed"     # 皮肉
    ]
    
    print("\n🧪 モデル推論テスト:")
    print("=" * 50)
    
    for sentence in test_sentences:
        # 前処理
        sequence = preprocessor.texts_to_sequences([sentence])
        padded = preprocessor.pad_sequences(sequence)
        
        # 予測
        with torch.no_grad():
            input_tensor = torch.LongTensor(padded).to(device)
            prediction = model(input_tensor).item()
            
        # 結果表示
        sentiment = "皮肉" if prediction > 0.5 else "非皮肉"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        print(f"文章: '{sentence}'")
        print(f"予測: {sentiment} (信頼度: {confidence:.3f})")
        print("-" * 30)

# メイン実行
def main():
    """メイン実行関数"""
    
    print("📖 PyTorch LSTM感情分析実験（書籍準拠版）")
    print("=" * 60)
    
    # 実験実行
    runner = ExperimentRunner()
    results = runner.run_experiments()
    
    # 結果プロット
    runner.plot_results()
    
    # 書籍風の分析表示
    print("\n📊 実験結果分析:")
    print("=" * 50)
    
    for name, result in results.items():
        final_train_acc = result['train_accuracies'][-1]
        final_val_acc = result['val_accuracies'][-1]
        overfitting = final_train_acc - final_val_acc
        
        print(f"{name}:")
        print(f"  訓練精度: {final_train_acc:.4f}")
        print(f"  検証精度: {final_val_acc:.4f}")
        print(f"  オーバーフィット度: {overfitting:.4f}")
        print()
    
    # ベストモデルでテスト推論
    best_model_name = max(results.keys(), 
                         key=lambda x: results[x]['val_accuracies'][-1])
    print(f"🏆 最高性能モデル: {best_model_name}")
    
    # 簡単なテスト用のモデルとプリプロセッサを作成
    texts, labels = runner.create_sample_data(100)
    preprocessor = TextPreprocessor(vocab_size=1000, max_length=50)
    preprocessor.build_vocab(texts)
    
    test_model = TextClassificationModel(len(preprocessor.word_to_index))
    test_model_predictions(test_model, preprocessor)


#%%

if __name__ == "__main__":
    main()