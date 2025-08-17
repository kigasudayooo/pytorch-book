"""
🌍 実世界応用LSTM実装集
書籍「PyTorch by Example」第7章の発展版

実際のプロダクションレベルの実装例：
1. 皮肉検出APIサーバー
2. リアルタイム感情分析システム
3. 多言語対応テキスト分類
4. A/Bテスト用モデル比較フレームワーク
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict
import json
import time
from typing import List, Dict, Tuple, Optional
import pickle
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Production-Ready LSTM Model（本格的な実装）
# =============================================================================

class ProductionLSTM(nn.Module):
    """プロダクション対応LSTM（書籍の知見を全て適用）"""
    
    def __init__(self, 
                 vocab_size: int,
                 embedding_dim: int = 100,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 bidirectional: bool = True,
                 pretrained_embeddings: Optional[torch.Tensor] = None,
                 freeze_embeddings: bool = False,
                 num_classes: int = 2):
        super(ProductionLSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_classes = num_classes
        
        # 埋め込み層（書籍の転移学習準拠）
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        if pretrained_embeddings is not None:
            print("🔄 事前学習済み埋め込みをロード中...")
            self.embedding.weight.data.copy_(pretrained_embeddings)
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False
                print("🔒 埋め込み層をフリーズしました")
        
        # ドロップアウト
        self.embedding_dropout = nn.Dropout(dropout)
        
        # LSTM層（書籍の積層手法準拠）
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention層（書籍で示唆された発展技術）
        lstm_output_size = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 分類器（書籍の構造を発展）
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # 重み初期化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """重みの初期化（安定した学習のため）"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x, lengths=None):
        # 埋め込み
        embedded = self.embedding(x)
        embedded = self.embedding_dropout(embedded)
        
        # LSTM
        if lengths is not None:
            # パディングを考慮した効率的な処理
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, (hidden, cell) = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Self-Attention
        attended_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global Max Pooling（グローバルプーリングの改良版）
        pooled, _ = torch.max(attended_out, dim=1)
        
        # 分類
        logits = self.classifier(pooled)
        
        if self.num_classes == 2:
            return torch.sigmoid(logits[:, 0])  # 二項分類
        else:
            return torch.softmax(logits, dim=1)  # 多項分類

# =============================================================================
# Advanced Text Preprocessor（高度な前処理）
# =============================================================================

class AdvancedTextPreprocessor:
    """高度なテキスト前処理（実世界対応）"""
    
    def __init__(self, 
                 vocab_size: int = 10000,
                 max_length: int = 128,
                 min_freq: int = 2,
                 special_tokens: List[str] = None):
        
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.min_freq = min_freq
        
        # 特殊トークン
        if special_tokens is None:
            special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
        
        self.special_tokens = special_tokens
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_freq = defaultdict(int)
        
    def build_vocab(self, texts: List[str]) -> Dict[str, int]:
        """語彙構築（書籍の方法を拡張）"""
        
        print("📚 語彙構築中...")
        
        # 特殊トークンを先に追加
        for i, token in enumerate(self.special_tokens):
            self.word_to_idx[token] = i
            self.idx_to_word[i] = token
        
        # 頻度カウント
        for text in texts:
            words = self._tokenize(text)
            for word in words:
                self.word_freq[word] += 1
        
        # 頻度でフィルタリング
        valid_words = [
            word for word, freq in self.word_freq.items() 
            if freq >= self.min_freq
        ]
        
        # 頻度順にソート
        valid_words.sort(key=lambda x: self.word_freq[x], reverse=True)
        
        # 語彙サイズ制限
        vocab_words = valid_words[:self.vocab_size - len(self.special_tokens)]
        
        # インデックス割り当て
        for word in vocab_words:
            idx = len(self.word_to_idx)
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
        
        print(f"✅ 語彙構築完了: {len(self.word_to_idx)} 語")
        print(f"   - 出現頻度 >= {self.min_freq} の語: {len(valid_words)}")
        print(f"   - 実際の語彙サイズ: {len(vocab_words)}")
        
        return self.word_to_idx
    
    def _tokenize(self, text: str) -> List[str]:
        """トークン化（実世界対応の改良版）"""
        import re
        
        # 小文字化
        text = text.lower()
        
        # URL、メンション、ハッシュタグの正規化
        text = re.sub(r'http\S+|www\S+', '<URL>', text)
        text = re.sub(r'@\w+', '<MENTION>', text)
        text = re.sub(r'#\w+', '<HASHTAG>', text)
        
        # 絵文字と特殊文字の処理
        text = re.sub(r'[^\w\s<>]', ' ', text)
        
        # トークン分割
        tokens = text.split()
        
        return tokens
    
    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """テキストを数値シーケンスに変換"""
        
        sequences = []
        unk_idx = self.word_to_idx.get('<UNK>', 1)
        
        for text in texts:
            tokens = self._tokenize(text)
            sequence = [
                self.word_to_idx.get(token, unk_idx) 
                for token in tokens
            ]
            sequences.append(sequence)
        
        return sequences
    
    def pad_sequences(self, sequences: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
        """シーケンスパディング（長さ情報も返す）"""
        
        lengths = [min(len(seq), self.max_length) for seq in sequences]
        padded = np.zeros((len(sequences), self.max_length), dtype=np.int64)
        
        for i, seq in enumerate(sequences):
            length = lengths[i]
            padded[i, :length] = seq[:length]
        
        return padded, np.array(lengths)

# =============================================================================
# Real-world Dataset Simulation（実世界データセットシミュレーション）
# =============================================================================

class RealWorldDatasetSimulator:
    """実世界のデータセットをシミュレート"""
    
    def __init__(self):
        self.sarcasm_patterns = [
            # 皮肉パターン（書籍の例を拡張）
            "oh {adjective} another {noun}",
            "just {adjective} I needed more {noun}",
            "{adjective} timing for {noun}",
            "I {adverb} love {noun} in the {time}",
            "nothing like {noun} to make my {time}",
            "because {noun} always makes everything {adjective}",
            "sure let me just {verb} this {noun}",
            "I was {adverb} hoping for more {noun}"
        ]
        
        self.positive_patterns = [
            "I {adverb} love this {adjective} {noun}",
            "this {noun} is {adjective}",
            "what a {adjective} {noun}",
            "I am {adjective} about this {noun}",
            "this {noun} brings me {emotion}",
            "I {verb} this {adjective} {noun}",
            "such a {adjective} {time} for {noun}",
            "I feel {emotion} about this {noun}"
        ]
        
        # 語彙データベース
        self.vocab_db = {
            'adjective': ['great', 'perfect', 'wonderful', 'terrible', 'awful', 'amazing', 'brilliant', 'fantastic'],
            'noun': ['day', 'weather', 'meeting', 'traffic', 'news', 'food', 'movie', 'book', 'music', 'work'],
            'adverb': ['really', 'absolutely', 'completely', 'totally', 'definitely', 'certainly'],
            'verb': ['enjoy', 'appreciate', 'understand', 'expect', 'welcome'],
            'emotion': ['joy', 'happiness', 'satisfaction', 'peace', 'excitement'],
            'time': ['morning', 'evening', 'day', 'week', 'moment', 'time']
        }
    
    def generate_sarcasm_dataset(self, num_samples: int = 5000) -> Tuple[List[str], List[int]]:
        """皮肉検出データセット生成（書籍風）"""
        
        texts = []
        labels = []
        
        for _ in range(num_samples // 2):
            # 皮肉な文章
            pattern = np.random.choice(self.sarcasm_patterns)
            text = self._fill_pattern(pattern)
            texts.append(text)
            labels.append(1)  # 皮肉
            
            # 非皮肉な文章
            pattern = np.random.choice(self.positive_patterns)
            text = self._fill_pattern(pattern)
            texts.append(text)
            labels.append(0)  # 非皮肉
        
        # シャッフル
        combined = list(zip(texts, labels))
        np.random.shuffle(combined)
        texts, labels = zip(*combined)
        
        return list(texts), list(labels)
    
    def generate_multilabel_dataset(self, num_samples: int = 3000) -> Tuple[List[str], List[List[int]]]:
        """多ラベル分類データセット（実世界っぽく）"""
        
        # 感情ラベル：[怒り, 喜び, 悲しみ, 恐怖, 驚き]
        emotion_patterns = {
            'anger': ["I hate {noun}", "this {noun} is terrible", "why is this {noun} so {adjective}"],
            'joy': ["I love {noun}", "this {noun} is amazing", "what a wonderful {noun}"],
            'sadness': ["I feel sad about {noun}", "this {noun} makes me cry", "such a disappointing {noun}"],
            'fear': ["I'm scared of {noun}", "this {noun} is frightening", "I worry about {noun}"],
            'surprise': ["wow this {noun}", "I can't believe this {noun}", "such an unexpected {noun}"]
        }
        
        texts = []
        labels = []
        
        for _ in range(num_samples):
            # ランダムに1-3個の感情を選択
            num_emotions = np.random.randint(1, 4)
            selected_emotions = np.random.choice(
                list(emotion_patterns.keys()), 
                size=num_emotions, 
                replace=False
            )
            
            # 文章生成
            sentences = []
            for emotion in selected_emotions:
                pattern = np.random.choice(emotion_patterns[emotion])
                sentence = self._fill_pattern(pattern)
                sentences.append(sentence)
            
            text = '. '.join(sentences)
            texts.append(text)
            
            # ラベル作成
            label = [0, 0, 0, 0, 0]  # [怒り, 喜び, 悲しみ, 恐怖, 驚き]
            emotion_to_idx = {'anger': 0, 'joy': 1, 'sadness': 2, 'fear': 3, 'surprise': 4}
            for emotion in selected_emotions:
                label[emotion_to_idx[emotion]] = 1
            
            labels.append(label)
        
        return texts, labels
    
    def _fill_pattern(self, pattern: str) -> str:
        """パターンに語彙を当てはめる"""
        
        result = pattern
        for pos_type in self.vocab_db:
            if f'{{{pos_type}}}' in result:
                word = np.random.choice(self.vocab_db[pos_type])
                result = result.replace(f'{{{pos_type}}}', word)
        
        return result

# =============================================================================
# Advanced Training Framework（高度な訓練フレームワーク）
# =============================================================================

class AdvancedTrainer:
    """高度な訓練フレームワーク（実世界対応）"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        self.train_history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'learning_rates': []
        }
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 50,
              lr: float = 0.001,
              weight_decay: float = 1e-5,
              lr_scheduler: str = 'cosine',
              early_stopping_patience: int = 10,
              gradient_clip_norm: float = 1.0):
        """高度な訓練ループ"""
        
        print(f"🚀 高度な訓練開始 (エポック数: {epochs})")
        
        # オプティマイザ（書籍より高度）
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 学習率スケジューラ
        if lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=lr*0.1
            )
        elif lr_scheduler == 'reduce':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )
        else:
            scheduler = None
        
        # 損失関数
        if self.model.num_classes == 2:
            criterion = nn.BCELoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        # 早期停止
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 訓練フェーズ
            train_metrics = self._train_epoch(
                train_loader, optimizer, criterion, gradient_clip_norm
            )
            
            # 検証フェーズ
            val_metrics = self._validate_epoch(val_loader, criterion)
            
            # 履歴更新
            self.train_history['train_loss'].append(train_metrics['loss'])
            self.train_history['train_acc'].append(train_metrics['accuracy'])
            self.train_history['val_loss'].append(val_metrics['loss'])
            self.train_history['val_acc'].append(val_metrics['accuracy'])
            self.train_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # 学習率更新
            if scheduler:
                if lr_scheduler == 'reduce':
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step()
            
            # 進捗表示
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}:")
                print(f"  Train: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']:.4f}")
                print(f"  Val:   Loss={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']:.4f}")
                print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 早期停止判定
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                self._save_best_model()
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"早期停止 at epoch {epoch+1}")
                break
        
        print("✅ 訓練完了！")
        return self.train_history
    
    def _train_epoch(self, train_loader, optimizer, criterion, grad_clip):
        """1エポックの訓練"""
        
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y, lengths in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            lengths = lengths.to(self.device)
            
            optimizer.zero_grad()
            
            # フォワードパス
            outputs = self.model(batch_x, lengths)
            
            # 損失計算
            if self.model.num_classes == 2:
                loss = criterion(outputs, batch_y.float())
                predicted = (outputs > 0.5).float()
                correct += (predicted == batch_y.float()).sum().item()
            else:
                loss = criterion(outputs, batch_y.long())
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_y).sum().item()
            
            # バックワード
            loss.backward()
            
            # 勾配クリッピング
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            
            optimizer.step()
            
            total_loss += loss.item()
            total += batch_y.size(0)
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': correct / total
        }
    
    def _validate_epoch(self, val_loader, criterion):
        """1エポックの検証"""
        
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y, lengths in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                lengths = lengths.to(self.device)
                
                outputs = self.model(batch_x, lengths)
                
                if self.model.num_classes == 2:
                    loss = criterion(outputs, batch_y.float())
                    predicted = (outputs > 0.5).float()
                    correct += (predicted == batch_y.float()).sum().item()
                else:
                    loss = criterion(outputs, batch_y.long())
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == batch_y).sum().item()
                
                total_loss += loss.item()
                total += batch_y.size(0)
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': correct / total
        }
    
    def _save_best_model(self):
        """ベストモデルの保存"""
        torch.save(self.model.state_dict(), 'best_model.pth')
    
    def plot_training_history(self):
        """訓練履歴の可視化（書籍風改良版）"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 損失
        axes[0, 0].plot(self.train_history['train_loss'], label='Train', linewidth=2)
        axes[0, 0].plot(self.train_history['val_loss'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Loss Progress (書籍 Fig 7-10, 7-12 風)')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 精度
        axes[0, 1].plot(self.train_history['train_acc'], label='Train', linewidth=2)
        axes[0, 1].plot(self.train_history['val_acc'], label='Validation', linewidth=2)
        axes[0, 1].set_title('Accuracy Progress (書籍 Fig 7-9, 7-11 風)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 学習率
        axes[1, 0].plot(self.train_history['learning_rates'], linewidth=2, color='orange')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True, alpha=0.3)
        
        # オーバーフィッティング分析
        overfitting = [
            train - val for train, val in 
            zip(self.train_history['train_acc'], self.train_history['val_acc'])
        ]
        axes[1, 1].plot(overfitting, linewidth=2, color='red')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Overfitting Analysis (Train - Val Acc)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy Difference')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# =============================================================================
# Model Evaluation & Analysis（詳細評価・分析）
# =============================================================================

class ModelAnalyzer:
    """モデルの詳細分析（実世界対応）"""
    
    def __init__(self, model, preprocessor, device='cpu'):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        
    def comprehensive_evaluation(self, test_loader):
        """包括的評価"""
        
        print("🔍 包括的モデル評価開始...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_x, batch_y, lengths in test_loader:
                batch_x = batch_x.to(self.device)
                lengths = lengths.to(self.device)
                
                outputs = self.model(batch_x, lengths)
                
                if self.model.num_classes == 2:
                    predictions = (outputs > 0.5).cpu().numpy()
                    probabilities = outputs.cpu().numpy()
                else:
                    _, predictions = torch.max(outputs, 1)
                    predictions = predictions.cpu().numpy()
                    probabilities = outputs.cpu().numpy()
                
                all_predictions.extend(predictions)
                all_labels.extend(batch_y.numpy())
                all_probabilities.extend(probabilities)
        
        # 分類レポート
        if self.model.num_classes == 2:
            target_names = ['Non-Sarcastic', 'Sarcastic']
        else:
            target_names = [f'Class_{i}' for i in range(self.model.num_classes)]
        
        report = classification_report(
            all_labels, all_predictions, 
            target_names=target_names,
            output_dict=True
        )
        
        print("📊 分類レポート:")
        print(classification_report(all_labels, all_predictions, target_names=target_names))
        
        # 混同行列
        self._plot_confusion_matrix(all_labels, all_predictions, target_names)
        
        return {
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities,
            'classification_report': report
        }
    
    def _plot_confusion_matrix(self, y_true, y_pred, target_names):
        """混同行列の可視化"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    def analyze_predictions(self, texts, true_labels, predictions, probabilities):
        """予測の詳細分析"""
        
        print("\n🔎 予測分析:")
        print("=" * 50)
        
        # 高信頼度で正解した例
        correct_high_conf = []
        incorrect_high_conf = []
        
        for i, (text, true, pred, prob) in enumerate(zip(texts, true_labels, predictions, probabilities)):
            confidence = max(prob, 1-prob) if self.model.num_classes == 2 else max(prob)
            
            if true == pred and confidence > 0.8:
                correct_high_conf.append((text, true, pred, confidence))
            elif true != pred and confidence > 0.8:
                incorrect_high_conf.append((text, true, pred, confidence))
        
        print("✅ 高信頼度で正解した例（上位5個）:")
        for text, true, pred, conf in correct_high_conf[:5]:
            print(f"  '{text}' | 正解: {true}, 予測: {pred}, 信頼度: {conf:.3f}")
        
        print("\n❌ 高信頼度で間違えた例（上位5個）:")
        for text, true, pred, conf in incorrect_high_conf[:5]:
            print(f"  '{text}' | 正解: {true}, 予測: {pred}, 信頼度: {conf:.3f}")

# =============================================================================
# Production API Server（本格的なAPIサーバー）
# =============================================================================

class SarcasmDetectionAPI:
    """皮肉検出API（書籍の応用例）"""
    
    def __init__(self, model_path: str, preprocessor_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # モデルとプリプロセッサーの読み込み
        self.preprocessor = self._load_preprocessor(preprocessor_path)
        self.model = self._load_model(model_path)
        
        print(f"🚀 皮肉検出API起動完了！（デバイス: {self.device}）")
    
    def _load_model(self, model_path: str):
        """モデル読み込み"""
        # 実装では実際のモデルパラメータから読み込み
        vocab_size = len(self.preprocessor.word_to_idx)
        model = ProductionLSTM(vocab_size=vocab_size)
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("✅ 訓練済みモデルを読み込みました")
        except:
            print("⚠️ 新規モデルを初期化しました")
        
        model.to(self.device)
        model.eval()
        return model
    
    def _load_preprocessor(self, preprocessor_path: str):
        """プリプロセッサー読み込み"""
        try:
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
            print("✅ プリプロセッサーを読み込みました")
        except:
            print("⚠️ 新規プリプロセッサーを初期化しました")
            preprocessor = AdvancedTextPreprocessor()
        
        return preprocessor
    
    def predict_single(self, text: str) -> Dict:
        """単一テキストの皮肉予測（書籍のテスト例準拠）"""
        
        start_time = time.time()
        
        # 前処理
        sequences = self.preprocessor.texts_to_sequences([text])
        padded, lengths = self.preprocessor.pad_sequences(sequences)
        
        # 推論
        with torch.no_grad():
            input_tensor = torch.LongTensor(padded).to(self.device)
            length_tensor = torch.LongTensor(lengths).to(self.device)
            
            probability = self.model(input_tensor, length_tensor).item()
        
        # 結果整理
        is_sarcastic = probability > 0.5
        confidence = probability if is_sarcastic else 1 - probability
        
        inference_time = time.time() - start_time
        
        return {
            'text': text,
            'is_sarcastic': is_sarcastic,
            'sarcasm_probability': probability,
            'confidence': confidence,
            'inference_time_ms': inference_time * 1000,
            'status': 'success'
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """バッチ予測"""
        
        start_time = time.time()
        
        # 前処理
        sequences = self.preprocessor.texts_to_sequences(texts)
        padded, lengths = self.preprocessor.pad_sequences(sequences)
        
        # 推論
        with torch.no_grad():
            input_tensor = torch.LongTensor(padded).to(self.device)
            length_tensor = torch.LongTensor(lengths).to(self.device)
            
            probabilities = self.model(input_tensor, length_tensor).cpu().numpy()
        
        # 結果整理
        results = []
        for text, prob in zip(texts, probabilities):
            is_sarcastic = prob > 0.5
            confidence = prob if is_sarcastic else 1 - prob
            
            results.append({
                'text': text,
                'is_sarcastic': bool(is_sarcastic),
                'sarcasm_probability': float(prob),
                'confidence': float(confidence)
            })
        
        batch_time = time.time() - start_time
        
        return {
            'results': results,
            'batch_size': len(texts),
            'total_time_ms': batch_time * 1000,
            'avg_time_per_text_ms': (batch_time * 1000) / len(texts),
            'status': 'success'
        }

# =============================================================================
# A/B Testing Framework（A/Bテストフレームワーク）
# =============================================================================

class ModelABTesting:
    """モデルのA/Bテストフレームワーク"""
    
    def __init__(self):
        self.experiments = {}
        
    def register_model(self, name: str, model, preprocessor):
        """モデルを登録"""
        self.experiments[name] = {
            'model': model,
            'preprocessor': preprocessor,
            'results': []
        }
        print(f"📝 モデル '{name}' を登録しました")
    
    def run_ab_test(self, test_texts: List[str], test_labels: List[int], 
                    metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1']):
        """A/Bテスト実行"""
        
        print("🧪 A/Bテスト開始...")
        
        results = {}
        
        for name, experiment in self.experiments.items():
            print(f"\n--- モデル {name} の評価 ---")
            
            model = experiment['model']
            preprocessor = experiment['preprocessor']
            
            # 前処理
            sequences = preprocessor.texts_to_sequences(test_texts)
            padded, lengths = preprocessor.pad_sequences(sequences)
            
            # 予測
            model.eval()
            predictions = []
            probabilities = []
            
            with torch.no_grad():
                # バッチサイズを制限して処理
                batch_size = 32
                for i in range(0, len(padded), batch_size):
                    batch_x = torch.LongTensor(padded[i:i+batch_size])
                    batch_lengths = torch.LongTensor(lengths[i:i+batch_size])
                    
                    outputs = model(batch_x, batch_lengths)
                    
                    if model.num_classes == 2:
                        batch_preds = (outputs > 0.5).cpu().numpy()
                        batch_probs = outputs.cpu().numpy()
                    else:
                        _, batch_preds = torch.max(outputs, 1)
                        batch_preds = batch_preds.cpu().numpy()
                        batch_probs = outputs.cpu().numpy()
                    
                    predictions.extend(batch_preds)
                    probabilities.extend(batch_probs)
            
            # メトリクス計算
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            model_results = {}
            
            if 'accuracy' in metrics:
                model_results['accuracy'] = accuracy_score(test_labels, predictions)
            
            if 'precision' in metrics:
                model_results['precision'] = precision_score(
                    test_labels, predictions, average='weighted'
                )
            
            if 'recall' in metrics:
                model_results['recall'] = recall_score(
                    test_labels, predictions, average='weighted'
                )
            
            if 'f1' in metrics:
                model_results['f1'] = f1_score(
                    test_labels, predictions, average='weighted'
                )
            
            results[name] = model_results
            
            # 結果表示
            for metric, value in model_results.items():
                print(f"  {metric.capitalize()}: {value:.4f}")
        
        # 比較表示
        self._display_comparison(results, metrics)
        
        return results
    
    def _display_comparison(self, results: Dict, metrics: List[str]):
        """結果比較の表示"""
        
        print("\n📊 A/Bテスト結果比較:")
        print("=" * 60)
        
        # 表形式で表示
        model_names = list(results.keys())
        
        print(f"{'メトリクス':<12}", end="")
        for name in model_names:
            print(f"{name:<15}", end="")
        print()
        print("-" * 60)
        
        for metric in metrics:
            print(f"{metric.capitalize():<12}", end="")
            values = []
            for name in model_names:
                value = results[name][metric]
                values.append(value)
                print(f"{value:<15.4f}", end="")
            print()
            
            # 最高性能のモデルを強調
            best_idx = np.argmax(values)
            print(f"{'最高性能:':<12}{model_names[best_idx]}")
            print()
        
        # 総合評価
        print("🏆 総合評価:")
        total_scores = {}
        for name in model_names:
            total_score = sum(results[name][metric] for metric in metrics)
            total_scores[name] = total_score
            print(f"  {name}: {total_score:.4f}")
        
        winner = max(total_scores.keys(), key=lambda x: total_scores[x])
        print(f"\n🥇 勝者: {winner}")

# =============================================================================
# Comprehensive Example Runner（包括的な実行例）
# =============================================================================

def run_comprehensive_example():
    """包括的な実行例（書籍の全内容を統合）"""
    
    print("🌟 実世界応用LSTM包括実験")
    print("書籍「PyTorch by Example」第7章の実践版")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Step 1: データ準備
    print("\n📊 Step 1: 実世界風データセット作成")
    simulator = RealWorldDatasetSimulator()
    texts, labels = simulator.generate_sarcasm_dataset(3000)
    
    print(f"  - データサンプル数: {len(texts)}")
    print(f"  - 皮肉の割合: {np.mean(labels):.2%}")
    
    # データサンプル表示
    print("\n📝 データサンプル例:")
    for i in range(5):
        label_text = "皮肉" if labels[i] else "非皮肉"
        print(f"  [{label_text}] {texts[i]}")
    
    # Step 2: 前処理
    print("\n🔧 Step 2: 高度な前処理")
    preprocessor = AdvancedTextPreprocessor(vocab_size=5000, max_length=64)
    preprocessor.build_vocab(texts)
    
    sequences = preprocessor.texts_to_sequences(texts)
    padded, lengths = preprocessor.pad_sequences(sequences)
    
    # 訓練/検証/テスト分割
    from sklearn.model_selection import train_test_split
    
    X_temp, X_test, y_temp, y_test, len_temp, len_test = train_test_split(
        padded, labels, lengths, test_size=0.2, random_state=42, stratify=labels
    )
    
    X_train, X_val, y_train, y_val, len_train, len_val = train_test_split(
        X_temp, y_temp, len_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"  - 訓練セット: {len(X_train)} samples")
    print(f"  - 検証セット: {len(X_val)} samples")
    print(f"  - テストセット: {len(X_test)} samples")
    
    # データローダー作成
    def create_dataloader(X, y, lengths, batch_size=32, shuffle=True):
        dataset = TensorDataset(
            torch.LongTensor(X),
            torch.LongTensor(y),
            torch.LongTensor(lengths)
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    train_loader = create_dataloader(X_train, y_train, len_train, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, len_val, shuffle=False)
    test_loader = create_dataloader(X_test, y_test, len_test, shuffle=False)
    
    # Step 3: モデル作成と訓練
    print("\n🧠 Step 3: 高度なモデル訓練")
    
    vocab_size = len(preprocessor.word_to_idx)
    
    # モデル1: 基本モデル
    model1 = ProductionLSTM(
        vocab_size=vocab_size,
        embedding_dim=64,
        hidden_dim=32,
        num_layers=1,
        bidirectional=True,
        dropout=0.2
    )
    
    trainer1 = AdvancedTrainer(model1, device)
    print("\n--- 基本モデルの訓練 ---")
    history1 = trainer1.train(
        train_loader, val_loader,
        epochs=30, lr=0.001,
        lr_scheduler='cosine',
        early_stopping_patience=8
    )
    
    # モデル2: 高度なモデル
    model2 = ProductionLSTM(
        vocab_size=vocab_size,
        embedding_dim=100,
        hidden_dim=64,
        num_layers=2,
        bidirectional=True,
        dropout=0.3
    )
    
    trainer2 = AdvancedTrainer(model2, device)
    print("\n--- 高度なモデルの訓練 ---")
    history2 = trainer2.train(
        train_loader, val_loader,
        epochs=30, lr=0.0005,
        lr_scheduler='reduce',
        early_stopping_patience=10
    )
    
    # Step 4: 訓練結果の可視化
    print("\n📈 Step 4: 訓練結果可視化")
    trainer1.plot_training_history()
    trainer2.plot_training_history()
    
    # Step 5: モデル比較（A/Bテスト）
    print("\n🆚 Step 5: A/Bテスト")
    ab_tester = ModelABTesting()
    ab_tester.register_model("Basic LSTM", model1, preprocessor)
    ab_tester.register_model("Advanced LSTM", model2, preprocessor)
    
    # テストデータから一部を使用
    test_texts_sample = [texts[i] for i in range(len(X_test))[:200]]
    test_labels_sample = y_test[:200].tolist()
    
    ab_results = ab_tester.run_ab_test(
        test_texts_sample, test_labels_sample,
        metrics=['accuracy', 'precision', 'recall', 'f1']
    )
    
    # Step 6: 詳細分析
    print("\n🔍 Step 6: 詳細モデル分析")
    analyzer = ModelAnalyzer(model2, preprocessor, device)  # 高度なモデルを使用
    eval_results = analyzer.comprehensive_evaluation(test_loader)
    
    # 予測例の詳細分析
    sample_texts = test_texts_sample[:20]
    sample_labels = test_labels_sample[:20]
    sample_predictions = eval_results['predictions'][:20]
    sample_probabilities = eval_results['probabilities'][:20]
    
    analyzer.analyze_predictions(
        sample_texts, sample_labels, 
        sample_predictions, sample_probabilities
    )
    
    # Step 7: API実装デモ
    print("\n🚀 Step 7: APIデモ")
    
    # モデルとプリプロセッサーを保存（実際の実装では）
    api = SarcasmDetectionAPI("dummy_model.pth", "dummy_preprocessor.pkl")
    api.model = model2  # デモ用に直接設定
    api.preprocessor = preprocessor
    
    # 書籍の例に基づくテスト
    test_sentences = [
        "Oh great another meeting that could have been an email",
        "I love this beautiful sunny weather",
        "Perfect timing for my computer to crash",
        "Thank you for your wonderful help",
        "Fantastic another traffic jam on Monday morning",
        "This book is absolutely amazing"
    ]
    
    print("\n🧪 皮肉検出APIテスト（書籍風）:")
    print("-" * 60)
    
    for sentence in test_sentences:
        result = api.predict_single(sentence)
        
        status = "🙄 皮肉" if result['is_sarcastic'] else "😊 真面目"
        print(f"{status} '{sentence}'")
        print(f"   皮肉度: {result['sarcasm_probability']:.1%}")
        print(f"   信頼度: {result['confidence']:.1%}")
        print(f"   処理時間: {result['inference_time_ms']:.1f}ms")
        print()
    
    # バッチ予測デモ
    batch_result = api.predict_batch(test_sentences)
    print(f"📦 バッチ処理結果:")
    print(f"   総処理時間: {batch_result['total_time_ms']:.1f}ms")
    print(f"   1件あたり平均: {batch_result['avg_time_per_text_ms']:.1f}ms")
    
    print("\n🎉 包括実験完了！")
    print("=" * 50)
    print("実装した機能:")
    print("✅ 高度なテキスト前処理")
    print("✅ プロダクションレベルLSTM")
    print("✅ 高度な訓練フレームワーク")
    print("✅ A/Bテストフレームワーク")
    print("✅ 詳細モデル分析")
    print("✅ 皮肉検出API")
    print("✅ 書籍の全知見を統合")
    
    return {
        'models': {'basic': model1, 'advanced': model2},
        'preprocessor': preprocessor,
        'ab_results': ab_results,
        'api': api
    }

# =============================================================================
# 学習課題とエクササイズ
# =============================================================================

def learning_exercises():
    """学習者向けの実践課題"""
    
    print("\n📚 実践学習課題")
    print("=" * 50)
    
    exercises = [
        {
            'title': '🔥 Exercise 1: カスタム損失関数',
            'description': 'Focal Lossを実装して不均衡データに対応してみよう',
            'difficulty': '★★☆',
            'hints': [
                'nn.Moduleを継承してカスタム損失関数を作成',
                'alpha（クラス重み）とgamma（難易度重み）パラメータを設定',
                '簡単なサンプルほど損失を小さくする仕組み'
            ]
        },
        {
            'title': '🔥 Exercise 2: アテンション可視化',
            'description': 'どの単語に注目してるかを可視化してみよう',
            'difficulty': '★★★',
            'hints': [
                'MultiheadAttentionの出力からattention_weightsを取得',
                'matplotlib/seabornでヒートマップ作成',
                '文章と注目度の対応関係を明確に'
            ]
        },
        {
            'title': '🔥 Exercise 3: ドメイン適応',
            'description': '異なるドメイン（ニュース→レビュー）での転移学習',
            'difficulty': '★★★',
            'hints': [
                'ソースドメインで事前訓練',
                'ターゲットドメインで微調整',
                'ドメイン間の性能差を分析'
            ]
        },
        {
            'title': '🔥 Exercise 4: リアルタイム推論最適化',
            'description': 'ONNX変換とTensorRT最適化で高速化',
            'difficulty': '★★★★',
            'hints': [
                'torch.onnx.exportでONNX形式に変換',
                'バッチサイズと系列長を固定',
                '推論時間とメモリ使用量を測定'
            ]
        }
    ]
    
    for i, exercise in enumerate(exercises, 1):
        print(f"\n{exercise['title']}")
        print(f"難易度: {exercise['difficulty']}")
        print(f"説明: {exercise['description']}")
        print("ヒント:")
        for hint in exercise['hints']:
            print(f"  • {hint}")
    
    print("\n💡 さらなる発展:")
    print("• BERT/RoBERTaとの性能比較")
    print("• マルチタスク学習（感情分析 + 皮肉検出）")
    print("• 説明可能AI（LIME/SHAP）の適用")
    print("• Docker化とクラウドデプロイ")
    print("• GraphQLベースのAPI設計")

if __name__ == "__main__":
    # 包括実験の実行
    results = run_comprehensive_example()
    
    # 学習課題の表示
    learning_exercises()