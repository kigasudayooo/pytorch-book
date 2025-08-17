#%%
import japanize_matplotlib
# 例：単語の順序によって意味が変わる
sentence1 = ["blue", "sky", "beautiful"]  # 美しい青空
sentence2 = ["sky", "blue", "beautiful"]  # 空が青くて美しい
sentence3 = ["beautiful", "blue", "sky"]  # 美しい青い空

# 単語は同じでも、順序によってニュアンスが変わる
# %%
import numpy as np
import matplotlib.pyplot as plt
# 簡単なRNNセルの実装
class SimpleRNN:
    def __init__(self, input_size, hidden_size, seed=111):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # seedを固定
        np.random.seed(seed)
        
        # 重み行列の初期化
        self.Wxh = np.random.randn(input_size, hidden_size) * 0.1   # 入力→隠れ層
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.1  # 隠れ層→隠れ層
        self.bh = np.zeros((1, hidden_size))                        # バイアス
        
    def forward(self, x, h_prev):
        """
        x: 現在の入力 (batch_size, input_size)
        h_prev: 前の隠れ状態 (batch_size, hidden_size)
        """
        h_next = np.tanh(np.dot(x, self.Wxh) + np.dot(h_prev, self.Whh) + self.bh)
        return h_next

# 実際に動かしてみる
rnn = SimpleRNN(input_size=3, hidden_size=4)
x1 = np.array([[1, 0, 0]])  # "I"
x2 = np.array([[0, 1, 0]])  # "love"
x3 = np.array([[0, 0, 1]])  # "you"

h0 = np.zeros((1, 4))  # 初期状態
h1 = rnn.forward(x1, h0)
h2 = rnn.forward(x2, h1)  # h1が h2の計算に影響
h3 = rnn.forward(x3, h2)  # h2が h3の計算に影響

print("各ステップの隠れ状態:")
print(f"h1: {h1}")
print(f"h2: {h2}")
print(f"h3: {h3}")
# %%
# 長期依存性の問題を可視化
def demonstrate_vanishing_gradient():
    # 長いシーケンスでRNNを実行
    rnn = SimpleRNN(input_size=1, hidden_size=5)
    sequence_length = 2000
    
    # ランダムな入力シーケンス
    inputs = [np.random.randn(1, 1) for _ in range(sequence_length)]
    
    hidden_states = []
    h = np.zeros((1, 5))
    
    for x in inputs:
        h = rnn.forward(x, h)
        hidden_states.append(h.copy())
    
    # 最初の入力の影響を可視化
    first_input_influence = []
    for i, h in enumerate(hidden_states):
        # 簡単な影響度計算（実際はより複雑）
        influence = np.mean(np.abs(h))
        first_input_influence.append(influence)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(first_input_influence)), first_input_influence)
    plt.title('RNNでの情報の減衰（勾配消失問題）')
    plt.xlabel('時間ステップ')
    plt.ylabel('最初の入力の影響度')
    plt.grid(True)
    plt.show()

demonstrate_vanishing_gradient()
# %%
import numpy as np
import matplotlib.pyplot as plt

def simulate_gradient_instability():
    """勾配不安定性のシミュレーション"""
    
    # RNNパラメータ
    hidden_size = 3
    W_hh = np.array([[0.5, -0.3, 0.2],
                     [0.1, 0.4, -0.5],
                     [-0.2, 0.3, 0.6]])  # 固有値が1前後
    
    # 入力シーケンス
    sequence_length = 200
    inputs = np.random.randn(sequence_length, hidden_size) * 0.1
    
    # 前向き計算
    h = np.zeros(hidden_size)
    hidden_states = []
    activations = []  # tanh の入力値
    
    for x in inputs:
        z = W_hh @ h + x
        h = np.tanh(z)
        hidden_states.append(h.copy())
        activations.append(z.copy())
    
    # 勾配の逆向き計算（簡略版）
    gradient_norms = []
    grad = np.ones(hidden_size)  # 最終勾配
    
    for t in range(sequence_length-1, -1, -1):
        # 活性化関数の微分
        tanh_grad = 1 - np.tanh(activations[t])**2
        
        # 勾配の更新
        grad = grad * tanh_grad  # 要素ごとの積
        grad = W_hh.T @ grad     # 重み行列による変換
        
        gradient_norm = np.linalg.norm(grad)
        gradient_norms.append(gradient_norm)
    
    # 順序を逆転（時間ステップ順に）
    gradient_norms = gradient_norms[::-1]
    
    # 可視化
    plt.figure(figsize=(15, 10))
    
    # 勾配の変化
    plt.subplot(2, 2, 1)
    plt.plot(gradient_norms, 'b-o', linewidth=2, markersize=6)
    plt.title('勾配ノルムの時間変化')
    plt.xlabel('時間ステップ')
    plt.ylabel('勾配ノルム')
    plt.grid(True, alpha=0.3)
    
    # 対数スケール
    plt.subplot(2, 2, 2)
    plt.semilogy(gradient_norms, 'r-s', linewidth=2, markersize=6)
    plt.title('勾配ノルム (対数スケール)')
    plt.xlabel('時間ステップ')
    plt.ylabel('勾配ノルム (log)')
    plt.grid(True, alpha=0.3)
    
    # 活性化関数の微分
    plt.subplot(2, 2, 3)
    tanh_derivatives = [1 - np.tanh(z)**2 for z in activations]
    avg_derivatives = [np.mean(d) for d in tanh_derivatives]
    plt.plot(avg_derivatives, 'g-^', linewidth=2, markersize=6)
    plt.title('Tanh微分の平均値')
    plt.xlabel('時間ステップ')
    plt.ylabel('平均微分値')
    plt.grid(True, alpha=0.3)
    
    # 重み行列の固有値
    plt.subplot(2, 2, 4)
    eigenvalues = np.linalg.eigvals(W_hh)
    eigenvalue_magnitudes = np.abs(eigenvalues)
    
    plt.bar(range(len(eigenvalues)), eigenvalue_magnitudes, alpha=0.7)
    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='|λ|=1')
    plt.title('重み行列の固有値の大きさ')
    plt.xlabel('固有値番号')
    plt.ylabel('|固有値|')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 分析結果
    print("🔍 シミュレーション結果分析:")
    print(f"勾配ノルムの範囲: {min(gradient_norms):.4f} ～ {max(gradient_norms):.4f}")
    print(f"変動比: {max(gradient_norms)/min(gradient_norms):.1f}倍")
    print(f"最大固有値: {max(eigenvalue_magnitudes):.3f}")
    print(f"最小固有値: {min(eigenvalue_magnitudes):.3f}")
    
    return gradient_norms, eigenvalue_magnitudes

# 実行
gradient_norms, eigenvalues = simulate_gradient_instability()
# %%
# LSTMの基本概念を図で理解
def visualize_lstm_concept():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # RNN vs LSTM の情報の流れ
    steps = range(1,31)
    rnn_memory = [1.0 * (0.8 ** i) for i in steps]  # 指数的減衰
    lstm_cell_state = [0.9] * len(steps)            # 長期記憶（安定）
    lstm_hidden = [0.7 + 0.3 * np.sin(i) for i in steps]  # 短期記憶（変動）
    
    ax1.plot(steps, rnn_memory, 'r-', label='RNN Memory', linewidth=3)
    ax1.set_title('RNN: 記憶の減衰')
    ax1.set_xlabel('時間ステップ')
    ax1.set_ylabel('記憶の強さ')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(steps, lstm_cell_state, 'b-', label='LSTM Cell State (長期記憶)', linewidth=3)
    ax2.plot(steps, lstm_hidden, 'g-', label='LSTM Hidden State (短期記憶)', linewidth=2)
    ax2.set_title('LSTM: 長期・短期記憶の分離')
    ax2.set_xlabel('時間ステップ')
    ax2.set_ylabel('記憶の強さ')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

visualize_lstm_concept()
# %%
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def tanh(x):
    return np.tanh(np.clip(x, -500, 500))

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 各ゲートの重み行列を初期化
        self.Wf = np.random.randn(input_size + hidden_size, hidden_size) * 0.1  # 忘却ゲート
        self.Wi = np.random.randn(input_size + hidden_size, hidden_size) * 0.1  # 入力ゲート
        self.Wo = np.random.randn(input_size + hidden_size, hidden_size) * 0.1  # 出力ゲート
        self.Wc = np.random.randn(input_size + hidden_size, hidden_size) * 0.1  # セル状態候補
        
        # バイアス
        self.bf = np.zeros((1, hidden_size))
        self.bi = np.zeros((1, hidden_size))
        self.bo = np.zeros((1, hidden_size))
        self.bc = np.zeros((1, hidden_size))
    
    def forward(self, x, h_prev, c_prev):
        """
        LSTM forward pass
        x: 現在の入力
        h_prev: 前の隠れ状態
        c_prev: 前のセル状態
        """
        # 入力と前の隠れ状態を結合
        combined = np.hstack([x, h_prev])
        
        # 1. 忘却ゲート: 何を忘れるか
        forget_gate = sigmoid(np.dot(combined, self.Wf) + self.bf)
        
        # 2. 入力ゲート: 何を記憶するか
        input_gate = sigmoid(np.dot(combined, self.Wi) + self.bi)
        
        # 3. セル状態の候補値
        cell_candidate = tanh(np.dot(combined, self.Wc) + self.bc)
        
        # 4. セル状態の更新
        c_next = forget_gate * c_prev + input_gate * cell_candidate
        
        # 5. 出力ゲート: 何を出力するか
        output_gate = sigmoid(np.dot(combined, self.Wo) + self.bo)
        
        # 6. 隠れ状態の更新
        h_next = output_gate * tanh(c_next)
        
        return h_next, c_next, {
            'forget_gate': forget_gate,
            'input_gate': input_gate,
            'output_gate': output_gate,
            'cell_candidate': cell_candidate
        }

# LSTMセルをテスト
lstm = LSTMCell(input_size=3, hidden_size=4)

# 初期状態
h0 = np.zeros((1, 4))
c0 = np.zeros((1, 4))

# 入力シーケンス
x1 = np.array([[1, 0, 0]])  # "I"
x2 = np.array([[0, 1, 0]])  # "lived"
x3 = np.array([[0, 0, 1]])  # "in"

# フォワードパス
h1, c1, gates1 = lstm.forward(x1, h0, c0)
h2, c2, gates2 = lstm.forward(x2, h1, c1)
h3, c3, gates3 = lstm.forward(x3, h2, c2)

print("LSTMの動作:")
print(f"ステップ1 - 隠れ状態: {h1.shape}, セル状態: {c1.shape}")
print(f"ステップ2 - 隠れ状態: {h2.shape}, セル状態: {c2.shape}")
print(f"ステップ3 - 隠れ状態: {h3.shape}, セル状態: {c3.shape}")
# %%
def visualize_gates():
    """各ゲートの働きを可視化"""
    lstm = LSTMCell(input_size=1, hidden_size=3)
    
    # テスト入力
    sequence = [np.array([[0.5]]), np.array([[1.0]]), np.array([[0.2]]), np.array([[-0.5]])]
    
    h = np.zeros((1, 3))
    c = np.zeros((1, 3))
    
    gate_history = {'forget': [], 'input': [], 'output': []}
    
    for i, x in enumerate(sequence):
        h, c, gates = lstm.forward(x, h, c)
        gate_history['forget'].append(np.mean(gates['forget_gate']))
        gate_history['input'].append(np.mean(gates['input_gate']))
        gate_history['output'].append(np.mean(gates['output_gate']))
    
    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    steps = range(len(sequence))
    
    axes[0].bar(steps, gate_history['forget'], color='red', alpha=0.7)
    axes[0].set_title('忘却ゲート (Forget Gate)')
    axes[0].set_ylabel('ゲート開放度')
    axes[0].set_ylim(0, 1)
    
    axes[1].bar(steps, gate_history['input'], color='blue', alpha=0.7)
    axes[1].set_title('入力ゲート (Input Gate)')
    axes[1].set_ylabel('ゲート開放度')
    axes[1].set_ylim(0, 1)
    
    axes[2].bar(steps, gate_history['output'], color='green', alpha=0.7)
    axes[2].set_title('出力ゲート (Output Gate)')
    axes[2].set_ylabel('ゲート開放度')
    axes[2].set_ylim(0, 1)
    
    for ax in axes:
        ax.set_xlabel('時間ステップ')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_gates()
# %%
class LSTM:
    def __init__(self, input_size, hidden_size, num_layers=1):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # 複数層のLSTMセルを作成
        self.cells = []
        for i in range(num_layers):
            if i == 0:
                cell_input_size = input_size
            else:
                cell_input_size = hidden_size
            self.cells.append(LSTMCell(cell_input_size, hidden_size))
    
    def forward(self, sequence):
        """
        sequence: リストの入力シーケンス
        """
        batch_size = sequence[0].shape[0]
        
        # 各層の初期状態
        h_states = [np.zeros((batch_size, self.hidden_size)) for _ in range(self.num_layers)]
        c_states = [np.zeros((batch_size, self.hidden_size)) for _ in range(self.num_layers)]
        
        outputs = []
        
        for x in sequence:
            layer_input = x
            
            # 各層を順番に通す
            for layer_idx in range(self.num_layers):
                h_states[layer_idx], c_states[layer_idx], _ = self.cells[layer_idx].forward(
                    layer_input, h_states[layer_idx], c_states[layer_idx]
                )
                layer_input = h_states[layer_idx]
            
            outputs.append(h_states[-1])  # 最後の層の出力
        
        return outputs, h_states, c_states

# 2層LSTMでテスト
lstm_network = LSTM(input_size=3, hidden_size=4, num_layers=2)

# テストシーケンス
test_sequence = [
    np.array([[1, 0, 0]]),  # "I"
    np.array([[0, 1, 0]]),  # "lived"
    np.array([[0, 0, 1]]),  # "in"
    np.array([[1, 1, 0]]),  # "Ireland"
]

outputs, final_h, final_c = lstm_network.forward(test_sequence)

print("2層LSTMの出力:")
for i, output in enumerate(outputs):
    print(f"ステップ {i+1}: {output.shape}")
# %%
