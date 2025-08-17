#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

# シグモイド関数とtanh関数の定義
def sigmoid(x):
    """シグモイド関数：0〜1の値を出力"""
    return 1 / (1 + np.exp(-x))

def tanh(x):
    """tanh関数：-1〜1の値を出力"""
    return np.tanh(x)

class SimpleLSTM:
    """LSTMセルの簡易実装（教育用）"""
    
    def __init__(self, input_size, hidden_size):
        """
        Parameters:
        -----------
        input_size : int
            入力データの次元数
        hidden_size : int
            隠れ状態の次元数
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 重み行列の初期化（簡単のため小さな乱数で初期化）
        scale = 0.1
        
        # forget gate の重み
        self.Wf = np.random.randn(hidden_size, input_size) * scale
        self.Uf = np.random.randn(hidden_size, hidden_size) * scale
        self.bf = np.zeros((hidden_size, 1))
        
        # input gate の重み
        self.Wi = np.random.randn(hidden_size, input_size) * scale
        self.Ui = np.random.randn(hidden_size, hidden_size) * scale
        self.bi = np.zeros((hidden_size, 1))
        
        # candidate values の重み
        self.Wc = np.random.randn(hidden_size, input_size) * scale
        self.Uc = np.random.randn(hidden_size, hidden_size) * scale
        self.bc = np.zeros((hidden_size, 1))
        
        # output gate の重み
        self.Wo = np.random.randn(hidden_size, input_size) * scale
        self.Uo = np.random.randn(hidden_size, hidden_size) * scale
        self.bo = np.zeros((hidden_size, 1))
        
    def forward_step(self, x_t, h_prev, c_prev, verbose=True):
        """
        LSTMの1ステップの順伝播を実行
        
        Parameters:
        -----------
        x_t : array (input_size, 1)
            時刻tの入力
        h_prev : array (hidden_size, 1)
            前の時刻の隠れ状態
        c_prev : array (hidden_size, 1)
            前の時刻のセル状態
        verbose : bool
            詳細な出力を表示するか
        
        Returns:
        --------
        h_t : 新しい隠れ状態
        c_t : 新しいセル状態
        gates : 各ゲートの値（可視化用）
        """
        
        if verbose:
            print("="*60)
            print("LSTMの順伝播計算（ステップバイステップ）")
            print("="*60)
        
        # 1. Forget Gate（忘却ゲート）
        # 過去の情報をどれだけ忘れるかを決定（0=完全に忘れる、1=完全に保持）
        f_t = sigmoid(self.Wf @ x_t + self.Uf @ h_prev + self.bf)
        if verbose:
            print("\n【1. Forget Gate (忘却ゲート)】")
            print(f"  式: f_t = σ(Wf·x_t + Uf·h_prev + bf)")
            print(f"  値: {f_t.flatten()}")
            print(f"  意味: 前の記憶をどれだけ保持するか（0=忘れる、1=保持）")
        
        # 2. Input Gate（入力ゲート）
        # 新しい情報をどれだけ記憶に追加するかを決定
        i_t = sigmoid(self.Wi @ x_t + self.Ui @ h_prev + self.bi)
        if verbose:
            print("\n【2. Input Gate (入力ゲート)】")
            print(f"  式: i_t = σ(Wi·x_t + Ui·h_prev + bi)")
            print(f"  値: {i_t.flatten()}")
            print(f"  意味: 新しい情報をどれだけ記憶に追加するか")
        
        # 3. Candidate Values（候補値）
        # 追加する可能性のある新しい情報
        c_tilde = tanh(self.Wc @ x_t + self.Uc @ h_prev + self.bc)
        if verbose:
            print("\n【3. Candidate Values (候補値)】")
            print(f"  式: c̃_t = tanh(Wc·x_t + Uc·h_prev + bc)")
            print(f"  値: {c_tilde.flatten()}")
            print(f"  意味: 追加する可能性のある新しい情報（-1〜1）")
        
        # 4. Cell State Update（セル状態の更新）
        # 古い記憶と新しい記憶を組み合わせる
        c_t = f_t * c_prev + i_t * c_tilde
        if verbose:
            print("\n【4. Cell State Update (セル状態の更新)】")
            print(f"  式: c_t = f_t·c_prev + i_t·c̃_t")
            print(f"  値: {c_t.flatten()}")
            print(f"  意味: 忘却ゲートで調整した古い記憶 + 入力ゲートで調整した新しい記憶")
        
        # 5. Output Gate（出力ゲート）
        # どの情報を出力するかを決定
        o_t = sigmoid(self.Wo @ x_t + self.Uo @ h_prev + self.bo)
        if verbose:
            print("\n【5. Output Gate (出力ゲート)】")
            print(f"  式: o_t = σ(Wo·x_t + Uo·h_prev + bo)")
            print(f"  値: {o_t.flatten()}")
            print(f"  意味: セル状態のどの部分を出力するか")
        
        # 6. Hidden State（隠れ状態の計算）
        # 最終的な出力
        h_t = o_t * tanh(c_t)
        if verbose:
            print("\n【6. Hidden State (隠れ状態)】")
            print(f"  式: h_t = o_t·tanh(c_t)")
            print(f"  値: {h_t.flatten()}")
            print(f"  意味: 出力ゲートで調整されたセル状態（最終出力）")
        
        # ゲートの値を返す（可視化用）
        gates = {
            'forget': f_t,
            'input': i_t,
            'candidate': c_tilde,
            'output': o_t,
            'cell_state': c_t,
            'hidden_state': h_t
        }
        
        return h_t, c_t, gates

def visualize_gates(gates_history):
    """各ゲートの値の時系列変化を可視化"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('LSTMの各ゲート値の時系列変化', fontsize=16, fontweight='bold')
    
    gate_names = ['forget', 'input', 'candidate', 'output', 'cell_state', 'hidden_state']
    gate_labels = ['Forget Gate', 'Input Gate', 'Candidate Values', 
                   'Output Gate', 'Cell State', 'Hidden State']
    
    for idx, (gate_name, label) in enumerate(zip(gate_names, gate_labels)):
        ax = axes[idx // 3, idx % 3]
        
        # 各時刻のゲート値を取得
        values = np.array([gates[gate_name].flatten() for gates in gates_history])
        
        # ヒートマップで表示
        im = ax.imshow(values.T, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(label, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Hidden Unit')
        ax.set_xticks(range(len(gates_history)))
        
        # カラーバーを追加
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

# ========================================
# 実験：LSTMの動作を確認
# ========================================

def experiment_lstm():
    """LSTMの動作を実験的に確認"""
    
    print("LSTMの実験を開始します！\n")
    
    # パラメータ設定
    input_size = 3   # 入力の次元数
    hidden_size = 4  # 隠れ状態の次元数
    seq_length = 5   # シーケンスの長さ
    
    # LSTMの初期化
    lstm = SimpleLSTM(input_size, hidden_size)
    
    # 初期状態
    h_t = np.zeros((hidden_size, 1))  # 隠れ状態
    c_t = np.zeros((hidden_size, 1))  # セル状態
    
    # テスト用の入力シーケンスを生成
    # （例：徐々に大きくなる値）
    input_sequence = []
    for t in range(seq_length):
        x_t = np.random.randn(input_size, 1) * (t + 1) * 0.2
        input_sequence.append(x_t)
    
    print(f"入力シーケンスの長さ: {seq_length}")
    print(f"入力の次元数: {input_size}")
    print(f"隠れ状態の次元数: {hidden_size}\n")
    
    # 各時刻でLSTMを実行
    gates_history = []
    
    for t, x_t in enumerate(input_sequence):
        print(f"\n{'='*60}")
        print(f"時刻 t={t}")
        print(f"{'='*60}")
        print(f"入力 x_t: {x_t.flatten()}")
        
        # LSTMの1ステップを実行
        h_t, c_t, gates = lstm.forward_step(x_t, h_t, c_t, verbose=True)
        gates_history.append(gates)
        
        print(f"\n最終出力 h_t: {h_t.flatten()}")
        
        input("\n[Enterキーを押して次の時刻へ進む...]")
    
    # ゲートの値を可視化
    print("\n" + "="*60)
    print("各ゲートの時系列変化を可視化します...")
    print("="*60)
    visualize_gates(gates_history)

# ========================================
# 特定のパターンに対するLSTMの反応を観察
# ========================================

def experiment_patterns():
    """特定のパターンに対するLSTMの反応を観察"""
    
    print("\n特定パターンに対するLSTMの反応実験")
    print("="*60)
    
    input_size = 1
    hidden_size = 2
    lstm = SimpleLSTM(input_size, hidden_size)
    
    # パターン1: ステップ関数（急激な変化）
    print("\n【パターン1: ステップ関数】")
    pattern1 = [0, 0, 1, 1, 1]
    
    h_t = np.zeros((hidden_size, 1))
    c_t = np.zeros((hidden_size, 1))
    
    for t, val in enumerate(pattern1):
        x_t = np.array([[val]])
        h_t, c_t, gates = lstm.forward_step(x_t, h_t, c_t, verbose=False)
        print(f"  時刻{t}: 入力={val:.1f}, Forget={gates['forget'][0,0]:.3f}, "
              f"Input={gates['input'][0,0]:.3f}, Output={h_t[0,0]:.3f}")
    
    # パターン2: 正弦波（周期的変化）
    print("\n【パターン2: 正弦波】")
    pattern2 = [np.sin(t * np.pi / 2) for t in range(5)]
    
    h_t = np.zeros((hidden_size, 1))
    c_t = np.zeros((hidden_size, 1))
    
    for t, val in enumerate(pattern2):
        x_t = np.array([[val]])
        h_t, c_t, gates = lstm.forward_step(x_t, h_t, c_t, verbose=False)
        print(f"  時刻{t}: 入力={val:.2f}, Forget={gates['forget'][0,0]:.3f}, "
              f"Input={gates['input'][0,0]:.3f}, Output={h_t[0,0]:.3f}")

# ========================================
# メイン実行
# ========================================

if __name__ == "__main__":
    print("LSTMの動作を学ぶインタラクティブな実験プログラム")
    print("="*60)
    print("\n実験メニュー:")
    print("1. LSTMの基本動作を詳しく見る（推奨）")
    print("2. 特定パターンに対する反応を見る")
    print("3. 両方実行")
    
    choice = input("\n選択してください (1/2/3): ")
    
    if choice == "1":
        experiment_lstm()
    elif choice == "2":
        experiment_patterns()
    elif choice == "3":
        experiment_lstm()
        experiment_patterns()
    else:
        print("1、2、または3を入力してください")
# %%
