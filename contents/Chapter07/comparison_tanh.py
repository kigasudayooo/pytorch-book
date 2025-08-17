#%%
import japanize_matplotlib

#%%
import numpy as np
import matplotlib.pyplot as plt

def demonstrate_gradient_impact():
    """勾配更新の違いを具体的に示す"""
    
    # シンプルなRNNの重み更新シミュレーション
    def simulate_weight_update(activation_func, input_seq, learning_rate=0.1):
        W_hh = np.array([[0.1, -0.2], [0.3, 0.1]])  # 初期重み
        h = np.array([0.0, 0.0])  # 初期隠れ状態
        
        weight_updates = []
        hidden_states = []
        
        for x in input_seq:
            # Forward pass
            z = W_hh @ h + x
            h_new = activation_func(z)
            
            # 簡単な損失関数の勾配 (例: MSE)
            target = np.array([0.5, -0.3])  # 目標値
            error = h_new - target
            
            # 勾配計算
            if activation_func == np.tanh:
                grad_act = 1 - h_new**2  # tanh'(z)
            else:  # sigmoid
                grad_act = h_new * (1 - h_new)  # sigmoid'(z)
            
            grad_W = np.outer(error * grad_act, h)
            
            # 重み更新
            W_hh -= learning_rate * grad_W
            
            weight_updates.append(np.linalg.norm(grad_W))
            hidden_states.append(h_new.copy())
            h = h_new
        
        return weight_updates, hidden_states, W_hh
    
    # 同じ入力シーケンス
    input_sequence = [np.array([0.5, -0.2]), np.array([-0.3, 0.7]), 
                     np.array([0.1, -0.5]), np.array([0.8, 0.2])]
    
    # tanh vs sigmoid
    updates_tanh, states_tanh, final_W_tanh = simulate_weight_update(np.tanh, input_sequence)
    updates_sigmoid, states_sigmoid, final_W_sigmoid = simulate_weight_update(
        lambda x: 1/(1+np.exp(-np.clip(x, -500, 500))), input_sequence)
    
    # 結果の可視化
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # 勾配の大きさ
    ax1.plot(updates_tanh, 'b-o', label='tanh', linewidth=2)
    ax1.plot(updates_sigmoid, 'r-s', label='sigmoid', linewidth=2)
    ax1.set_title('勾配の大きさ (||∇W||)')
    ax1.set_xlabel('時間ステップ')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 隠れ状態の軌跡
    states_tanh = np.array(states_tanh)
    states_sigmoid = np.array(states_sigmoid)
    
    ax2.plot(states_tanh[:, 0], states_tanh[:, 1], 'b-o', label='tanh path', linewidth=2)
    ax2.plot(states_sigmoid[:, 0], states_sigmoid[:, 1], 'r-s', label='sigmoid path', linewidth=2)
    ax2.scatter([0.5], [-0.3], c='green', s=100, marker='*', label='target')
    ax2.set_title('隠れ状態の軌跡')
    ax2.set_xlabel('h[0]')
    ax2.set_ylabel('h[1]')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 最終的な重み行列の差
    weight_diff = np.abs(final_W_tanh - final_W_sigmoid)
    im = ax3.imshow(weight_diff, cmap='viridis')
    ax3.set_title('最終重み行列の差\n|W_tanh - W_sigmoid|')
    plt.colorbar(im, ax=ax3)
    
    plt.tight_layout()
    plt.show()
    
    return updates_tanh, updates_sigmoid, final_W_tanh, final_W_sigmoid

# 実行
updates_tanh, updates_sigmoid, W_tanh, W_sigmoid = demonstrate_gradient_impact()

print("勾配更新の統計:")
print(f"Tanh平均勾配: {np.mean(updates_tanh):.4f}")
print(f"Sigmoid平均勾配: {np.mean(updates_sigmoid):.4f}")
print(f"勾配比: {np.mean(updates_tanh)/np.mean(updates_sigmoid):.2f}")
# %%
def analyze_zero_centering():
    """ゼロ中心性の統計的効果を分析"""
    
    # 勾配の符号分析
    def gradient_sign_analysis(activation_outputs):
        """勾配の符号パターンを分析"""
        # 簡単な損失関数を仮定: L = (h - target)^2
        target = 0.0  # ゼロターゲット
        
        gradients = 2 * (activation_outputs - target)  # dL/dh
        
        positive_grads = np.sum(gradients > 0)
        negative_grads = np.sum(gradients < 0)
        zero_grads = np.sum(gradients == 0)
        
        return positive_grads, negative_grads, zero_grads
    
    # 入力データ
    z = np.linspace(-3, 3, 1000)
    
    # 活性化関数の出力
    tanh_out = np.tanh(z)
    sigmoid_out = 1 / (1 + np.exp(-z))
    
    # 勾配符号分析
    pos_tanh, neg_tanh, zero_tanh = gradient_sign_analysis(tanh_out)
    pos_sig, neg_sig, zero_sig = gradient_sign_analysis(sigmoid_out)
    
    print("勾配の符号分布 (ゼロターゲット時):")
    print(f"Tanh: 正({pos_tanh}), 負({neg_tanh}), ゼロ({zero_tanh})")
    print(f"Sigmoid: 正({pos_sig}), 負({neg_sig}), ゼロ({zero_sig})")
    
    # 平均と分散
    print(f"\n出力の統計:")
    print(f"Tanh: 平均={np.mean(tanh_out):.4f}, 分散={np.var(tanh_out):.4f}")
    print(f"Sigmoid: 平均={np.mean(sigmoid_out):.4f}, 分散={np.var(sigmoid_out):.4f}")
    
    # 勾配の分散
    tanh_grad = 1 - tanh_out**2
    sigmoid_grad = sigmoid_out * (1 - sigmoid_out)
    
    print(f"\n勾配の統計:")
    print(f"Tanh: 平均={np.mean(tanh_grad):.4f}, 分散={np.var(tanh_grad):.4f}")
    print(f"Sigmoid: 平均={np.mean(sigmoid_grad):.4f}, 分散={np.var(sigmoid_grad):.4f}")

analyze_zero_centering()
# %%
def convergence_comparison():
    """学習収束性の実際の比較（修正版）"""
    
    class SimpleRNN:
        def __init__(self, activation='tanh'):
            self.W_hh = np.random.randn(2, 2) * 0.1
            self.W_xh = np.random.randn(2, 1) * 0.1  # 修正: (2, 1) に変更
            self.activation = activation
            self.losses = []
    
        def forward(self, sequence, targets):
            h = np.zeros(2)
            total_loss = 0
            
            for i, (x, target) in enumerate(zip(sequence, targets)):
                # 修正: スカラーのxをベクトルとして扱う
                x_vec = np.array([x])  # スカラーを1次元配列に
                z = (self.W_xh @ x_vec).flatten() + self.W_hh @ h
                
                if self.activation == 'tanh':
                    h = np.tanh(z)
                else:
                    h = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
                
                loss = 0.5 * np.sum((h - target)**2)
                total_loss += loss
            
            return total_loss
    
        def train(self, sequence, targets, epochs=100, lr=0.01):
            for epoch in range(epochs):
                loss = self.forward(sequence, targets)
                self.losses.append(loss)
                
                # 簡単な数値微分による重み更新
                eps = 1e-6
                
                # W_xh の更新
                for i in range(2):
                    for j in range(1):
                        self.W_xh[i,j] += eps
                        loss_plus = self.forward(sequence, targets)
                        self.W_xh[i,j] -= 2*eps
                        loss_minus = self.forward(sequence, targets)
                        self.W_xh[i,j] += eps
                        
                        grad = (loss_plus - loss_minus) / (2*eps)
                        self.W_xh[i,j] -= lr * grad
                
                # W_hh の更新
                for i in range(2):
                    for j in range(2):
                        self.W_hh[i,j] += eps
                        loss_plus = self.forward(sequence, targets)
                        self.W_hh[i,j] -= 2*eps
                        loss_minus = self.forward(sequence, targets)
                        self.W_hh[i,j] += eps
                        
                        grad = (loss_plus - loss_minus) / (2*eps)
                        self.W_hh[i,j] -= lr * grad
    
    # トレーニングデータ
    sequence = [0.5, -0.3, 0.8, -0.2]
    targets = [np.array([0.2, -0.1]), np.array([-0.3, 0.4]), 
               np.array([0.1, -0.2]), np.array([0.0, 0.3])]
    
    # モデル訓練
    rnn_tanh = SimpleRNN('tanh')
    rnn_sigmoid = SimpleRNN('sigmoid')
    
    print("学習開始...")
    rnn_tanh.train(sequence, targets, epochs=50)
    rnn_sigmoid.train(sequence, targets, epochs=50)
    
    # 結果比較
    plt.figure(figsize=(12, 8))
    
    # 上段: 損失の推移
    plt.subplot(2, 2, 1)
    plt.plot(rnn_tanh.losses, 'b-', label='tanh', linewidth=2)
    plt.plot(rnn_sigmoid.losses, 'r-', label='sigmoid', linewidth=2)
    plt.xlabel('エポック')
    plt.ylabel('損失')
    plt.title('学習収束性の比較')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # 上段右: 損失の線形スケール
    plt.subplot(2, 2, 2)
    plt.plot(rnn_tanh.losses, 'b-', label='tanh', linewidth=2)
    plt.plot(rnn_sigmoid.losses, 'r-', label='sigmoid', linewidth=2)
    plt.xlabel('エポック')
    plt.ylabel('損失')
    plt.title('学習収束性（線形スケール）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 下段: 最終的な性能比較
    plt.subplot(2, 2, 3)
    final_losses = [rnn_tanh.losses[-1], rnn_sigmoid.losses[-1]]
    colors = ['blue', 'red']
    bars = plt.bar(['Tanh', 'Sigmoid'], final_losses, color=colors, alpha=0.7)
    plt.ylabel('最終損失')
    plt.title('最終性能比較')
    plt.grid(True, alpha=0.3)
    
    # 数値をバーの上に表示
    for bar, loss in zip(bars, final_losses):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(final_losses)*0.01,
                f'{loss:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 下段右: 学習速度比較
    plt.subplot(2, 2, 4)
    # 最初の10エポックの改善率
    tanh_improvement = rnn_tanh.losses[0] - rnn_tanh.losses[9] if len(rnn_tanh.losses) > 9 else 0
    sigmoid_improvement = rnn_sigmoid.losses[0] - rnn_sigmoid.losses[9] if len(rnn_sigmoid.losses) > 9 else 0
    
    improvements = [tanh_improvement, sigmoid_improvement]
    bars = plt.bar(['Tanh', 'Sigmoid'], improvements, color=colors, alpha=0.7)
    plt.ylabel('初期10エポックの改善量')
    plt.title('学習速度比較')
    plt.grid(True, alpha=0.3)
    
    for bar, improvement in zip(bars, improvements):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(improvements)*0.01,
                f'{improvement:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 詳細な結果分析:")
    print(f"最終損失:")
    print(f"  Tanh: {rnn_tanh.losses[-1]:.6f}")
    print(f"  Sigmoid: {rnn_sigmoid.losses[-1]:.6f}")
    
    if rnn_sigmoid.losses[-1] != 0:
        improvement = (rnn_sigmoid.losses[-1] - rnn_tanh.losses[-1])/rnn_sigmoid.losses[-1]*100
        print(f"  Tanh改善率: {improvement:.1f}%")
    
    print(f"\n初期学習速度 (最初10エポックの改善):")
    print(f"  Tanh: {tanh_improvement:.6f}")
    print(f"  Sigmoid: {sigmoid_improvement:.6f}")
    
    # 収束性の分析
    tanh_variance = np.var(rnn_tanh.losses[-10:])  # 最後の10エポックの分散
    sigmoid_variance = np.var(rnn_sigmoid.losses[-10:])
    
    print(f"\n収束安定性 (最後10エポックの分散):")
    print(f"  Tanh: {tanh_variance:.8f}")
    print(f"  Sigmoid: {sigmoid_variance:.8f}")
    print(f"  → {'Tanh' if tanh_variance < sigmoid_variance else 'Sigmoid'}の方が安定")

# 実行
convergence_comparison()
# %%
import numpy as np
import matplotlib.pyplot as plt

def explain_gradient_saturation():
    """勾配飽和現象の数学的解析"""
    
    # 入力範囲
    z = np.linspace(-5, 5, 1000)
    
    # 活性化関数
    sigmoid = 1 / (1 + np.exp(-z))
    tanh = np.tanh(z)
    
    # 勾配（微分）
    sigmoid_grad = sigmoid * (1 - sigmoid)
    tanh_grad = 1 - tanh**2
    
    # 可視化
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 活性化関数
    ax1.plot(z, sigmoid, 'r-', linewidth=3, label='sigmoid(z)')
    ax1.set_title('Sigmoid活性化関数')
    ax1.set_xlabel('入力 z')
    ax1.set_ylabel('出力')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax1.legend()
    
    ax2.plot(z, tanh, 'b-', linewidth=3, label='tanh(z)')
    ax2.set_title('Tanh活性化関数')
    ax2.set_xlabel('入力 z')
    ax2.set_ylabel('出力')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='blue', linestyle='--', alpha=0.5)
    ax2.legend()
    
    # 勾配
    ax3.plot(z, sigmoid_grad, 'r-', linewidth=3, label="sigmoid'(z)")
    ax3.set_title('Sigmoid勾配（微分）')
    ax3.set_xlabel('入力 z')
    ax3.set_ylabel('勾配の大きさ')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    ax3.legend()
    
    ax4.plot(z, tanh_grad, 'b-', linewidth=3, label="tanh'(z)")
    ax4.set_title('Tanh勾配（微分）')
    ax4.set_xlabel('入力 z')
    ax4.set_ylabel('勾配の大きさ')
    ax4.grid(True, alpha=0.3)
    ax4.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 数値解析
    print("🔍 勾配飽和の数値解析:")
    print()
    
    # 特定の入力値での勾配比較
    test_values = [-3, -1, 0, 1, 3]
    
    print("入力値 |  Sigmoid勾配  |   Tanh勾配   |   比率")
    print("-" * 50)
    
    for z_val in test_values:
        sig_grad = (1/(1+np.exp(-z_val))) * (1 - 1/(1+np.exp(-z_val)))
        tanh_grad_val = 1 - np.tanh(z_val)**2
        ratio = tanh_grad_val / sig_grad if sig_grad > 0 else float('inf')
        
        print(f"  {z_val:2d}   |    {sig_grad:.4f}    |    {tanh_grad_val:.4f}    |  {ratio:.2f}x")
    
    # 飽和領域の特定
    sigmoid_saturated = np.sum(sigmoid_grad < 0.1)  # 勾配が0.1未満の点
    tanh_saturated = np.sum(tanh_grad < 0.1)
    
    print(f"\n📊 飽和現象の比較:")
    print(f"Sigmoid飽和点数: {sigmoid_saturated}/1000 ({sigmoid_saturated/10:.1f}%)")
    print(f"Tanh飽和点数: {tanh_saturated}/1000 ({tanh_saturated/10:.1f}%)")

explain_gradient_saturation()
# %%
def information_theoretic_analysis():
    """情報理論的な解析"""
    
    print("📡 情報理論的観点からの解析:")
    print()
    
    # エントロピー計算
    def calculate_entropy(probabilities):
        """確率分布のエントロピー計算"""
        # 0に近い値を避けるため小さな値を加算
        p = np.array(probabilities) + 1e-8
        p = p / np.sum(p)  # 正規化
        return -np.sum(p * np.log2(p))
    
    # 異なる活性化関数での出力分布
    z_inputs = np.random.normal(0, 1, 1000)  # 標準正規分布からの入力
    
    sigmoid_outputs = 1 / (1 + np.exp(-z_inputs))
    tanh_outputs = np.tanh(z_inputs)
    
    # 出力を離散化してエントロピー計算
    bins = 50
    sigmoid_hist, _ = np.histogram(sigmoid_outputs, bins=bins, density=True)
    tanh_hist, _ = np.histogram(tanh_outputs, bins=bins, density=True)
    
    sigmoid_entropy = calculate_entropy(sigmoid_hist)
    tanh_entropy = calculate_entropy(tanh_hist)
    
    print(f"Sigmoid出力のエントロピー: {sigmoid_entropy:.3f} bits")
    print(f"Tanh出力のエントロピー: {tanh_entropy:.3f} bits")
    print(f"情報量の比: {tanh_entropy/sigmoid_entropy:.2f}")
    print()
    
    # 相互情報量の概念的説明
    print("🔄 相互情報量の観点:")
    print("• Tanh: より広い出力範囲 → 高い情報量")
    print("• Sigmoid: 制限された出力範囲 → 低い情報量")
    print("• 高い情報量 → より効率的な学習")
    
    # 可視化
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(sigmoid_outputs, bins=50, alpha=0.7, color='red', density=True, label='Sigmoid')
    plt.title('Sigmoid出力分布')
    plt.xlabel('出力値')
    plt.ylabel('密度')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.hist(tanh_outputs, bins=50, alpha=0.7, color='blue', density=True, label='Tanh')
    plt.title('Tanh出力分布')
    plt.xlabel('出力値')
    plt.ylabel('密度')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    entropy_values = [sigmoid_entropy, tanh_entropy]
    colors = ['red', 'blue']
    bars = plt.bar(['Sigmoid', 'Tanh'], entropy_values, color=colors, alpha=0.7)
    plt.title('情報エントロピー比較')
    plt.ylabel('エントロピー (bits)')
    plt.grid(True, alpha=0.3)
    
    for bar, entropy in zip(bars, entropy_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{entropy:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

information_theoretic_analysis()
# %%
def probability_task_analysis():
    """確率予測タスクでの活性化関数比較"""
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("🎲 確率予測タスクでの選択肢:")
    print()
    print("選択肢1: 直接Sigmoid")
    print("  入力 → Sigmoid → [0,1] 確率")
    print()
    print("選択肢2: Tanh + 補正")
    print("  入力 → Tanh → [-1,1] → 補正 → [0,1] 確率")
    print("  補正: (tanh(x) + 1) / 2")
    print()
    
    # 補正されたTanhとSigmoidの比較
    z = np.linspace(-5, 5, 1000)
    
    # 元の関数
    sigmoid = 1 / (1 + np.exp(-z))
    tanh = np.tanh(z)
    
    # Tanh補正版
    tanh_corrected = (tanh + 1) / 2
    
    # 勾配
    sigmoid_grad = sigmoid * (1 - sigmoid)
    tanh_grad = 1 - tanh**2
    tanh_corrected_grad = tanh_grad / 2  # 連鎖律
    
    # 可視化
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 活性化関数
    ax1.plot(z, sigmoid, 'r-', linewidth=3, label='Sigmoid')
    ax1.plot(z, tanh_corrected, 'b--', linewidth=3, label='(Tanh+1)/2')
    ax1.set_title('確率出力の比較')
    ax1.set_xlabel('入力 z')
    ax1.set_ylabel('確率出力 [0,1]')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 差分
    ax2.plot(z, sigmoid - tanh_corrected, 'g-', linewidth=3)
    ax2.set_title('Sigmoid - (Tanh+1)/2')
    ax2.set_xlabel('入力 z')
    ax2.set_ylabel('差分')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 勾配比較
    ax3.plot(z, sigmoid_grad, 'r-', linewidth=3, label="Sigmoid'")
    ax3.plot(z, tanh_corrected_grad, 'b--', linewidth=3, label="(Tanh+1)'/2")
    ax3.set_title('勾配の比較')
    ax3.set_xlabel('入力 z')
    ax3.set_ylabel('勾配')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 勾配比
    ax4.plot(z, tanh_corrected_grad / (sigmoid_grad + 1e-8), 'purple', linewidth=3)
    ax4.set_title('勾配比: (Tanh+1)\'÷2 / Sigmoid\'')
    ax4.set_xlabel('入力 z')
    ax4.set_ylabel('勾配比')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='等しい')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 数値解析
    print("📊 数値解析:")
    print(f"最大差分: {np.max(np.abs(sigmoid - tanh_corrected)):.8f}")
    print(f"平均差分: {np.mean(np.abs(sigmoid - tanh_corrected)):.8f}")
    print(f"勾配最大値比: {np.max(tanh_corrected_grad) / np.max(sigmoid_grad):.3f}")

probability_task_analysis()
# %%
def mathematical_revelation():
    """数学的な驚きの事実"""
    
    print("🔮 驚きの数学的事実:")
    print()
    print("実は、(tanh(x) + 1) / 2 と sigmoid(x) は...")
    print()
    
    # 数学的関係の証明
    print("📐 数学的関係:")
    print("sigmoid(x) = 1 / (1 + e^(-x))")
    print("tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))")
    print()
    print("変形すると:")
    print("tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)")
    print("     = 2 * sigmoid(2x) - 1")
    print()
    print("つまり:")
    print("(tanh(x) + 1) / 2 = sigmoid(2x)")
    print()
    print("🎯 結論:")
    print("Tanh補正版は、入力を2倍にしたSigmoid！")
    
    # 実証
    import numpy as np
    x = np.linspace(-3, 3, 100)
    
    tanh_corrected = (np.tanh(x) + 1) / 2
    sigmoid_2x = 1 / (1 + np.exp(-2*x))
    
    difference = np.max(np.abs(tanh_corrected - sigmoid_2x))
    print(f"実証: 最大差分 = {difference:.10f} (ほぼ0)")

mathematical_revelation()
# %%
def learning_efficiency_comparison():
    """学習効率の実際の比較"""
    
    print("⚡ 学習効率の比較:")
    print()
    
    # 勾配の比較
    print("勾配の特性:")
    print("• Sigmoid: 最大勾配 = 0.25")
    print("• (Tanh+1)/2: 最大勾配 = 0.5 (2倍大きい!)")
    print()
    
    print("🚀 これが意味すること:")
    print("1. Tanh + 補正の方が学習が早い")
    print("2. しかし関数形は本質的に同じ")
    print("3. 単に「スケールされた」Sigmoid")
    print()
    
    # 実用的な考慮事項
    print("🤔 実用的な考慮事項:")
    print()
    print("利点 (Tanh + 補正):")
    print("✓ より大きな勾配 → 高速学習")
    print("✓ 中間層でゼロ中心の利点")
    print("✓ 数値的安定性")
    print()
    print("欠点 (Tanh + 補正):")
    print("✗ 計算が1ステップ多い")
    print("✗ わずかに複雑")
    print("✗ メモリ使用量がわずかに増加")

learning_efficiency_comparison()
# %%
def experimental_validation():
    """実験による検証"""
    
    print("🧪 実験的検証:")
    print()
    
    # シンプルな学習タスクでの比較
    class ProbabilityPredictor:
        def __init__(self, activation_type='sigmoid'):
            self.W = np.random.randn(1) * 0.1
            self.b = np.random.randn(1) * 0.1
            self.activation_type = activation_type
            self.losses = []
        
        def forward(self, x):
            z = self.W * x + self.b
            
            if self.activation_type == 'sigmoid':
                return 1 / (1 + np.exp(-z))
            else:  # tanh_corrected
                return (np.tanh(z) + 1) / 2
        
        def train_step(self, x, y_true, lr=0.1):
            # Forward pass
            y_pred = self.forward(x)
            
            # Loss (Binary Cross Entropy)
            loss = -(y_true * np.log(y_pred + 1e-8) + 
                    (1 - y_true) * np.log(1 - y_pred + 1e-8))
            self.losses.append(loss)
            
            # Backward pass (numerical gradient)
            eps = 1e-6
            
            # Gradient for W
            self.W += eps
            loss_plus = -(y_true * np.log(self.forward(x) + 1e-8) + 
                         (1 - y_true) * np.log(1 - self.forward(x) + 1e-8))
            self.W -= 2*eps
            loss_minus = -(y_true * np.log(self.forward(x) + 1e-8) + 
                          (1 - y_true) * np.log(1 - self.forward(x) + 1e-8))
            self.W += eps
            
            grad_W = (loss_plus - loss_minus) / (2*eps)
            self.W -= lr * grad_W
    
    # 訓練データ
    X = np.array([1.0, 2.0, -1.0, 0.5, -0.5])
    Y = np.array([0.8, 0.9, 0.1, 0.7, 0.3])  # 確率ターゲット
    
    # モデル訓練
    model_sigmoid = ProbabilityPredictor('sigmoid')
    model_tanh = ProbabilityPredictor('tanh_corrected')
    
    epochs = 50
    for epoch in range(epochs):
        for x, y in zip(X, Y):
            model_sigmoid.train_step(x, y)
            model_tanh.train_step(x, y)
    
    # 結果比較
    print("📊 学習結果:")
    print(f"Sigmoid最終損失: {model_sigmoid.losses[-1]:.6f}")
    print(f"Tanh+補正最終損失: {model_tanh.losses[-1]:.6f}")
    
    improvement = (model_sigmoid.losses[-1] - model_tanh.losses[-1]) / model_sigmoid.losses[-1] * 100
    print(f"改善率: {improvement:.1f}%")

# 実行（簡略版）
print("📈 予想される結果:")
print("Tanh+補正の方が5-15%程度高速に収束")
print("ただし、最終性能はほぼ同等")
experimental_validation()
# %%
