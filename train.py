import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.distance import cdist
import random
import time
import os
from datetime import datetime
import ast

# --- I. 全局参数定义 ---
N_RECORDS = 110000
DATASET_FILENAME = "data.parquet"
TRIPLET_FILE = "triplet_training_samples.csv"
LOG_FILE = "training_metadata.txt"

# 歌单模拟参数
NUM_PLAYLISTS_TO_SIMULATE = 2000
MIN_PLAYLIST_LENGTH = 20
MAX_PLAYLIST_LENGTH = 40
NUM_TARGETS = 5
MAX_SIMILAR_CANDIDATES = 200

# PyTorch 训练参数
EMBEDDING_DIM = 128
MARGIN = 0.2
BATCH_SIZE = 1024
LR = 1e-3
NUM_EPOCHS = 100
LOSS_FUNCTION = "TripletLoss (Euclidean)"

# 模型文件定义
LATEST_MODEL_PATH = "model_latest.pt"
BEST_MODEL_PATH = "model_best.pt"
GLOBAL_BEST_LOSS_FILE = "best_loss_global.txt"


# ----------------------------------------------------------------------------------
# --- II. 辅助函数 ---
# ----------------------------------------------------------------------------------

def log_metadata(data):
    """将参数和结果记录到日志文件，移除评估字段。"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = [f"--- 运行时间: {timestamp} ---"]

    log_entry.append("### 设定参数 (Configuration) ###")
    for key, value in data['params'].items():
        log_entry.append(f"{key}: {value}")

    log_entry.append("\n### 训练结果 (Results) ###")
    log_entry.append(f"最终平均损失 (Final Avg Loss): {data['results']['final_loss']:.4f}")
    log_entry.append(f"历史最低损失 (Global Best Loss): {data['results']['global_best_loss']:.4f}")
    log_entry.append(f"最佳模型文件: {data['results']['best_model_saved']}")
    log_entry.append("-------------------------------------------\n")

    with open(LOG_FILE, 'a') as f:
        f.write('\n'.join(log_entry))
    print(f"-> 元数据和结果已记录到 {LOG_FILE}")


def load_global_best_loss():
    """从文件中加载历史最佳损失，如果文件不存在则返回无穷大。"""
    try:
        with open(GLOBAL_BEST_LOSS_FILE, 'r') as f:
            return float(f.read().strip())
    except (FileNotFoundError, ValueError):
        return float('inf')


def save_global_best_loss(loss):
    """将新的历史最佳损失保存到文件中。"""
    with open(GLOBAL_BEST_LOSS_FILE, 'w') as f:
        f.write(str(loss))


def string_to_numpy(s):
    """将存储的字符串转换为 NumPy 数组"""
    s = s.strip('[]')
    if not s: return np.array([], dtype=np.float32)
    return np.fromstring(s, sep=' ', dtype=np.float32)


def numpy_to_string(arr):
    """将 NumPy 数组转换为可存储的字符串"""
    return np.array2string(arr, separator=' ')[1:-1].strip()


# ----------------------------------------------------------------------------------
# --- III. Phase I: 数据加载、特征工程与归一化 ---
# ----------------------------------------------------------------------------------
def run_phase_i(filename):
    print("\n[Phase I]：数据加载与特征工程开始 (Parquet & Multi-Label)...")
    start_time = time.time()

    try:
        df = pd.read_parquet(filename)
        # 转换体裁列表字符串为 Python 列表
        df['track_genre'] = df['track_genre'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    except FileNotFoundError:
        raise FileNotFoundError(f"致命错误: 找不到 {filename} 文件。请检查文件路径和名称。")
    except Exception as e:
        raise Exception(f"Parquet加载或体裁解析失败: {e}")

    print(f"-> 步骤 1.1: 原始数据加载完成。记录数: {len(df)}。")

    # 定义特征组
    continuous_features = ['popularity', 'duration_ms', 'danceability', 'energy', 'loudness', 'speechiness',
                           'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    categorical_features = ['key', 'mode', 'time_signature', 'explicit']
    id_metadata_features = ['track_id', 'artists', 'album_name', 'track_name']

    # 1. 独立处理 MultiLabelBinarizer (MLP)
    mlb = MultiLabelBinarizer()
    genre_encoded = mlb.fit_transform(df['track_genre'])
    genre_df = pd.DataFrame(genre_encoded, columns=[f'genre_{g}' for g in mlb.classes_])
    genre_features = genre_df.columns.tolist()

    # 2. ColumnTransformer 处理数值和简单分类特征
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), continuous_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    # 3. 执行特征工程
    df_temp = pd.concat(
        [df.drop(columns=['track_genre'], errors='ignore').reset_index(drop=True), genre_df.reset_index(drop=True)],
        axis=1)

    cols_to_process = continuous_features + categorical_features + id_metadata_features
    processed_data = preprocessor.fit_transform(df_temp[cols_to_process])

    # 4. 重建最终 DataFrame
    feature_names_base = continuous_features + list(
        preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))

    df_processed_temp = pd.DataFrame(processed_data, columns=feature_names_base + id_metadata_features)

    df_processed = pd.concat([df_processed_temp.drop(columns=id_metadata_features).reset_index(drop=True),
                              genre_df.reset_index(drop=True),
                              df_processed_temp[id_metadata_features].reset_index(drop=True)], axis=1)

    # 提取最终特征矩阵
    final_features = feature_names_base + genre_features
    X_features = df_processed[final_features].astype(np.float32).values
    INPUT_DIM = X_features.shape[1]

    print(f"-> 步骤 1.3: 特征工程完成。输入特征维度: {INPUT_DIM}。耗时: {time.time() - start_time:.2f} 秒")
    return X_features, df_processed, final_features, INPUT_DIM


# ----------------------------------------------------------------------------------
# --- V. Phase II: 歌单模拟与三元组样本生成 (持久化支持) ---
# ----------------------------------------------------------------------------------
def simulate_coherent_playlist(features_matrix, length, track_ids, M=MAX_SIMILAR_CANDIDATES):
    if features_matrix.shape[0] < length: return []
    playlist_indices = []
    current_index = random.randint(0, features_matrix.shape[0] - 1)
    playlist_indices.append(current_index)

    for _ in range(length - 1):
        current_song_vector = features_matrix[current_index].reshape(1, -1)
        distances = cdist(current_song_vector, features_matrix, metric='euclidean')[0]
        distances[playlist_indices] = np.inf

        M_eff = min(M, features_matrix.shape[0] - len(playlist_indices))
        if M_eff <= 0: break

        closest_m_indices = np.argpartition(distances, M_eff)[:M_eff]
        closest_m_indices = closest_m_indices[distances[closest_m_indices] != np.inf]

        if len(closest_m_indices) > 0:
            current_index = random.choice(closest_m_indices)
            playlist_indices.append(current_index)
        else:
            break

    return [track_ids[i] for i in playlist_indices if i is not None]


def generate_or_load_triplets(X_features, df_processed, final_features, num_playlists, file_path):
    print("\n[Phase II]：三元组样本生成开始...")

    # 1. 检查文件是否已存在
    if os.path.exists(file_path):
        print(f"-> 步骤 2.0: 检测到已存在的样本文件: {file_path}。正在加载...")
        start_time = time.time()

        df_triplets = pd.read_csv(file_path)
        for col in ['anchor', 'positive', 'negative']:
            df_triplets[col] = df_triplets[col].astype(str).apply(string_to_numpy)

        print(f"-> 步骤 2.0: 样本加载完成。总样本数: {len(df_triplets)}。耗时: {time.time() - start_time:.2f} 秒")
        return df_triplets

    # 2. 文件不存在，执行生成
    print("-> 步骤 2.0: 未找到样本文件，正在执行耗时的歌单模拟...")
    start_time = time.time()

    triplet_samples = []
    track_ids_map = df_processed['track_id'].values

    for i in range(num_playlists):
        current_playlist_length = random.randint(MIN_PLAYLIST_LENGTH, MAX_PLAYLIST_LENGTH)
        if current_playlist_length <= NUM_TARGETS: continue
        current_input_length = current_playlist_length - NUM_TARGETS

        simulated_playlist_ids = simulate_coherent_playlist(X_features, current_playlist_length, track_ids_map)

        if len(simulated_playlist_ids) < current_playlist_length: continue

        df_anchor = df_processed[df_processed['track_id'].isin(simulated_playlist_ids[:current_input_length])]
        anchor_vector = df_anchor[final_features].values.mean(axis=0, dtype=np.float32)

        negative_candidates = df_processed[~df_processed['track_id'].isin(simulated_playlist_ids)]
        if negative_candidates.empty: continue

        df_target = df_processed[df_processed['track_id'].isin(simulated_playlist_ids[current_input_length:])]
        target_vectors = df_target[final_features].values

        N_NEGATIVES = 5
        N_NEGATIVES = min(N_NEGATIVES, len(negative_candidates))
        random_neg_indices = np.random.choice(negative_candidates.index, N_NEGATIVES, replace=False)
        negative_vectors = negative_candidates.loc[random_neg_indices][final_features].values

        for target_vec in target_vectors:
            for neg_vec in negative_vectors:
                triplet_samples.append({'anchor': anchor_vector, 'positive': target_vec, 'negative': neg_vec})

        if (i + 1) % 100 == 0 or i == num_playlists - 1:
            print(f"-> 步骤 2.1: 已处理 {i + 1}/{num_playlists} 个模拟歌单。当前样本总数: {len(triplet_samples)}。")

    df_triplets = pd.DataFrame(triplet_samples)

    # 3. 存储三元组样本
    if not df_triplets.empty:
        df_save = df_triplets.copy()
        for col in ['anchor', 'positive', 'negative']:
            df_save[col] = df_save[col].apply(numpy_to_string)

        df_save.to_csv(file_path, index=False)
        print(
            f"-> 步骤 2.2: 三元组样本生成完成并保存到 {file_path}。最终样本总数: {len(df_triplets)}。耗时: {time.time() - start_time:.2f} 秒")
    else:
        print("-> 步骤 2.2: 警告：未生成任何有效样本，未保存。")

    return df_triplets


# ----------------------------------------------------------------------------------
# --- V. PyTorch 模型定义 (Phase III) ---
# ----------------------------------------------------------------------------------
class TripletDataset(Dataset):
    def __init__(self, df):
        self.anchors = torch.tensor(df['anchor'].tolist(), dtype=torch.float32)
        self.positives = torch.tensor(df['positive'].tolist(), dtype=torch.float32)
        self.negatives = torch.tensor(df['negative'].tolist(), dtype=torch.float32)

    def __len__(self): return len(self.anchors)

    def __getitem__(self, idx): return self.anchors[idx], self.positives[idx], self.negatives[idx]


class EmbeddingNet(nn.Module):
    def __init__(self, input_dim, embedding_dim=EMBEDDING_DIM):
        super(EmbeddingNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x): return self.net(x)


class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance_metric = nn.PairwiseDistance(p=2)

    def forward(self, anchor, positive, negative):
        d_ap = self.distance_metric(anchor, positive)
        d_an = self.distance_metric(anchor, negative)
        loss = torch.relu(d_ap - d_an + self.margin)
        return loss.mean()


# ----------------------------------------------------------------------------------
# --- VI. Phase III 主执行函数 (仅训练和保存) ---
# ----------------------------------------------------------------------------------
def run_phase_iii_and_save_model(df_triplets, INPUT_DIM):
    # 训练准备
    full_dataset = TripletDataset(df_triplets)
    full_dataloader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = EmbeddingNet(INPUT_DIM, EMBEDDING_DIM)
    criterion = TripletLoss(margin=MARGIN)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    TOTAL_BATCHES = len(full_dataloader)

    global_best_loss = load_global_best_loss()

    print("\n[Phase III]：PyTorch 模型训练开始...")
    print(f"-> 步骤 3.1: 模型参数初始化完成。历史最低损失 (Global Best): {global_best_loss:.4f}")

    # 训练循环
    start_time = time.time()
    final_loss = 0
    best_model_saved = LATEST_MODEL_PATH

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        epoch_start_time = time.time()

        for batch_idx, (anchor, positive, negative) in enumerate(full_dataloader):
            anchor_embed = model(anchor)
            positive_embed = model(positive)
            negative_embed = model(negative)
            loss = criterion(anchor_embed, positive_embed, negative_embed)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 100 == 0 or batch_idx == TOTAL_BATCHES - 1:
                print(f"-> Epoch {epoch + 1}: Batch {batch_idx + 1}/{TOTAL_BATCHES} - Loss: {loss.item():.6f}")

        avg_loss = total_loss / TOTAL_BATCHES
        print(
            f"-> 步骤 3.2: Epoch {epoch + 1}/{NUM_EPOCHS} 训练结束。平均损失: {avg_loss:.4f}。耗时: {time.time() - epoch_start_time:.2f} 秒")

        # --- 模型保存与追踪逻辑 ---

        # 1. 保存最新模型 (latest)
        torch.save(model.state_dict(), LATEST_MODEL_PATH)

        # 2. 保存历史全局最佳模型 (best)
        if avg_loss < global_best_loss:
            global_best_loss = avg_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            save_global_best_loss(global_best_loss)
            best_model_saved = BEST_MODEL_PATH
            print(f"-> 模型保存: 发现新的**历史全局最佳**损失 ({global_best_loss:.4f})，模型已保存至 {BEST_MODEL_PATH}")

    final_loss = avg_loss
    print(f"-> 步骤 3.3: 训练完成。总耗时: {time.time() - start_time:.2f} 秒")

    # ------------------- 记录元数据 -------------------
    metadata = {
        'params': {
            'N_RECORDS': N_RECORDS, 'NUM_PLAYLISTS_TO_SIMULATE': NUM_PLAYLISTS_TO_SIMULATE,
            'MIN_PLAYLIST_LENGTH': MIN_PLAYLIST_LENGTH, 'MAX_PLAYLIST_LENGTH': MAX_PLAYLIST_LENGTH,
            'NUM_EPOCHS': NUM_EPOCHS, 'BATCH_SIZE': BATCH_SIZE, 'LR': LR,
            'LOSS_FUNCTION': LOSS_FUNCTION, 'EMBEDDING_DIM': EMBEDDING_DIM, 'MARGIN': MARGIN,
            'INPUT_DIM': INPUT_DIM
        },
        'results': {
            'final_loss': final_loss,
            'global_best_loss': global_best_loss,
            'avg_recall': 'N/A (Evaluation Skipped)',
            'avg_ndcg': 'N/A (Evaluation Skipped)',
            'best_model_saved': best_model_saved
        }
    }
    log_metadata(metadata)


# ----------------------------------------------------------------------------------
# --- VII. 主执行逻辑 ---
# ----------------------------------------------------------------------------------
if __name__ == "__main__":

    print(f"目标文件: {DATASET_FILENAME} | 日志文件: {LOG_FILE}")

    # 运行 Phase I
    try:
        X_features, df_processed, final_features, INPUT_DIM = run_phase_i(DATASET_FILENAME)
    except Exception as e:
        print(f"致命错误：Phase I 失败。{e}")
        exit()

    # 运行 Phase II (生成或加载三元组样本)
    df_triplets = generate_or_load_triplets(X_features, df_processed, final_features, NUM_PLAYLISTS_TO_SIMULATE,
                                            TRIPLET_FILE)

    if not df_triplets.empty:
        # 运行 Phase III (仅训练和保存)
        run_phase_iii_and_save_model(df_triplets, INPUT_DIM)
    else:
        print("--- 流程终止 ---：由于没有生成有效的三元组样本，训练和保存无法进行。")