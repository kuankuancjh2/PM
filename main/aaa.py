import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import time

# --------------------------
# 设备与数据配置
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# 模型统计工具
# --------------------------
def print_model_stats(model):
    print("\n模型统计信息:")
    total_params = 0
    trainable_params = 0
    
    # 按模块类型聚合统计
    stats = {}
    for name, param in model.named_parameters():
        module_type = name.split('.')[0]
        if module_type not in stats:
            stats[module_type] = {'count':0, 'size':0}
        stats[module_type]['count'] += 1
        stats[module_type]['size'] += param.numel()
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    # 单行打印每个模块
    for module, data in stats.items():
        print(f"{module}: {data['count']}层/{data['size']:,}参数", end=" | ")
    
    print(f"\n总计: {total_params:,}参数 (可训练: {trainable_params:,})")

def generate_samples(model, embedder, tokenizer, sample_texts):
    model.eval()
    print("\n生成文本预览:")
    with torch.no_grad():
        for i, text in enumerate(sample_texts[:3]):  # 只显示前3个样本
            inputs = tokenizer(text, return_tensors="pt").to(device)
            emb = embedder(**inputs).last_hidden_state
            
            # 添加噪声并去噪
            t = torch.tensor([500], device=device)  # 中间步数
            noise = torch.randn_like(emb)
            noisy_emb = emb * 0.5 + noise * 0.5
            pred_emb, _ = model(noisy_emb, emb, t)
            
            # 解码生成文本
            logits = pred_emb @ embedder.get_input_embeddings().weight.t()
            pred_ids = torch.argmax(logits, dim=-1)
            generated = tokenizer.decode(pred_ids[0].cpu().numpy())
            
            print(f"\n输入: {text[:60]}...")
            print(f"生成: {generated[:60]}...")
            print("-"*50)

# --------------------------
# 时间嵌入模块
# --------------------------
class TimestepEmbedding(nn.Module):
    def __init__(self, dim, scale=16):
        super().__init__()
        self.scale = scale
        half_dim = dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        self.register_buffer('emb', emb)
        self.proj = nn.Linear(dim, dim)

    def forward(self, t):
        emb = t[:, None].float() * self.emb[None, :] * self.scale
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.proj(emb)

# --------------------------
# 动态Mask生成器（改）
# --------------------------
class MaskNet(nn.Module):
    def __init__(self, dim, num_qk, temperature=1.0):
        super().__init__()
        self.dim = dim
        self.num_qk = num_qk
        self.temperature = nn.Parameter(torch.tensor(temperature))

        self.content_proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )

        self.position_proj = nn.Linear(dim, dim)
        self.expert_embed = nn.Embedding(num_qk, dim)
        self.ctrl_net = nn.Sequential(
            nn.Linear(dim * 2, num_qk * 4),
            nn.Softplus(),
            nn.Linear(num_qk * 4, num_qk),
            nn.Sigmoid()
        )

    def forward(self, latent, prompt_emb, t_emb):
        B, L, D = latent.shape
        content = self.content_proj(latent + prompt_emb)

        # 增加位置相关的上下文动态权重
        pos_weights = self.position_proj(latent)
        ctx = content + pos_weights

        global_t = t_emb.unsqueeze(1).expand(-1, L, -1)  # [B, L, D]
        combined = torch.cat([ctx, global_t], dim=-1)  # [B, L, 2D]

        ctrl_weight = self.ctrl_net(combined)  # [B, L, num_qk]
        gate = F.softmax(ctrl_weight / self.temperature.clamp(0.3, 3.0), dim=-1)
        return gate, self._entropy_loss(gate)

    def _entropy_loss(self, mask):
        entropy = -(mask * torch.log(mask + 1e-8)).sum(dim=-1)
        return (1.0 - entropy / math.log(self.num_qk)).mean()

# --------------------------
# 专家Qk模块
# --------------------------
class QkBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(dim * 2, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )
        self.style = nn.Parameter(torch.randn(dim))

    def forward(self, x, prompt_emb):
        h = torch.cat([x, prompt_emb], dim=-1)
        h = self.feature_net(h)
        return h * torch.sigmoid(self.style.view(1, 1, -1))

# --------------------------
# 扩散模型
# --------------------------
class SelectiveDiffusion(nn.Module):
    def __init__(self, dim, num_qk):
        super().__init__()
        self.t_embed = TimestepEmbedding(dim)
        self.mask_net = MaskNet(dim, num_qk)
        self.q_blocks = nn.ModuleList([QkBlock(dim) for _ in range(num_qk)])
        self.decoder = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, latent, prompt_emb, t):
        t_emb = self.t_embed(t)
        mask, ent_loss = self.mask_net(latent, prompt_emb, t_emb)
        q_outputs = torch.stack([
            block(latent, prompt_emb) for block in self.q_blocks
        ], dim=-1)  # [B, L, D, num_qk]
        fused = torch.einsum('blq,bldq->bld', mask, q_outputs)
        return self.decoder(fused), mask, ent_loss
    
# --------------------------
# 数据加载与嵌入处理
# --------------------------
class DiffusionDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data, embedder, tokenizer, max_length=128):
        self.data = list(tokenized_data)  # 确保转换为列表
        self.embedder = embedder
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        with torch.no_grad():
            # 获取原始文本
            input_ids = item['input_ids'][:self.max_length]
            original_text = self.tokenizer.decode(input_ids)
            
            # 数据增强：随机替换部分token
            if np.random.rand() < 0.1:  # 10%的概率进行替换
                replace_indices = np.random.choice(len(input_ids), size=2, replace=False)
                for i in replace_indices:
                    input_ids[i] = np.random.randint(self.tokenizer.vocab_size)
            
            # 确保输入是张量
            if isinstance(input_ids, list):
                input_ids = torch.tensor(input_ids)
            if isinstance(item['attention_mask'], list):
                attention_mask = torch.tensor(item['attention_mask'][:self.max_length])
            else:
                attention_mask = item['attention_mask'][:self.max_length]
            
            # 获取文本嵌入
            inputs = {
                'input_ids': input_ids.unsqueeze(0).to(device),
                'attention_mask': attention_mask.unsqueeze(0).to(device)
            }
            emb = self.embedder(**inputs).last_hidden_state.squeeze(0)
            
            # 生成噪声和timestep
            t = torch.randint(0, 1000, (1,)).item()
            noise = torch.randn_like(emb)
            noisy_emb = self._add_noise(emb, noise, t/1000)
            
            return {
                'noisy_latent': noisy_emb.cpu(),
                'prompt_emb': emb.cpu(),
                't': torch.tensor(t),
                'target': emb.cpu(),
                'original_text': original_text,
                'input_ids': input_ids  # 保存原始input_ids用于生成
            }

    def _add_noise(self, x, noise, alpha):
        return alpha * x + (1 - alpha) * noise

# --------------------------
# 训练函数
# --------------------------
def train(args):
    # 初始化模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    embedder = AutoModel.from_pretrained("gpt2").to(device)
    embedder.eval()

    # 加载数据集并转换为列表
    dataset = load_from_disk(args.dataset_path)
    train_data = list(dataset["train"].select(range(int(len(dataset["train"]) * args.sample_pct))))
    
    # 创建数据加载器
    train_dataset = DiffusionDataset(train_data, embedder, tokenizer, args.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: {
            'noisy_latent': torch.nn.utils.rnn.pad_sequence(
                [item['noisy_latent'] for item in batch],
                batch_first=True,
                padding_value=0
            ),
            'prompt_emb': torch.nn.utils.rnn.pad_sequence(
                [item['prompt_emb'] for item in batch],
                batch_first=True,
                padding_value=0
            ),
            't': torch.stack([item['t'] for item in batch]),
            'target': torch.nn.utils.rnn.pad_sequence(
                [item['target'] for item in batch],
                batch_first=True,
                padding_value=0
            ),
            'original_text': [item['original_text'] for item in batch],
            'input_ids': torch.nn.utils.rnn.pad_sequence(
                [item['input_ids'] for item in batch],
                batch_first=True,
                padding_value=tokenizer.pad_token_id
            )
        }
    )

    # 初始化模型
    model = SelectiveDiffusion(
        dim=args.dim,
        num_qk=args.num_qk
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs*len(train_loader))
    
    # 打印初始模型统计信息
    print("="*50)
    print("初始模型统计信息:")
    print_model_stats(model)
    print("="*50)

    # 训练循环
    for epoch in range(args.epochs):
        # 动态调整温度（从1.5线性衰减到0.7）
        current_temp = max(0.7, 1.5 - epoch * 0.008)
        model.mask_net.temperature.data.fill_(current_temp)
        
        # 动态熵系数（从0.3线性增加到0.8）
        current_entropy_coef = min(0.8, 0.3 + epoch * 0.005)
        model.mask_net.entropy_coef = current_entropy_coef

        
        # 动态调整策略
        model.mask_net.temperature.data.clamp_(0.5, 2.0)
        
        model.train()   
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        # 增强的统计变量
        epoch_stats = {
            'loss': [], 'mse': [], 'cos': [], 'entropy': [],
            'expert_participation': [],  # 各专家参与比例
            'grad_norm': [], 'emb_sim': [],
            'min_expert': [], 'max_expert': []  # 最小/最大专家利用率
        }
        
        for batch in pbar:
            # 准备数据
            noisy_latent = batch['noisy_latent'].to(device)
            prompt_emb = batch['prompt_emb'].to(device)
            t = batch['t'].to(device)
            target = batch['target'].to(device)
            
            # 前向计算
            optimizer.zero_grad()
            pred_emb, mask, entropy_loss = model(noisy_latent, prompt_emb, t)
            
            # 改进的损失计算（增强多样性约束）
            cos_loss = 1 - F.cosine_similarity(pred_emb, target, dim=-1).mean()
            mse_loss = F.mse_loss(pred_emb, target)
            
            # 动态平衡两种损失
            alpha = min(1.0, epoch/10)  # 逐步增加cosine权重
            recon_loss = (1-alpha)*mse_loss + alpha*cos_loss
            
            # 增强的熵约束（基于当前epoch调整强度）
            entropy_weight = args.entropy_coef * min(1.0, epoch/3)  # 加快熵约束权重的增长
            total_loss = recon_loss + entropy_weight * entropy_loss
            
            # 反向传播
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # 增强的监控指标
            with torch.no_grad():
                # 专家参与度统计
                expert_participation = (mask > 0.05).float().mean(dim=[0,1])  # [num_qk]
                min_expert = expert_participation.min().item()
                max_expert = expert_participation.max().item()
                
                # 更新统计
                epoch_stats['loss'].append(total_loss.item())
                epoch_stats['mse'].append(mse_loss.item())
                epoch_stats['cos'].append(cos_loss.item())
                epoch_stats['entropy'].append(entropy_loss.item())
                epoch_stats['expert_participation'].append(expert_participation.cpu().numpy())
                epoch_stats['grad_norm'].append(grad_norm.item())
                epoch_stats['emb_sim'].append(1 - cos_loss.item())
                epoch_stats['min_expert'].append(min_expert)
                epoch_stats['max_expert'].append(max_expert)
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f"{np.mean(epoch_stats['loss'][-20:]):.4f}",
                    'mse': f"{np.mean(epoch_stats['mse'][-20:]):.4f}",
                    'cos': f"{np.mean(epoch_stats['cos'][-20:]):.4f}",
                    'ent': f"{np.mean(epoch_stats['entropy'][-20:]):.4f}",
                    'min_exp': f"{np.mean(epoch_stats['min_expert'][-20:]):.2f}",
                    'max_exp': f"{np.mean(epoch_stats['max_expert'][-20:]):.2f}",
                    'grad': f"{grad_norm:.1e}"
                })
                
        print(f"\nEpoch {epoch} 诊断报告:")
        print(f"Loss: {np.mean(epoch_stats['loss']):.4f} ± {np.std(epoch_stats['loss']):.4f}")
        print(f"  - MSE: {np.mean(epoch_stats['mse']):.4f} | Cos: {np.mean(epoch_stats['cos']):.4f}")
        print(f"  - Entropy: {np.mean(epoch_stats['entropy']):.4f}")
        print(f"专家最小利用率: {np.mean(epoch_stats['min_expert']):.2%}")
        print(f"专家最大利用率: {np.mean(epoch_stats['max_expert']):.2%}")
        print(f"梯度范数: {np.mean(epoch_stats['grad_norm']):.2f} ± {np.std(epoch_stats['grad_norm']):.2f}")
        print(f"嵌入相似度: {np.mean(epoch_stats['emb_sim']):.4f}")
        
        # Mask深度分析
        print("\n[Mask分析]")
        print(f"- 样例分布 (前3个token):")
        print(mask[0, :30].detach().cpu().numpy().round(3))
        
        # 专家激活热力图
        expert_heatmap = mask.mean(dim=1).mean(dim=0).detach().cpu().numpy()
        print("- 专家激活热力:")
        print("   " + " ".join(f"Exp{i}:{v:.3f}" for i,v in enumerate(expert_heatmap)))
        
        # 文本生成预览
        if epoch % 2 == 0:
            print("\n[生成示例]")
            with torch.no_grad():
                for i in range(min(3, len(batch['original_text']))):
                    input_text = batch['original_text'][i][:100] + "..."
                    inputs = tokenizer(input_text, return_tensors="pt").to(device)
                    emb = embedder(**inputs).last_hidden_state
                    
                    # 分步生成过程
                    noisy = emb * 0.3 + torch.randn_like(emb) * 0.7
                    generated, _, _ = model(noisy, emb, torch.tensor([500], device=device))
                    
                    # 解码
                    logits = generated @ embedder.get_input_embeddings().weight.t()
                    pred_ids = torch.argmax(logits, dim=-1)
                    output_text = tokenizer.decode(pred_ids[0].cpu().numpy())
                    
                    print(f"\n输入: {input_text}")
                    print(f"生成: {output_text[:100]}...")
                    print("-"*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="D:/my_data/processed/tokenized_dataset")
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--num_qk", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--sample_pct", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--entropy_coef", type=float, default=0.1)
    parser.add_argument("--save_freq", type=int, default=5)
    args = parser.parse_args()
    
    train(args)
