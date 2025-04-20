# train_selective_diffusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from tqdm import tqdm
import os

# --- 模型结构 ---
class SimpleDenoiser(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, z, t):
        B, K, D = z.shape
        t_embed = t / 1000.0
        t_embed = torch.full((B, K, 1), t_embed, device=z.device)
        x = torch.cat([z, t_embed], dim=-1)
        return self.net(x)


def default_schedule(t, K):
    active = torch.zeros(K, dtype=torch.bool)
    active[t % K] = True
    return active


class SelectiveDiffusion(nn.Module):
    def __init__(self, K=6, D=768, hidden_dim=1024, schedule_fn=default_schedule):
        super().__init__()
        self.K = K
        self.D = D
        self.schedule_fn = schedule_fn
        self.denoiser = SimpleDenoiser(D, hidden_dim)

    def forward(self, z_noisy, t):
        B, K, D = z_noisy.shape
        active_mask = self.schedule_fn(t, K).to(z_noisy.device)
        active_mask = active_mask.unsqueeze(0).unsqueeze(-1)
        z_input = z_noisy * active_mask
        z_denoised = self.denoiser(z_input, t)
        z_updated = z_noisy * (~active_mask) + z_denoised * active_mask
        return z_updated


# --- 数据准备 ---
def embed_tokens(batch, embedder, tokenizer, device, K=6):
    ids = torch.tensor(batch["input_ids"]).to(device)  # (B, T)
    with torch.no_grad():
        emb = embedder(input_ids=ids).last_hidden_state  # (B, T, D)
    B, T, D = emb.shape
    L = T // K
    zk = torch.stack([emb[:, i * L:(i + 1) * L].mean(dim=1) for i in range(K)], dim=1)  # (B, K, D)
    return zk


# --- 解码器辅助函数 ---
def preview_generation(zk, embedder, tokenizer, device):
    """将 denoised zk 还原为句子，使用 MLP 解码为 token 级 embedding"""
    with torch.no_grad():
        B, K, D = zk.shape
        T = 96  # 生成长度（token-level embedding）

        # 解码器嵌入器（简易）：MLP 将 K 个 zk 解码为 T 个 token embedding
        mlp = nn.Sequential(
            nn.Linear(K, T),
            nn.ReLU(),
            nn.Linear(T, T)
        ).to(device)

        zk_transposed = zk.transpose(1, 2)  # (B, D, K)
        token_embeds = mlp(zk_transposed)  # (B, D, T)
        token_embeds = token_embeds.transpose(1, 2)  # (B, T, D)

        decoder = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
        decoder.eval()

        attention_mask = torch.ones(token_embeds.shape[:-1], dtype=torch.long).to(device)
        output = decoder.generate(
            inputs_embeds=token_embeds,
            attention_mask=attention_mask,
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1.0
        )
        texts = tokenizer.batch_decode(output, skip_special_tokens=True)
        return texts


# --- 训练过程 ---
def train(resume_from_latest=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = "D:/my_data/processed/tokenized_dataset"
    dataset = load_from_disk(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    embedder = AutoModel.from_pretrained("gpt2").to(device)
    embedder.eval()

    train_set = dataset["train"].select(range(int(len(dataset["train"]) * 0.05))).with_format("torch")  # 使用1%子集训练
    loader = DataLoader(train_set, batch_size=8, shuffle=True)

    model = SelectiveDiffusion(K=6, D=768).to(device)

    # --- 如果开启恢复功能，自动加载最新模型 ---
    if resume_from_latest:
        checkpoints = [f for f in os.listdir('.') if f.startswith('diff_model_epoch') and f.endswith('.pt')]
        if checkpoints:
            latest_ckpt = max(checkpoints, key=lambda x: int(x.split('epoch')[1].split('.pt')[0]))
            model.load_state_dict(torch.load(latest_ckpt))
            print(f"✅ 已加载模型权重: {latest_ckpt}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.95), weight_decay=0.01)

    T = 20  # diffusion steps
    from transformers import get_cosine_schedule_with_warmup
    total_steps = len(loader) * 10  # 假设最大训练轮数为 10
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),  # warmup 前期加速
        num_training_steps=total_steps,
        num_cycles=4
    )

    for epoch in range(10):  # 可以适当延长训练轮数
        for batch in tqdm(loader, desc=f"Epoch {epoch}"):
            zk_gt = embed_tokens(batch, embedder, tokenizer, device, K=6)
            noise = torch.randn_like(zk_gt)
            zk_noisy = zk_gt + noise
            zk_pred = zk_noisy.clone()

            for t in range(T):
                zk_pred = model(zk_pred, t)

            loss = F.mse_loss(zk_pred, zk_gt)

            # --- 调试信息 ---
            with torch.no_grad():
                pred_mean, pred_std = zk_pred.mean().item(), zk_pred.std().item()
                gt_mean, gt_std = zk_gt.mean().item(), zk_gt.std().item()
                tqdm.write(f"Loss: {loss.item():.4f} | Pred μ={pred_mean:.2f} σ={pred_std:.2f} | GT μ={gt_mean:.2f} σ={gt_std:.2f} | lr={optimizer.param_groups[0]['lr']:.6f}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            tqdm.write(f"Loss: {loss.item():.4f}")

        torch.save(model.state_dict(), f"diff_model_epoch{epoch}.pt")

        # --- 每轮后预览生成 ---
        previews = preview_generation(zk_pred[:2], embedder, tokenizer, device)
        print("=== 生成预览 ===")
        for i, text in enumerate(previews):
            print(f"[Sample {i}]: {text}")


if __name__ == "__main__":
    train()
