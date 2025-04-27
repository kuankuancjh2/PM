import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import argparse
import os
import time

# === 数据集定义 ===

class DiffusionDataset(Dataset):
    def __init__(self, tokenized_data, embedder, tokenizer, max_length=128, device='cpu', sample_pct=1.0):
        self.data = list(tokenized_data)
        if sample_pct < 1.0:
            sample_len = int(len(self.data) * sample_pct)
            self.data = self.data[:sample_len]
        self.embedder = embedder
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = item['input_ids'][:self.max_length].clone()
        attention_mask = item['attention_mask'][:self.max_length].clone()

        original_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)

        if np.random.rand() < 0.1:
            replace_indices = np.random.choice(len(input_ids), size=2, replace=False)
            for i in replace_indices:
                input_ids[i] = np.random.randint(self.tokenizer.vocab_size)

        inputs = {
            'input_ids': input_ids.unsqueeze(0).to(self.device),
            'attention_mask': attention_mask.unsqueeze(0).to(self.device)
        }
        with torch.no_grad():
            emb = self.embedder(**inputs).last_hidden_state.squeeze(0)

        t = torch.randint(0, 1000, (1,)).item()
        noise = torch.randn_like(emb)
        noisy_emb = self._add_noise(emb, noise, t/1000)

        return {
            'L': noisy_emb.cpu(),
            'L_target': emb.cpu(),
            'mL': torch.randn_like(emb).cpu(),
            'mL_target': emb.cpu(),
            'prompt': emb.cpu(),
            'text': input_ids.cpu(),
            't': torch.tensor(t),
            'original_text': original_text
        }

    def _add_noise(self, x, noise, alpha):
        return alpha * x + (1 - alpha) * noise

# === 核心模块定义 ===

class Denoiser(nn.Module):
    def __init__(self, latent_dim, cond_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, latent, cond):
        x = torch.cat([latent, cond], dim=-1)
        return self.net(x)

class MaskDecoder(nn.Module):
    def __init__(self, latent_dim, num_experts):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, num_experts)
        )

    def forward(self, sL):
        logits = self.decoder(sL)
        mask = torch.softmax(logits, dim=-1)
        return mask

class Expert(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x):
        return self.net(x)

# === 主模型结构 ===

class DiffusionTextModel(nn.Module):
    def __init__(self, latent_dim, prompt_dim, num_experts, vocab_size, embedder):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_experts = num_experts

        self.qa = Denoiser(latent_dim, prompt_dim + latent_dim)
        self.qb = Denoiser(latent_dim, prompt_dim + latent_dim)
        self.qc = Denoiser(latent_dim, prompt_dim + latent_dim)

        self.qk_pool = nn.ModuleList([Expert(latent_dim) for _ in range(num_experts)])
        self.mask_decoder = MaskDecoder(latent_dim, num_experts)

        self.text_decoder = nn.Linear(latent_dim, vocab_size)
        
        self.embedder = embedder

    def forward(self, L, mL, prompt):
        # mL 表示模型记忆，可视为对当前对话上下文的内隐表示
        # prompt 表示当前输入（本轮对话）

        # 1. 用 mL 和 prompt 生成本轮的 L（内容生成基础）
        L = self.qa(L, torch.cat([prompt, mL], dim=-1))

        # 2. 对 L 加噪，生成 sL，输入给 mask 解码器
        sL = self.qb(torch.randn_like(L), torch.cat([prompt, L], dim=-1))
        mask = self.mask_decoder(sL)

        # 3. 用 mask 混合多个专家输出
        expert_outputs = []
        for k, qk in enumerate(self.qk_pool):
            out = qk(L)
            expert_outputs.append(out.unsqueeze(-2))
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        qk_output = (expert_outputs * mask.unsqueeze(-1)).sum(dim=-2)

        # 4. 再次精炼 L，同时保持 mL 参与其中
        L = self.qa(qk_output, torch.cat([prompt, mL], dim=-1))

        # 5. 更新对话上下文记忆 mL
        mL_pred = self.qc(mL, torch.cat([prompt, L], dim=-1))

        # 6. 最终解码生成文本
        text_logits = self.text_decoder(L)

        return L, mL_pred, mask, text_logits, expert_outputs

# === 损失函数 ===

def compute_losses(L_pred, L_target, mL_pred, mL_target, text_logits, text_target, mask, expert_outputs, lambda_dict):
    loss_L = F.mse_loss(L_pred, L_target)
    loss_mL = F.mse_loss(mL_pred, mL_target)
    loss_text = F.cross_entropy(text_logits.view(-1, text_logits.size(-1)), text_target.view(-1))

    loss_mask_entropy = (mask * torch.log(mask + 1e-8)).sum(dim=-1).mean()

    mask_usage = mask.sum(dim=1)
    mask_usage = mask_usage / (mask_usage.sum(dim=-1, keepdim=True) + 1e-6)
    loss_diversity = ((mask_usage.mean(dim=0) - 1.0/mask.size(-1))**2).sum()

    i, j = 0, 1
    ek1 = expert_outputs[:, :, i, :]
    ek2 = expert_outputs[:, :, j, :]
    loss_contrast = -F.cosine_similarity(ek1.detach(), ek2, dim=-1).mean()

    total_loss = (loss_L +
                  lambda_dict['mL'] * loss_mL +
                  lambda_dict['text'] * loss_text +
                  lambda_dict['entropy'] * loss_mask_entropy +
                  lambda_dict['diversity'] * loss_diversity +
                  lambda_dict['contrast'] * loss_contrast)

    return total_loss

# === 训练循环 ===

def train_model(model, dataloader, optimizer, lambda_dict, device, tokenizer, num_epochs=100, previous_epoch=0, save_path="checkpoint"):
    model.train()
    os.makedirs(save_path, exist_ok=True)
    for epoch in range(previous_epoch, num_epochs):
        for batch in dataloader:
            L = batch['L'].to(device)
            mL = batch['mL'].to(device)
            prompt = batch['prompt'].to(device)
            text_target = batch['text'].to(device)
            L_target = batch['L_target'].to(device)
            mL_target = batch['mL_target'].to(device)

            optimizer.zero_grad()
            L_pred, mL_pred, mask, text_logits, expert_outputs = model(L, mL, prompt)

            loss = compute_losses(L_pred, L_target, mL_pred, mL_target, text_logits, text_target, mask, expert_outputs, lambda_dict)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
            sample_prompt = batch['original_text'][:2]
            sampled_texts = generate_text(model, sample_prompt, tokenizer, n_steps=20, device=device)
            for i, txt in enumerate(sampled_texts):
                print(f"[Prompt {i}] {sample_prompt[i]}")
                print(f"[Generated {i}] {txt}")

        if epoch % 20 == 0:
            print(f"\n[Epoch {epoch}] Lambda weights: {lambda_dict}")
            with torch.no_grad():
                example_mask = mask[0]
                for pos in range(min(3, example_mask.size(0))):
                    weights = example_mask[pos].cpu().numpy()
                    print(f"  Position {pos} expert weights: {weights}")

        if epoch % 50 == 0  and epoch >= 100:
            ckpt = os.path.join(save_path, f"model_epoch{epoch}.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"Saved checkpoint to {ckpt}")

            print("Input a prompt within 20s to test generation (or wait to continue):")
            print("Type 'exit' to stop prompt input and continue training.")
            try:
                start_time = time.time()
                while True:
                    if time.time() - start_time > 20:
                        break
                    prompt = input("Prompt: ")
                    if prompt.strip().lower() == "exit":
                        break
                    if prompt.strip():
                        result = generate_text(model, [prompt], tokenizer, n_steps=30, device=device)
                        print("\nGenerated text:", result[0])
                        start_time = time.time()
            except Exception as e:
                print("Prompting skipped.", e)

@torch.no_grad()
def generate_text(model, prompt_texts, tokenizer, n_steps=10, device='cpu', temperature=1.0):
    model.eval()
    inputs = tokenizer(prompt_texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    prompt_emb = model.embedder(**inputs).last_hidden_state

    # mL 初始化为空记忆；一次 session 中可复用
    mL = torch.zeros_like(prompt_emb).to(device)
    latent = torch.randn_like(prompt_emb).to(device)

    for step in range(n_steps):
        L = model.qa(latent, torch.cat([prompt_emb, mL], dim=-1))
        sL = model.qb(torch.randn_like(L), torch.cat([prompt_emb, L], dim=-1))
        mask = model.mask_decoder(sL)

        expert_outputs = []
        for k, qk in enumerate(model.qk_pool):
            out = qk(L)
            expert_outputs.append(out.unsqueeze(-2))
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        qk_output = (expert_outputs * mask.unsqueeze(-1)).sum(dim=-2)

        latent = model.qa(qk_output, torch.cat([prompt_emb, mL], dim=-1))
        mL = model.qc(mL, torch.cat([prompt_emb, latent], dim=-1))  # 更新记忆 mL

    logits = model.text_decoder(latent)
    probs = torch.softmax(logits / temperature, dim=-1)
    ids = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(probs.size(0), -1)
    texts = [tokenizer.decode(x.tolist(), skip_special_tokens=True) for x in ids]
    return texts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_pct", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_path", type=str, default="checkpoint")
    parser.add_argument("--resume", action="store_false", help="Resume training from last checkpoint")
    args = parser.parse_args()  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir="D:/my_data/hf_cache")
    embedder = AutoModel.from_pretrained("bert-base-uncased", cache_dir="D:/my_data/hf_cache").to(device)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]", cache_dir="D:/my_data/hf_cache")
    
    
    tokenized = dataset.map(
        lambda e: tokenizer(e['text'], truncation=True, padding="max_length", max_length=128),
        batched=True,
        remove_columns=['text']  # 可选，清理原始字段
    ).with_format("torch")
    diffusion_dataset = DiffusionDataset(tokenized, embedder, tokenizer, max_length=128, device=device, sample_pct=args.sample_pct)
    dataloader = DataLoader(diffusion_dataset, batch_size=args.batch_size, shuffle=True)

    latent_dim = embedder.config.hidden_size
    prompt_dim = latent_dim
    num_experts = 12
    vocab_size = tokenizer.vocab_size
    model = DiffusionTextModel(latent_dim, prompt_dim, num_experts, vocab_size, embedder=embedder).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lambda_dict = {
        'mL': 1.0,
        'text': 1.0,
        'entropy': 0.1,
        'diversity': 0.1,
        'contrast': 0.1
    }
    
    previous_epoch = 0
    args.resume = False
                
    if args.resume:
        ckpt_files = [f for f in os.listdir(args.save_path) if f.endswith(".pt")]
        if ckpt_files:
            # Extract epoch numbers and find the maximum
            max_epoch = -1
            latest_ckpt = None
            for f in ckpt_files:
                try:
                    epoch = int(f.split("_")[-1].split(".")[0][5:])  # Assuming format like "model_epochXX.pt"
                    if epoch > max_epoch:
                        max_epoch = epoch
                        latest_ckpt = f
                except (IndexError, ValueError):
                    continue
            
            if latest_ckpt:
                latest_ckpt_path = os.path.join(args.save_path, latest_ckpt)
                model.load_state_dict(torch.load(latest_ckpt_path))
                print(f"✅ Resumed model from {latest_ckpt_path}")

    train_model(model, dataloader, optimizer, lambda_dict, device, tokenizer, num_epochs=args.epochs + previous_epoch, previous_epoch=previous_epoch, save_path=args.save_path)

if __name__ == "__main__":
    main()
