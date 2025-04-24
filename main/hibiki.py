import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset


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
        input_ids = item['input_ids'][:self.max_length]
        original_text = self.tokenizer.decode(input_ids)

        if np.random.rand() < 0.1:
            replace_indices = np.random.choice(len(input_ids), size=2, replace=False)
            for i in replace_indices:
                input_ids[i] = np.random.randint(self.tokenizer.vocab_size)

        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(item['attention_mask'][:self.max_length])

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
    def __init__(self, latent_dim, prompt_dim, num_experts, vocab_size):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_experts = num_experts

        self.qa = Denoiser(latent_dim, prompt_dim + latent_dim)
        self.qb = Denoiser(latent_dim, prompt_dim + latent_dim)
        self.qc = Denoiser(latent_dim, prompt_dim + latent_dim)

        self.qk_pool = nn.ModuleList([Expert(latent_dim) for _ in range(num_experts)])
        self.mask_decoder = MaskDecoder(latent_dim, num_experts)

        self.text_decoder = nn.Linear(latent_dim, vocab_size)

    def forward(self, L, mL, prompt):
        L = self.qa(L, torch.cat([prompt, mL], dim=-1))
        sL = self.qb(torch.randn_like(L), torch.cat([prompt, L], dim=-1))
        mask = self.mask_decoder(sL)

        expert_outputs = []
        for k, qk in enumerate(self.qk_pool):
            out = qk(L)
            expert_outputs.append(out.unsqueeze(-2))
        expert_outputs = torch.cat(expert_outputs, dim=-2)

        mask_expanded = mask.unsqueeze(-1)
        qk_output = (expert_outputs * mask_expanded).sum(dim=-2)

        L = self.qa(qk_output, torch.cat([prompt, mL], dim=-1))
        mL_pred = self.qc(mL, torch.cat([prompt, L], dim=-1))
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

def train_model(model, dataloader, optimizer, lambda_dict, device, tokenizer, preview_interval=100):
    model.train()
    for step, batch in enumerate(dataloader):
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

        if step % preview_interval == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")
            #preds = torch.argmax(text_logits, dim=-1)
            #for i in range(min(1, preds.size(0))):
            #    print("[Preview] Text:", tokenizer.decode(preds[i]))
            
            sample_prompt = batch['prompt'][:2].to(device)
            sampled_texts = generate_text(model, sample_prompt, tokenizer, n_steps=100, device=device)
            for i, txt in enumerate(sampled_texts):
                try:
                    print(f"[Prompt {i}] {tokenizer.decode(batch['prompt'][i], skip_special_tokens=True)}")
                except Exception as e:
                    print (f"[Prompt {i}] {e}")
                print(f"[Sample {i}] {txt}")


@torch.no_grad()
def generate_text(model, prompt_emb, tokenizer, n_steps=100, device='cuda' if torch.cuda.is_available() else 'cpu', max_len=128):
    model.eval()
    latent = torch.randn_like(prompt_emb).to(device)
    mL = torch.randn_like(prompt_emb).to(device)

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
        mL = model.qc(mL, torch.cat([prompt_emb, latent], dim=-1))

    logits = model.text_decoder(latent)
    ids = torch.argmax(logits, dim=-1)
    texts = [tokenizer.decode(x, skip_special_tokens=True) for x in ids]
    return texts

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载Tokenizer与Embedder
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir="D:/my_data/hf_cache")
    embedder = AutoModel.from_pretrained("bert-base-uncased", cache_dir="D:/my_data/hf_cache").to(device)

    # 加载与预处理数据
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]", cache_dir="D:/my_data/hf_cache")  # 用较小数据子集调试
    tokenized = dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding="max_length", max_length=128), batched=True)
    diffusion_dataset = DiffusionDataset(tokenized, embedder, tokenizer, max_length=128, device=device, sample_pct=0.1)
    dataloader = DataLoader(diffusion_dataset, batch_size=8, shuffle=True)

    # 初始化模型
    latent_dim = embedder.config.hidden_size
    prompt_dim = latent_dim
    num_experts = 12
    vocab_size = tokenizer.vocab_size
    model = DiffusionTextModel(latent_dim, prompt_dim, num_experts, vocab_size).to(device)

    # 优化器与权重系数
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lambda_dict = {
        'mL': 1.0,
        'text': 1.0,
        'entropy': 0.1,
        'diversity': 0.1,
        'contrast': 0.1
    }

    num_epochs = 1000
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}")
        # 开始训练
        train_model(model, dataloader, optimizer, lambda_dict, device, tokenizer, preview_interval=10)

if __name__ == "__main__":
    main()
