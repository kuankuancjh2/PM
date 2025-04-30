import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import argparse
import os
import time
import logging

class PreEmbedDataset(Dataset):
    def __init__(self, prompts, responses, tokenizer, embedder, device, max_length=128):
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.device = device
        self.prompts = prompts
        self.responses = responses
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        response = self.responses[idx]

        encoded_prompt = self.tokenizer(prompt, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        encoded_response = self.tokenizer(response, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        input_ids = encoded_prompt['input_ids'].to(self.device)
        attention_mask = encoded_prompt['attention_mask'].to(self.device)

        with torch.no_grad():
            prompt_emb = self.embedder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.squeeze(0)

        return {
            'prompt_emb': prompt_emb.cpu(),
            'response_ids': encoded_response['input_ids'].squeeze(0)
        }

def load_everyday_conversations_ja(tokenizer, embedder, device, max_length=128):
    dataset = load_dataset("U23-lab/everyday_conversations_ja", split="train")
    prompts, responses = [], []
    for i, row in enumerate(dataset):
        if i >= len(dataset) * 0.05:  # 只取前5%的数据
            break
        try:
            topic = row["topic"].strip()
            user = row["user"].strip()
            assistant = row["assistant"].strip()
            if user and assistant:
                prompts.append(f"{topic} | {user}")
                responses.append(assistant)
        except Exception as e:
            logging.warning(f"Error in row: {row}, error: {e}")
    logging.info(f"Loaded {len(prompts)} conversations")
    return PreEmbedDataset(prompts, responses, tokenizer, embedder, device, max_length=max_length)

class Denoiser(nn.Module):
    def __init__(self, latent_dim, cond_dim, seq_len):
        super().__init__()
        # 修改为处理嵌入的线性层
        self.fc1 = nn.Linear(latent_dim + cond_dim, latent_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.seq_len = seq_len

    def forward(self, latent, cond):
        # 在特征维度拼接嵌入 (假设 latent: [batch, seq_len, latent_dim])
        x = torch.cat([latent, cond], dim=-1)
        x = self.act(self.fc1(x))
        return latent + self.fc2(x)

class MaskDecoder(nn.Module):
    def __init__(self, latent_dim, num_experts, seq_len):
        super().__init__()
        # 输出每个专家的选择概率
        self.fc = nn.Linear(latent_dim, num_experts)
        self.seq_len = seq_len

    def forward(self, sL):
        return self.fc(sL)  # [batch, seq_len, num_experts]

class Expert(nn.Module):
    def __init__(self, latent_dim, seq_len):
        super().__init__()
        # 专家网络处理嵌入
        self.fc = nn.Linear(latent_dim, latent_dim)
        self.seq_len = seq_len

    def forward(self, x):
        return self.fc(x)

class DiffusionTextModel(nn.Module):
    def __init__(self, latent_dim, cond_dim, num_experts, vocab_size, seq_len, batch):
        super().__init__()
        self.qa = Denoiser(latent_dim, cond_dim, seq_len)
        self.qb = Denoiser(latent_dim, cond_dim + latent_dim, seq_len)
        self.qk_pool = nn.ModuleList([Expert(latent_dim, seq_len) for _ in range(num_experts)])
        self.mask_decoder = MaskDecoder(latent_dim, num_experts, seq_len)
        self.text_decoder = nn.Linear(latent_dim, vocab_size)
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.batch = batch

    def forward(self, prompt_emb, n_steps=1, mask_denoise_steps=1):
        prompt_emb = prompt_emb.to("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化潜在嵌入 (添加了序列长度维度)
        L = torch.randn([prompt_emb.size(0), self.seq_len, self.latent_dim]).to(prompt_emb.device) * 0.02

        for _ in range(n_steps):
            L = self.qa(L, prompt_emb)  # 去噪步骤1

            # 生成mask的潜在表示
            sL = torch.randn_like(L) * 0.02
            for _ in range(mask_denoise_steps):
                sL = self.qb(sL, torch.cat([prompt_emb.squeeze(1), L], dim=-1))

            # 专家混合机制
            mask = torch.softmax(self.mask_decoder(sL), dim=-1)  # [batch, seq_len, num_experts]
            experts = torch.stack([qk(L) for qk in self.qk_pool], dim=2)  # [batch, seq_len, num_experts, latent_dim]
            L = torch.einsum('bsk,bskd->bsd', mask, experts)  # 加权求和

        # 最终输出logits
        logits = self.text_decoder(L)  # [batch, seq_len, vocab_size]
        return logits, mask

# === 损失 ===
def compute_loss(logits, targets, mask, expert_outputs, lambdas):
    valid = (targets != 0) & (targets != 101) & (targets != 102)
    loss_text = F.cross_entropy(logits.view(-1, logits.size(-1))[valid.view(-1)],
                                targets.view(-1)[valid.view(-1)],
                                ignore_index=0)
    mask_probs = torch.softmax(mask, dim=-1)
    loss_entropy = -(mask_probs * torch.log(mask_probs + 1e-8)).sum(dim=-1).mean()
    usage = mask_probs.sum(dim=(0, 1))
    usage = usage / (usage.sum() + 1e-8)
    loss_diversity = ((usage - 1 / mask.size(-1)) ** 2).sum()
    return lambdas['text'] * loss_text + lambdas['entropy'] * loss_entropy + lambdas['diversity'] * loss_diversity

# === 训练 ===
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    embedder = AutoModel.from_pretrained(args.model_name, cache_dir=args.cache_dir).to(device).eval()

    # 数据加载保持不变，允许batch_size > 1
    dataset = load_everyday_conversations_ja(tokenizer, embedder, device, max_length=args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    model = DiffusionTextModel(embedder.config.hidden_size, embedder.config.hidden_size, args.num_experts, tokenizer.vocab_size, args.max_length, args.batch_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lambdas = {'text': args.lambda_text, 'entropy': args.lambda_entropy, 'diversity': args.lambda_diversity}

    scaler = torch.amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu", enabled=torch.cuda.is_available())

    for epoch in range(args.epochs):
        model.train()
        for batch in loader:

            optimizer.zero_grad()

            with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu", enabled=torch.cuda.is_available()):
                # batch['prompt_emb']: (batch_size, seq_len, hidden_size)
                # batch['response_ids']: (batch_size, seq_len)

                # 拆解批次并逐个处理
                batch_logits, batch_mask = [], []
                for i in range(batch['prompt_emb'].size(0)):  # 遍历每个样本
                    # 取出单个样本 (移除批次维度)
                    prompt_emb = batch['prompt_emb'][i]  # (seq_len, hidden_size)
                    targets = batch['response_ids'][i]   # (seq_len,)

                    # 模型前向 (无批次)
                    logits, mask = model(prompt_emb.unsqueeze(0), n_steps=args.n_steps, mask_denoise_steps=args.mask_denoise_steps)

                    # 保存结果（重新添加批次维度）
                    batch_logits.append(logits.squeeze(0))  # (seq_len, vocab_size)
                    batch_mask.append(mask.squeeze(0))     # (seq_len, num_experts)

                # 重新聚合批次
                logits = torch.stack(batch_logits, dim=0)  # (batch_size, seq_len, vocab_size)
                mask = torch.stack(batch_mask, dim=0)      # (batch_size, seq_len, num_experts)
                targets = batch['response_ids'].to(device) # (batch_size, seq_len)

                # 计算损失
                loss = compute_loss(logits, targets, mask, None, lambdas)

            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if epoch % 1 == 0 and epoch > 0:
            #torch.save(model.state_dict(), os.path.join(args.save_path, f"model_epoch{epoch}.pt"))
            print(f"Epoch {epoch}: loss = {loss.item():.4f}")

        if epoch % 2 == 0 and epoch > 0:
            sample_prompt = "こんにちは。"
            generated = generate_text(model, sample_prompt, tokenizer, embedder, device, args.n_steps)
            print(f"[Prompt] {sample_prompt}")
            print(f"[Generated] {generated}")

        if epoch % 10 == 0 and epoch > 0:
            print("Manual prompt input mode (20s timeout):")
            try:
                start_time = time.time()
                while True:
                    if time.time() - start_time > 20:
                        break
                    prompt = input("Prompt (or 'exit' to skip): ")
                    if prompt.strip().lower() == "exit":
                        break
                    if prompt.strip():
                        result = generate_text(model, prompt, tokenizer, embedder, device, args.n_steps)
                        print("[Generated]", result)
                        start_time = time.time()
            except Exception as e:
                print("Manual prompt input skipped.", e)


@torch.no_grad()
def generate_text(model, prompt_text, tokenizer, embedder, device, n_steps=30, temperature=1.0, top_k=None, top_p=None):
    model.eval()
    inputs = tokenizer(prompt_text, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    prompt_emb = embedder(**inputs).last_hidden_state.squeeze(0)  # (seq_len, embed_dim)
    seq_len, _ = prompt_emb.size()

    prompt_emb = prompt_emb.repeat(model.batch, 1, 1)

    latent_dim = model.qa.fc1.in_features - prompt_emb.size(-1)
    latent = torch.randn(prompt_emb.size(0), seq_len, latent_dim, device=device)

    for _ in range(n_steps):
        latent = model.qa(latent, prompt_emb)
        sL = torch.randn_like(latent)
        for _ in range(3):
            sL = model.qb(sL, torch.cat([prompt_emb, latent], dim=-1))
        mask_logits = model.mask_decoder(sL)
        mask = torch.softmax(mask_logits, dim=-1)

        experts = torch.stack([qk(latent) for qk in model.qk_pool], dim=2)
        latent = torch.einsum('bsk,bskd->bsd', mask, experts)

    logits = model.text_decoder(latent)
    logits = logits / temperature

    if top_k is not None:
        top_k = max(top_k, 1)
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float('-inf'))

    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for i in range(logits.size(0)):
            logits[i, sorted_indices[i][sorted_indices_to_remove[i]]] = float('-inf')

    probs = torch.softmax(logits, dim=-1)
    generated_ids = torch.multinomial(probs[0], num_samples=1).squeeze(-1)

    generated_text = tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="cl-tohoku/bert-base-japanese-v2")
    parser.add_argument("--cache_dir", type=str, default="hf_cache")
    parser.add_argument("--save_path", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_experts", type=int, default=12)
    parser.add_argument("--n_steps", type=int, default=2)
    parser.add_argument("--mask_denoise_steps", type=int, default=4)
    parser.add_argument("--lambda_text", type=float, default=1.0)
    parser.add_argument("--lambda_entropy", type=float, default=0.1)
    parser.add_argument("--lambda_diversity", type=float, default=0.1)

    args = parser.parse_args([])
    train(args)