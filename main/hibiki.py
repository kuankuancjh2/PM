import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from datasets import load_dataset
import argparse
import os
import time
import math

def load_everyday_conversations_ja(tokenizer, embedder, device, max_length=128, percentage=[0, 0.1]):
    dataset = load_dataset("U23-lab/everyday_conversations_ja", split="train")
    prompts, responses = [], []
    for i, row in enumerate(dataset):
        if i < len(dataset) * percentage[0]: continue
        if i >= len(dataset) * percentage[1]: break
        try:
            topic = row["topic"].strip()
            user = row["user"].strip()
            assistant = row["assistant"].strip()
            if user and assistant:
                prompts.append(f"{topic} | {user}")
                responses.append(assistant)
        except Exception as e:
            print(f"Error processing row {i}: {e}")
    return PreEmbedDataset(prompts, responses, tokenizer, embedder, device, max_length)

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

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1).unsqueeze(1)

class Denoiser(nn.Module):
    def __init__(self, latent_dim, cond_dim, time_dim, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim)
        self.fc1 = nn.Linear(latent_dim + cond_dim + time_dim, latent_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.seq_len = seq_len

    def forward(self, latent, cond, t_embed):
        t_embed = t_embed.expand(-1, latent.size(1), -1)
        x = torch.cat([latent, cond, t_embed], dim=-1)
        x = self.act(self.fc1(x))
        return latent + self.fc2(x)

class TransformerDenoiser(nn.Module):
    def __init__(self, latent_dim, cond_dim, time_dim, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(latent_dim + cond_dim + time_dim, latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, latent, cond, t_embed):
        """
        latent: [B, L, D]
        cond: [B, L, D]
        t_embed: [B, 1, T] → expand to [B, L, T]
        """
        t_embed = t_embed.expand(-1, latent.size(1), -1)
        x = torch.cat([latent, cond, t_embed], dim=-1)      # [B, L, D+C+T]
        x = self.input_proj(x)                              # [B, L, D]
        x = self.encoder(self.norm(x))                      # [B, L, D]
        return latent + x                                   # 残差加回


class MaskDecoder(nn.Module):
    def __init__(self, latent_dim, num_experts, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim)
        self.fc1 = nn.Linear(latent_dim, num_experts)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(num_experts, num_experts)
        self.act2 = nn.GELU()
        self.fc3 = nn.Linear(num_experts, num_experts)
        self.seq_len = seq_len
        self.weight_logits = nn.Parameter(torch.randn(3))  # [α, β, γ]

    def forward(self, sL):
        sL0 = self.norm(sL)
        sL1 = self.act1(self.fc1(sL0))     # [B, L, num_experts]
        sL2 = self.act2(self.fc2(sL1))     # [B, L, num_experts]
        sL3 = self.fc3(sL2)                # [B, L, num_experts]

        w = torch.softmax(self.weight_logits, dim=0)  # [3]
        return w[0] * sL1 + w[1] * sL2 + w[2] * sL3
    
class LatentDecoder(nn.Module):
    def __init__(self, latent_dim, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim)
        self.fc1 = nn.Linear(latent_dim, latent_dim)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.act2 = nn.GELU()
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.seq_len = seq_len
        self.weight_logits = nn.Parameter(torch.randn(3))  # [α, β, γ]

    def forward(self, sL):
        sL0 = self.norm(sL)
        sL1 = self.act1(self.fc1(sL0))     # [B, L, seq_len]
        sL2 = self.act2(self.fc2(sL1))     # [B, L, seq_len]
        sL3 = self.fc3(sL2)                # [B, L, seq_len]
        
        w = torch.softmax(self.weight_logits, dim=0)  # [3]
        return w[0] * sL1 + w[1] * sL2 + w[2] * sL3

class Expert(nn.Module):
    def __init__(self, latent_dim, seq_len):
        super().__init__()
        self.fc = nn.Linear(latent_dim, latent_dim)
        self.seq_len = seq_len

    def forward(self, x):
        return self.fc(x)

class DiffusionTextModel(nn.Module):
    def __init__(self, latent_dim, cond_dim, num_experts, seq_len, time_dim=64):
        super().__init__()
        self.time_dim = time_dim
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.qa = Denoiser(latent_dim, cond_dim, time_dim, seq_len)
        self.qb = TransformerDenoiser(latent_dim, cond_dim + latent_dim, time_dim)
        self.qk_pool = nn.ModuleList([Expert(latent_dim, seq_len) for _ in range(num_experts)])
        self.mask_decoder = MaskDecoder(latent_dim, num_experts, seq_len)
        self.latent_decoder = LatentDecoder(latent_dim, seq_len)
        self.seq_len = seq_len
        self.latent_dim = latent_dim

    def forward(self, prompt_emb, n_steps=1, mask_denoise_steps=1):
        device = prompt_emb.device
        B = prompt_emb.size(0)
        L = torch.randn([B, self.seq_len, self.latent_dim], device=device) * 0.02

        for t in range(n_steps):
            t_embed = self.time_embed(torch.full((B,), t, device=device, dtype=torch.long))
            L = self.qa(L, prompt_emb, t_embed)

            sL = torch.randn_like(L) * 0.02
            for k in range(mask_denoise_steps):
                t_embed_k = self.time_embed(torch.full((B,), k, device=device, dtype=torch.long))
                sL = self.qb(sL, torch.cat([prompt_emb.squeeze(1), L], dim=-1), t_embed_k)

            mask = torch.softmax(self.mask_decoder(sL), dim=-1)
            experts = torch.stack([qk(L) for qk in self.qk_pool], dim=2)
            L = torch.einsum('bsk,bskd->bsd', mask, experts)

            logits = self.latent_decoder(L)
            
        return logits, mask

def compute_loss(gpt2, logits, targets, mask, latent, lambdas):
    valid = (targets != 0) & (targets != 101) & (targets != 102)
    loss_text = F.cross_entropy(logits.view(-1, logits.size(-1))[valid.view(-1)],
                                targets.view(-1)[valid.view(-1)],
                                ignore_index=0)
    mask_probs = torch.softmax(mask, dim=-1)
    loss_entropy = -(mask_probs * torch.log(mask_probs + 1e-8)).sum(dim=-1).mean()
    usage = mask_probs.sum(dim=(0, 1))
    usage = usage / (usage.sum() + 1e-8)
    loss_diversity = ((usage - 1 / mask.size(-1)) ** 2).sum()
    preds = torch.argmax(logits, dim=-1)  # [B, L]
    # 每个样本中有多少重复token
    repeat_penalty = 0.0
    for seq in preds:
        uniq = len(set(seq.tolist()))
        total = len(seq)
        repeat_ratio = 1.0 - (uniq / total)
        repeat_penalty += repeat_ratio
    repeat_penalty = repeat_penalty / preds.size(0)  # 平均重复率
    with torch.no_grad():
        gpt2_target_emb = unwrap(gpt2).transformer.wte(targets)  # [B, L, 1024]
    latent_loss = F.mse_loss(latent, gpt2_target_emb)
    return lambdas['text'] * loss_text + lambdas['entropy'] * loss_entropy + lambdas['diversity'] * loss_diversity + lambdas['repeat'] * repeat_penalty + lambdas['latent'] * latent_loss

def unwrap(model):
    return model.module if hasattr(model, "module") else model

@torch.no_grad()
def decode_latent_to_text(latent, gpt2_model, tokenizer, temperature=1.0):
    gpt_input = latent
    logits = gpt2_model(inputs_embeds=gpt_input).logits
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    generated_ids = torch.argmax(probs, dim=-1)
    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    embedder = AutoModel.from_pretrained(args.model_name, cache_dir=args.cache_dir).to(device).eval()
    gpt2 = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    gpt2.resize_token_embeddings(tokenizer.vocab_size)

    for name, param in gpt2.named_parameters():
        if any(key in name for key in ["h.10", "h.11", "ln_f"]):
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    gpt2 = nn.DataParallel(gpt2)

    dataset = load_everyday_conversations_ja(tokenizer, embedder, device, max_length=args.max_length, percentage=args.data_percentage)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    model = DiffusionTextModel(
        latent_dim=embedder.config.hidden_size,
        cond_dim=embedder.config.hidden_size,
        num_experts=args.num_experts,
        seq_len=args.max_length,
    ).to(device)
    
    model = nn.DataParallel(model)

    os.makedirs(args.save_path, exist_ok=True)
    existing_epochs = [
        int(f.replace("model_epoch", "").replace(".pt", ""))
        for f in os.listdir(args.save_path)
        if f.startswith("model_epoch") and f.endswith(".pt")
    ]
    start_epoch = max(existing_epochs) if existing_epochs else 0
    target_epoch = start_epoch + args.epochs
    print(f"Resuming from epoch {start_epoch}, training to {target_epoch}.")

    model_path = os.path.join(args.save_path, f"model_epoch{start_epoch}.pt")
    if os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        unwrap(model).load_state_dict(torch.load(model_path, map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lambdas = {'text': args.lambda_text, 'entropy': args.lambda_entropy, 'diversity': args.lambda_diversity, 'repeat': args.lambda_repeat, 'latent': args.lambda_latent}

    num_training_steps = len(loader) * (target_epoch - start_epoch)
    num_warmup_steps = int(num_training_steps * 0.04)  # 前10%为 warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=0.5  # 半周期，代表衰减到最低再回升一点（可以调节）
    )


    for epoch in range(start_epoch + 1, target_epoch + 1):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            prompt_embs = batch['prompt_emb'].to(device)
            targets = batch['response_ids'].to(device)
            logits, mask = model(prompt_embs, n_steps=args.n_steps, mask_denoise_steps=args.mask_denoise_steps)
            gpt2_logits = gpt2(inputs_embeds=logits).logits

            loss = compute_loss(gpt2, gpt2_logits, targets, mask, logits, lambdas)
            torch.nn.utils.clip_grad_norm_(unwrap(model).parameters(), max_norm=1.0)
            loss.backward()
            optimizer.step()
            scheduler.step()

        print(f"Epoch {epoch}: loss = {loss.item():.4f}, lr = {scheduler.get_last_lr()[0]:.6f}")
        
        with torch.no_grad():
            # 随机取一个 batch 做专家分析
            batch_iter = iter(loader)
            batch = next(batch_iter)
            prompt_embs = batch['prompt_emb'].to(device)
            latent, mask = model(prompt_embs, n_steps=args.n_steps, mask_denoise_steps=args.mask_denoise_steps)

            # [B, L, num_experts]
            mask_probs = torch.softmax(mask, dim=-1)

            # === 统计专家平均使用率 ===
            expert_usage = mask_probs.sum(dim=(0, 1))  # [num_experts]
            expert_usage = expert_usage / expert_usage.sum()
            usage_str = ", ".join([f"{u.item():.3f}" for u in expert_usage])
            print(f"[Expert Usage] {usage_str}")

            # === 检查专家输出之间的余弦相似度 ===
            experts_latent = [qk(latent) for qk in unwrap(model).qk_pool]  # 每个为 [B, L, D]
            avg_latent = [e.mean(dim=(0, 1)) for e in experts_latent]  # 每个为 [D]
            sims = []
            for i in range(len(avg_latent)):
                for j in range(i + 1, len(avg_latent)):
                    sim = F.cosine_similarity(avg_latent[i], avg_latent[j], dim=0).item()
                    sims.append(sim)
            if sims:
                avg_sim = sum(sims) / len(sims)
            print(f"[Expert Cosine Similarity Avg] {avg_sim:.4f}")
            
        if epoch % 15 == 0:
            os.makedirs(args.save_path, exist_ok=True)
            torch.save(unwrap(model).state_dict(), os.path.join(args.save_path, f"model_epoch{epoch}.pt"))

        if epoch % 1 == 0:
            sample_prompt = "こんにちは。"
            generated = generate(model, gpt2, tokenizer, embedder, sample_prompt, device)
            print(f"[Prompt] {sample_prompt}")
            print(f"[Generated] {generated}")
            
        if epoch % 30 == 0 and epoch > 0:
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
                        result = generate(model, gpt2, tokenizer, embedder, prompt, device)
                        print("[Generated]", result)
                        start_time = time.time()
            except Exception as e:
                print("Manual prompt input skipped.", e)


@torch.no_grad()
def generate(model, gpt2_model, tokenizer, embedder, prompt, device, max_length=128, n_steps=1, mask_denoise_steps=1):
    model = unwrap(model)
    gpt2_model = unwrap(gpt2_model)
    
    model.eval()
    gpt2_model.eval()
    inputs = tokenizer(prompt, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length).to(device)
    prompt_emb = embedder(**inputs).last_hidden_state.squeeze(0).unsqueeze(0)
    latent, _ = model(prompt_emb, n_steps=n_steps, mask_denoise_steps=mask_denoise_steps)
    texts = decode_latent_to_text(latent, gpt2_model, tokenizer)
    return texts[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="rinna/japanese-gpt2-medium")
    parser.add_argument("--cache_dir", type=str, default="hf_cache")
    parser.add_argument("--save_path", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_experts", type=int, default=12)
    parser.add_argument("--n_steps", type=int, default=2)
    parser.add_argument("--mask_denoise_steps", type=int, default=3)
    parser.add_argument("--lambda_text", type=float, default=1.0)
    parser.add_argument("--lambda_entropy", type=float, default=0.1)
    parser.add_argument("--lambda_diversity", type=float, default=0.2)
    parser.add_argument("--lambda_repeat", type=float, default=0.2)
    parser.add_argument("--lambda_latent", type=float, default=0.1)
    parser.add_argument("--data_percentage", type=list, default=[0.4, 0.42])
    args = parser.parse_args([])

    train(args)
