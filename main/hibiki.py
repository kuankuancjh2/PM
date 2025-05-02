import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
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
        self.fc1 = nn.Linear(latent_dim + cond_dim + time_dim, latent_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.seq_len = seq_len

    def forward(self, latent, cond, t_embed):
        t_embed = t_embed.expand(-1, latent.size(1), -1)
        x = torch.cat([latent, cond, t_embed], dim=-1)
        x = self.act(self.fc1(x))
        return latent + self.fc2(x)

class MaskDecoder(nn.Module):
    def __init__(self, latent_dim, num_experts, seq_len):
        super().__init__()
        self.fc = nn.Linear(latent_dim, num_experts)
        self.seq_len = seq_len

    def forward(self, sL):
        return self.fc(sL)

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
        self.qb = Denoiser(latent_dim, cond_dim + latent_dim, time_dim, seq_len)
        self.qk_pool = nn.ModuleList([Expert(latent_dim, seq_len) for _ in range(num_experts)])
        self.mask_decoder = MaskDecoder(latent_dim, num_experts, seq_len)
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

        return L, mask

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
    preds = torch.argmax(logits, dim=-1)  # [B, L]
    # 每个样本中有多少重复token
    repeat_penalty = 0.0
    for seq in preds:
        uniq = len(set(seq.tolist()))
        total = len(seq)
        repeat_ratio = 1.0 - (uniq / total)
        repeat_penalty += repeat_ratio
    repeat_penalty = repeat_penalty / preds.size(0)  # 平均重复率
    return lambdas['text'] * loss_text + lambdas['entropy'] * loss_entropy + lambdas['diversity'] * loss_diversity + lambdas['repeat'] * repeat_penalty

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

    dataset = load_everyday_conversations_ja(tokenizer, embedder, device, max_length=args.max_length, percentage=args.data_percentage)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    model = DiffusionTextModel(
        latent_dim=embedder.config.hidden_size,
        cond_dim=embedder.config.hidden_size,
        num_experts=args.num_experts,
        seq_len=args.max_length,
    ).to(device)

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
        model.load_state_dict(torch.load(model_path, map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lambdas = {'text': args.lambda_text, 'entropy': args.lambda_entropy, 'diversity': args.lambda_diversity}

    for epoch in range(start_epoch + 1, target_epoch + 1):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            prompt_embs = batch['prompt_emb'].to(device)
            targets = batch['response_ids'].to(device)
            latent, mask = model(prompt_embs, n_steps=args.n_steps, mask_denoise_steps=args.mask_denoise_steps)
            logits = gpt2(inputs_embeds=latent).logits

            vocab_size = logits.size(-1)
            valid = (targets != 0) & (targets != 101) & (targets != 102) & (targets < vocab_size)
            loss_text = F.cross_entropy(logits.view(-1, vocab_size)[valid.view(-1)], targets.view(-1)[valid.view(-1)], ignore_index=0)

            mask_probs = torch.softmax(mask, dim=-1)
            loss_entropy = -(mask_probs * torch.log(mask_probs + 1e-8)).sum(dim=-1).mean()
            usage = mask_probs.sum(dim=(0, 1))
            usage = usage / (usage.sum() + 1e-8)
            loss_diversity = ((usage - 1 / mask.size(-1)) ** 2).sum()

            loss = lambdas['text'] * loss_text + lambdas['entropy'] * loss_entropy + lambdas['diversity'] * loss_diversity
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}: loss = {loss.item():.4f}")
            
        if epoch % 50 == 0:
            os.makedirs(args.save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.save_path, f"model_epoch{epoch}.pt"))

        if epoch % 2 == 0:
            sample_prompt = "こんにちは。"
            generated = generate(model, gpt2, tokenizer, embedder, sample_prompt, device)
            print(f"[Prompt] {sample_prompt}")
            print(f"[Generated] {generated}")
            
        if epoch % 50 == 0 and epoch > 0:
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
def generate(model, gpt2_model, tokenizer, embedder, prompt, device, max_length=128, n_steps=20, mask_denoise_steps=4):
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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_experts", type=int, default=12)
    parser.add_argument("--n_steps", type=int, default=1)
    parser.add_argument("--mask_denoise_steps", type=int, default=2)
    parser.add_argument("--lambda_text", type=float, default=1.0)
    parser.add_argument("--lambda_entropy", type=float, default=0.1)
    parser.add_argument("--lambda_diversity", type=float, default=0.1)
    parser.add_argument("--lambda_repeat", type=float, default=0.2)
    parser.add_argument("--data_percentage", type=list, default=[0.2, 0.21])
    args = parser.parse_args([])

    train(args)
