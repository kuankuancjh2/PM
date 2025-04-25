import torch
import os
import argparse
from transformers import AutoTokenizer, AutoModel
from hibiki import DiffusionTextModel, generate_text

def find_latest_ckpt(path):
    ckpts = sorted([f for f in os.listdir(path) if f.endswith(".pt")])
    return os.path.join(path, ckpts[-1]) if ckpts else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="checkpoint")
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir="D:/my_data/hf_cache")
    embedder = AutoModel.from_pretrained("bert-base-uncased", cache_dir="D:/my_data/hf_cache").to(device)

    latent_dim = embedder.config.hidden_size
    num_experts = 12
    vocab_size = tokenizer.vocab_size
    model = DiffusionTextModel(latent_dim, latent_dim, num_experts, vocab_size, embedder).to(device)

    ckpt = find_latest_ckpt(args.save_path)
    if not ckpt:
        print(f"No checkpoint found in {args.save_path}")
        return
    model.load_state_dict(torch.load(ckpt))
    print(f"✅ Loaded checkpoint: {ckpt}")

    while True:
        prompt = input("Enter a prompt (or type 'exit'): ").strip()
        if prompt.lower() == "exit":
            break
        result = generate_text(model, [prompt], tokenizer, n_steps=args.n_steps, device=device, temperature=args.temperature)
        print(f"\n🔹 Generated Text:\n{result[0]}\n")

if __name__ == "__main__":
    main()
