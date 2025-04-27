import torch
import os
import argparse
from transformers import AutoTokenizer, AutoModel
from hibiki import DiffusionTextModel, generate_text

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

    ckpts = sorted([f for f in os.listdir(args.save_path) if f.endswith(".pt")])
    if ckpts:
        # Extract epoch numbers and find the maximum
        max_epoch = -1
        latest_ckpt = None
        for f in ckpts:
            try:
                epoch = int(f.split("_")[-1].split(".")[0][5:])
                if epoch > max_epoch:
                    max_epoch = epoch
                    latest_ckpt = f
            except (IndexError, ValueError):
                continue
    
    if latest_ckpt:
        latest_ckpt_path = os.path.join(args.save_path, latest_ckpt)
        print(f"Resuming model from {latest_ckpt_path}")
        model.load_state_dict(torch.load(latest_ckpt_path))
        print(f"✅ Resumed model from {latest_ckpt_path}")

    while True:
        prompt = input("Enter a prompt (or type 'exit'): ").strip()
        if prompt.lower() == "exit":
            break
        result = generate_text(model, [prompt], tokenizer, n_steps=args.n_steps, device=device, temperature=args.temperature)
        print(f"\n🔹 Generated Text:\n{result[0]}\n")

if __name__ == "__main__":
    main()
