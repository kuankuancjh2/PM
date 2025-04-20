from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

AutoModelForCausalLM.from_pretrained("gpt2", cache_dir="D:/hf_cache")
AutoTokenizer.from_pretrained("gpt2", cache_dir="D:/hf_cache")
AutoModel.from_pretrained("gpt2", cache_dir="D:/hf_cache")