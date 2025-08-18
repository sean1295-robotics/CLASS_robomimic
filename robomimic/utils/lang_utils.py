import os
import torch
from transformers import AutoModel, pipeline, AutoTokenizer, CLIPTextModelWithProjection

os.environ["TOKENIZERS_PARALLELISM"] = "true" # needed to suppress warning about potential deadlock

device = "cuda" if torch.cuda.is_available() else "cpu"
# tokenizer = "openai/clip-vit-large-patch14" #"openai/clip-vit-base-patch32"
# lang_emb_model = CLIPTextModelWithProjection.from_pretrained(
#     tokenizer,
#     cache_dir=os.path.expanduser(os.path.join(os.environ.get("HF_HOME", "~/tmp"), "clip"))
# ).eval().to(device)
# tz = AutoTokenizer.from_pretrained(tokenizer, TOKENIZERS_PARALLELISM=True)

tokenizer = "distilbert-base-uncased"
lang_emb_model = AutoModel.from_pretrained("distilbert-base-uncased", torch_dtype=torch.float16).eval().to(device)
tz = AutoTokenizer.from_pretrained(tokenizer, TOKENIZERS_PARALLELISM=True)

LANG_EMB_OBS_KEY = "lang_emb"

def get_lang_emb(lang):
    if lang is None:
        return None
    
    tokens = tz(
        text=lang,                   # the sentence to be encoded
        add_special_tokens=True,             # Add [CLS] and [SEP]
        max_length=25,  # maximum length of a sentence
        padding="max_length",
        return_attention_mask=True,        # Generate the attention mask
        return_tensors="pt",               # ask the function to return PyTorch tensors
    )
    tokens = {k:v.to(device) for k,v in tokens.items()}
    lang_emb = lang_emb_model(**tokens).last_hidden_state.sum(1).squeeze().cpu()
    # lang_emb = lang_emb_model(**tokens)['text_embeds'].detach()[0]

    return lang_emb

# def batch_get_lang_emb(lang):
#     assert isinstance(lang, list), "Input should be a list of strings"

#     # Get unique strings to avoid redundant computation
#     unique_lang = sorted(list(set(lang)))

#     # Process only the unique strings
#     unique_tokens = tz(
#         text=unique_lang,
#         add_special_tokens=True,
#         max_length=77,
#         padding="max_length",
#         truncation=True,
#         return_attention_mask=True,
#         return_tensors="pt",
#     )
#     unique_tokens = {k: v.to(device) for k, v in unique_tokens.items()}
#     unique_lang_emb = lang_emb_model(**unique_tokens).last_hidden_state.sum(1)

#     # Create a mapping from each unique string to its embedding
#     emb_map = {text: emb for text, emb in zip(unique_lang, unique_lang_emb)}

#     # Populate the final list by looking up embeddings for each element
#     lang_emb = torch.stack([emb_map[s] for s in lang])

#     return lang_emb

def batch_get_lang_emb(lang):
    assert isinstance(lang, list), "Input should be a list of strings"

    tokens = tz(
        text=lang,
        add_special_tokens=True,
        max_length=77,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}
    lang_emb = lang_emb_model(**tokens).last_hidden_state.sum(1)
    return lang_emb

def get_lang_emb_shape():
    return list(get_lang_emb('dummy').shape)