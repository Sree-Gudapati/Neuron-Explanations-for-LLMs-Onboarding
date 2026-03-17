"""
Neuron Compositional Explanation Analysis

This script extracts neuron activations and uses a beam search to find 
compositional logical explanations for neuron behavior. 

The core feature extraction logic and Intersection over Union (IoU) 
scoring methodology were adapted from the COMPEXP codebase:
Original Authors: Mu and Andreas (2020)
Repository: https://github.com/jayelm/compexp
"""


import torch, re
import numpy as np, pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from scipy.spatial.distance import cdist

MODEL_PATH = "/mnt/data/gemma-2-2b-it"
VECPATH = "/mnt/data/compexp/nli/data/analysis/snli_1.0_dev.vec"
COMPLEXITY_PENALTY = 0.9
MIN_ACTS = 10
N_NEIGHBORS = 10

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.float16)

df = pd.read_csv("/mnt/data/compexp/nli/data/analysis/snli_1.0_dev.txt", sep='\t')
df_clean = df[df['gold_label'].isin(['entailment', 'neutral', 'contradiction'])]

sentences_list = []
for i in range(1000):
    row = df_clean.iloc[i]
    sentences_list.append((row['sentence1'], row['sentence2']))

all_sentence_activations = []

def get_activation(module, input, output):
    if isinstance(output, tuple):
        hidden_states = output[0]
    else:
        hidden_states = output
    
    peaks = hidden_states.max(dim=1)[0]
    
    clean_numbers = peaks.detach().cpu().numpy()
    all_sentence_activations.append(clean_numbers)

handle = model.model.layers[25].mlp.register_forward_hook(get_activation)

feature_patterns = {} 
SKIP_WORDS = {"a", "an", "the", "of", ".", ",", "is", "are", "UNK", "PAD"}

for i in range(len(sentences_list)):
    premise, hypothesis = sentences_list[i]
    
    p_words = premise.lower().split()
    h_words = hypothesis.lower().split()
    
    current_features = []
    for w in p_words:
        if w not in SKIP_WORDS:
            current_features.append(f"pre:tok:{w}")
    for w in h_words:
        if w not in SKIP_WORDS:
            current_features.append(f"hyp:tok:{w}")
            
    p_set, h_set = set(p_words), set(h_words)
    overlap_score = len(p_set & h_set) / (len(p_set | h_set) + 1e-5)
    if overlap_score > 0.25: current_features.append("oth:overlap:overlap25")
    if overlap_score > 0.50: current_features.append("oth:overlap:overlap50")
    if overlap_score > 0.75: current_features.append("oth:overlap:overlap75")

    for f in current_features:
        if f not in feature_patterns:
            feature_patterns[f] = np.zeros(1000, dtype=bool)
        feature_patterns[f][i] = True

    text = f"Premise: {premise} Hypothesis: {hypothesis}"
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        model(**inputs)

handle.remove()

def load_vecs(path):
    vecs, stoi, itos = [], {}, {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            tok, nums = parts[0], np.array(parts[1:], dtype=np.float32)
            idx = len(stoi)
            stoi[tok] = idx
            itos[idx] = tok
            vecs.append(nums)
    return np.array(vecs), stoi, itos

VECS, VECS_STOI, VECS_ITOS = load_vecs(VECPATH)
NEIGHBORS_CACHE = {}

def get_neighbors(word):
    if word not in VECS_STOI:
        return []
    if word in NEIGHBORS_CACHE:
        return NEIGHBORS_CACHE[word]
    idx = VECS_STOI[word]
    dists = cdist(VECS[idx][np.newaxis], VECS, metric="cosine")[0]
    nearest = [VECS_ITOS[i] for i in np.argsort(dists)[1:N_NEIGHBORS + 1]]
    NEIGHBORS_CACHE[word] = nearest
    return nearest

for feat_name, feat_pat in list(feature_patterns.items()):
    if ":tok:" not in feat_name:
        continue
    prefix, _, word = feat_name.split(":", 2)
    neighbor_union = np.zeros(1000, dtype=bool)
    for nbr in get_neighbors(word):
        nbr_key = f"{prefix}:tok:{nbr}"
        if nbr_key in feature_patterns:
            neighbor_union |= feature_patterns[nbr_key]
    if neighbor_union.any():
        feature_patterns[f"{prefix}:neighbors:{word}"] = neighbor_union

acts_matrix = np.vstack(all_sentence_activations)

thresholds = np.percentile(acts_matrix, 90, axis=0)
neuron_fired_binary = acts_matrix > thresholds

weights_matrix = model.score.weight.detach().cpu().float().numpy().T
num_labels = weights_matrix.shape[1]
def w(neuron, i): return float(weights_matrix[neuron, i]) if i < num_labels else 0.0

def calculate_iou(pattern_a, pattern_b):
    intersection = np.logical_and(pattern_a, pattern_b).sum()
    union = np.logical_or(pattern_a, pattern_b).sum()
    return intersection / (union + 1e-8)

final_results = []
all_known_features = list(feature_patterns.items())
MAX_FORMULA_LENGTH = 5
BEAM_SIZE = 5 
for neuron_id in tqdm(range(100), desc="Analyzing Neurons"):
    this_neuron_pattern = neuron_fired_binary[:, neuron_id]

    if this_neuron_pattern.sum() < MIN_ACTS:
        final_results.append({"neuron": neuron_id, "explanation": "NULL", "iou_score": 0.0, "best_noncomp": "NULL", "noncomp_iou": 0.0, "w_entail": w(neuron_id, 0), "w_neutral": w(neuron_id, 1), "w_contra": w(neuron_id, 2)})
        continue

    current_best_candidates = []
    for name, pattern in all_known_features:
        score = calculate_iou(this_neuron_pattern, pattern)
        current_best_candidates.append((name, pattern, score))
    
    current_best_candidates = sorted(current_best_candidates, key=lambda x: x[2], reverse=True)[:BEAM_SIZE]

    best_noncomp_name, _, best_noncomp_score = current_best_candidates[0]

    for length in range(MAX_FORMULA_LENGTH - 1):
        new_growth_ideas = []
        
        for existing_name, existing_pattern, _ in current_best_candidates:
            for word_name, word_pattern in all_known_features:
                
                new_name = f"({existing_name} AND {word_name})"
                new_pattern = existing_pattern & word_pattern
                new_score = calculate_iou(this_neuron_pattern, new_pattern) * (COMPLEXITY_PENALTY ** length)
                new_growth_ideas.append((new_name, new_pattern, new_score))
                

                new_name = f"({existing_name} OR {word_name})"
                new_pattern = existing_pattern | word_pattern
                new_score = calculate_iou(this_neuron_pattern, new_pattern) * (COMPLEXITY_PENALTY ** length)
                new_growth_ideas.append((new_name, new_pattern, new_score))

                new_name = f"({existing_name} AND (NOT {word_name}))"
                new_pattern = existing_pattern & (~word_pattern)
                new_score = calculate_iou(this_neuron_pattern, new_pattern) * (COMPLEXITY_PENALTY ** length)
                new_growth_ideas.append((new_name, new_pattern, new_score))

        current_best_candidates = sorted(new_growth_ideas, key=lambda x: x[2], reverse=True)[:BEAM_SIZE]

    winner_name, _, winner_score = current_best_candidates[0]
    
    final_results.append({
        "neuron": neuron_id,
        "explanation": winner_name,
        "iou_score": winner_score,
        "best_noncomp": best_noncomp_name,
        "noncomp_iou": best_noncomp_score,
        "w_entail": w(neuron_id, 0),
        "w_neutral": w(neuron_id, 1),
        "w_contra": w(neuron_id, 2)
    })
df_results = pd.DataFrame(final_results)
df_results.to_csv("my_neuron_analysis.csv", index=False)
print("done")
