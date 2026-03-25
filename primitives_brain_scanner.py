"""
SEMANTIC PRIMITIVES - LAYER-BY-LAYER BRAIN SCANNER (Encoder models)
====================================================================
Scans BERT-style encoder models layer by layer.

Usage:
    py primitives_brain_scanner.py bert-base-uncased
    py primitives_brain_scanner.py bert-large-uncased
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

# ============================================================
# GET MODEL NAME FROM COMMAND LINE
# ============================================================

if len(sys.argv) < 2:
    print("Usage: py primitives_brain_scanner.py <model-name>")
    print("Example: py primitives_brain_scanner.py bert-base-uncased")
    sys.exit(1)

MODEL_NAME = sys.argv[1]
# Create a safe filename tag from model name
FILE_TAG = MODEL_NAME.replace("/", "_").replace("\\", "_")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# SENTENCE PAIRS
# ============================================================

PRIMITIVES = {
    "CAUSATION": [
        ("The window is broken.", "The rock broke the window."),
        ("The forest is burning.", "Lightning set the forest on fire."),
        ("The ice melted.", "The sun melted the ice."),
        ("The boat is sinking.", "The storm sank the boat."),
        ("The road is wet.", "The rain made the road wet."),
        ("The wall collapsed.", "The earthquake collapsed the wall."),
        ("The flowers are dead.", "The frost killed the flowers."),
        ("The bridge is destroyed.", "The flood destroyed the bridge."),
    ],
    "AGENCY": [
        ("The door opened.", "She opened the door."),
        ("The ball rolled down the hill.", "The boy rolled the ball down the hill."),
        ("The fire started in the kitchen.", "He started the fire in the kitchen."),
        ("The letter arrived this morning.", "The mailman delivered the letter this morning."),
        ("The song played softly.", "The musician played the song softly."),
        ("The rope snapped.", "The climber cut the rope."),
        ("The alarm went off.", "The guard triggered the alarm."),
        ("The car stopped suddenly.", "The driver stopped the car suddenly."),
    ],
    "CONTAINMENT": [
        ("The cat sat beside the box.", "The cat sat inside the box."),
        ("The bird landed near the cage.", "The bird was trapped inside the cage."),
        ("He stood next to the building.", "He stood inside the building."),
        ("The fish swam near the net.", "The fish was caught in the net."),
        ("She waited outside the room.", "She waited inside the room."),
        ("The key lay beside the drawer.", "The key lay inside the drawer."),
        ("The dog slept near the house.", "The dog slept inside the house."),
        ("The coin fell beside the jar.", "The coin fell into the jar."),
    ],
    "SEQUENCE": [
        ("She sang and danced.", "She sang and then danced."),
        ("He read and ate.", "He read and then ate."),
        ("They laughed and cried.", "They laughed and then cried."),
        ("The crowd cheered and clapped.", "The crowd cheered and then clapped."),
        ("He slept and dreamed.", "He slept and then dreamed."),
        ("She wrote and edited.", "She wrote and then edited."),
        ("The child ran and fell.", "The child ran and then fell."),
        ("He cooked and served the meal.", "He cooked and then served the meal."),
    ],
    "PROPERTY": [
        ("There is a house on the hill.", "There is a red house on the hill."),
        ("She wore a dress.", "She wore a silk dress."),
        ("He picked up a stone.", "He picked up a heavy stone."),
        ("A dog crossed the road.", "A large dog crossed the road."),
        ("She drank the coffee.", "She drank the bitter coffee."),
        ("A bird sat on the branch.", "A tiny bird sat on the branch."),
        ("He carried a bag.", "He carried a leather bag."),
        ("The river flowed south.", "The deep river flowed south."),
    ],
    "QUANTITY": [
        ("A star appeared in the sky.", "Many stars appeared in the sky."),
        ("A child played in the park.", "Several children played in the park."),
        ("She read a book.", "She read many books."),
        ("A wave hit the shore.", "Countless waves hit the shore."),
        ("He planted a tree.", "He planted dozens of trees."),
        ("A bell rang in the distance.", "Many bells rang in the distance."),
        ("She lit a candle.", "She lit hundreds of candles."),
        ("A bird flew overhead.", "A flock of birds flew overhead."),
    ],
    "CONDITIONALITY": [
        ("The garden blooms in spring.", "If it rains, the garden blooms in spring."),
        ("She passes the exam.", "If she studies hard, she passes the exam."),
        ("The bridge holds the weight.", "If the cables are strong, the bridge holds the weight."),
        ("He gets the job.", "If he interviews well, he gets the job."),
        ("The boat reaches the shore.", "If the wind is right, the boat reaches the shore."),
        ("The team wins.", "If they score first, the team wins."),
        ("The project succeeds.", "If funding continues, the project succeeds."),
        ("The plant grows tall.", "If it gets enough sunlight, the plant grows tall."),
    ],
    "PART_WHOLE": [
        ("The wheel and the car are in the garage.", "The wheel of the car is damaged."),
        ("The handle and the cup sat on the table.", "The handle of the cup broke off."),
        ("The roof and the house look old.", "The roof of the house is leaking."),
        ("The pages and the book are on the shelf.", "The pages of the book are torn."),
        ("The branch and the tree stood in the yard.", "The branch of the tree snapped."),
        ("The wing and the bird are visible.", "The wing of the bird is broken."),
        ("The engine and the plane are ready.", "The engine of the plane is running."),
        ("The leg and the table are wooden.", "The leg of the table is cracked."),
    ],
    "IDENTITY": [
        ("The morning star and the evening star are bright.", "The morning star and the evening star are the same star."),
        ("The man at the door and her father arrived.", "The man at the door and her father are the same person."),
        ("This road and that path go through the forest.", "This road and that path are the same route."),
        ("The singer and the actor performed last night.", "The singer and the actor are the same person."),
        ("The old name and the new name appear on the map.", "The old name and the new name refer to the same place."),
        ("The blue car and the red car are parked outside.", "The car outside today and yesterday is the same car."),
    ],
}


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():
    print(f"  Loading model: {MODEL_NAME}")
    print(f"  Device: {DEVICE}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = model.to(DEVICE)
    model.eval()
    
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        n_layers = len(model.encoder.layer)
    elif hasattr(model, 'layers'):
        n_layers = len(model.layers)
    else:
        n_layers = getattr(model.config, 'num_hidden_layers', None)
    
    print(f"  Number of transformer layers: {n_layers}")
    print(f"  Hidden dimension: {model.config.hidden_size}")
    print()
    
    return model, tokenizer, n_layers


def get_all_layer_embeddings(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states
    attention_mask = inputs['attention_mask'].unsqueeze(-1).float()
    
    layer_embeddings = []
    for layer_hidden in hidden_states:
        masked = layer_hidden * attention_mask
        summed = masked.sum(dim=1)
        counts = attention_mask.sum(dim=1)
        mean_pooled = summed / counts
        layer_embeddings.append(mean_pooled.squeeze(0).cpu().numpy())
    
    return layer_embeddings


def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


# ============================================================
# EXPERIMENT
# ============================================================

def run_experiment():
    print("=" * 65)
    print("   SEMANTIC PRIMITIVES - LAYER-BY-LAYER BRAIN SCANNER")
    print(f"   Model: {MODEL_NAME}")
    print("=" * 65)
    print()

    model, tokenizer, n_layers = load_model()
    total_layers = n_layers + 1

    # -- Step 1: Collect embeddings at every layer --
    print("[1/3] Collecting embeddings at every layer...")
    
    primitive_vectors_by_layer = [{} for _ in range(total_layers)]
    
    for prim_name, pairs in PRIMITIVES.items():
        print(f"  {prim_name}...", end=" ", flush=True)
        
        for pair_idx, (without, with_prim) in enumerate(pairs):
            emb_without = get_all_layer_embeddings(without, model, tokenizer)
            emb_with = get_all_layer_embeddings(with_prim, model, tokenizer)
            
            for layer_idx in range(total_layers):
                diff = emb_with[layer_idx] - emb_without[layer_idx]
                
                if prim_name not in primitive_vectors_by_layer[layer_idx]:
                    primitive_vectors_by_layer[layer_idx][prim_name] = []
                primitive_vectors_by_layer[layer_idx][prim_name].append(diff)
        
        print(f"({len(pairs)} pairs)")
    
    # -- Step 2: Compute consistency at every layer --
    print("\n[2/3] Computing consistency at every layer...")
    
    prim_names = sorted(PRIMITIVES.keys())
    consistency_matrix = np.zeros((len(prim_names), total_layers))
    
    for layer_idx in range(total_layers):
        for p_idx, prim_name in enumerate(prim_names):
            vectors = primitive_vectors_by_layer[layer_idx].get(prim_name, [])
            if len(vectors) < 2:
                continue
            
            normed = []
            for v in vectors:
                n = np.linalg.norm(v)
                if n > 0:
                    normed.append(v / n)
            
            sims = []
            for i in range(len(normed)):
                for j in range(i + 1, len(normed)):
                    sims.append(cosine_similarity(normed[i], normed[j]))
            
            consistency_matrix[p_idx, layer_idx] = np.mean(sims) if sims else 0
    
    # -- Step 3: Compute orthogonality at every layer --
    print("[3/3] Computing orthogonality at every layer...")
    
    orthogonality_by_layer = np.zeros(total_layers)
    
    for layer_idx in range(total_layers):
        mean_vecs = {}
        for prim_name in prim_names:
            vectors = primitive_vectors_by_layer[layer_idx].get(prim_name, [])
            if vectors:
                mean_vec = np.mean(vectors, axis=0)
                norm = np.linalg.norm(mean_vec)
                if norm > 0:
                    mean_vecs[prim_name] = mean_vec / norm
        
        cross_sims = []
        names = sorted(mean_vecs.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                sim = cosine_similarity(mean_vecs[names[i]], mean_vecs[names[j]])
                cross_sims.append(abs(sim))
        
        orthogonality_by_layer[layer_idx] = np.mean(cross_sims) if cross_sims else 0

    # ============================================================
    # DISPLAY RESULTS
    # ============================================================
    
    layer_labels = ["Emb"] + [f"L{i+1}" for i in range(n_layers)]
    
    print("\n" + "=" * 65)
    print("   RESULTS: CONSISTENCY BY LAYER")
    print("=" * 65)
    
    print(f"\n  {'Primitive':<16}", end="")
    for label in layer_labels:
        print(f" {label:>5}", end="")
    print("   Peak")
    print("  " + "-" * (16 + 6 * total_layers + 8))
    
    for p_idx, prim_name in enumerate(prim_names):
        row = consistency_matrix[p_idx]
        peak_layer = np.argmax(row)
        peak_val = row[peak_layer]
        
        print(f"  {prim_name:<16}", end="")
        for layer_idx in range(total_layers):
            print(f" {row[layer_idx]:.2f}", end="")
        print(f"   {layer_labels[peak_layer]} ({peak_val:.2f})")
    
    print(f"\n  {'ORTHOGONALITY':<16}", end="")
    for layer_idx in range(total_layers):
        print(f" {orthogonality_by_layer[layer_idx]:.2f}", end="")
    best_ortho = np.argmin(orthogonality_by_layer)
    print(f"   {layer_labels[best_ortho]} ({orthogonality_by_layer[best_ortho]:.2f})")

    # ============================================================
    # GENERATE HEATMAP
    # ============================================================
    
    print("\n" + "=" * 65)
    print("   Generating brain scan heatmap...")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [4, 1]})
    
    ax1 = axes[0]
    im = ax1.imshow(consistency_matrix, aspect='auto', cmap='RdYlBu_r', 
                     vmin=-0.1, vmax=0.8)
    ax1.set_yticks(range(len(prim_names)))
    ax1.set_yticklabels(prim_names, fontsize=10)
    ax1.set_xticks(range(total_layers))
    ax1.set_xticklabels(layer_labels, fontsize=9)
    ax1.set_xlabel("Layer", fontsize=12)
    ax1.set_title(f"BRAIN SCAN: {MODEL_NAME} ({n_layers} layers)\n"
                   "(Warmer = stronger directional consistency)", fontsize=13, fontweight='bold')
    
    for i in range(len(prim_names)):
        for j in range(total_layers):
            val = consistency_matrix[i, j]
            color = 'white' if val > 0.4 else 'black'
            ax1.text(j, i, f'{val:.2f}', ha='center', va='center', 
                    fontsize=7, color=color)
    
    plt.colorbar(im, ax=ax1, label='Consistency Score')
    
    ax2 = axes[1]
    colors = ['#2ecc71' if v < 0.1 else '#f39c12' if v < 0.2 else '#e74c3c' 
              for v in orthogonality_by_layer]
    ax2.bar(range(total_layers), orthogonality_by_layer, color=colors)
    ax2.set_xticks(range(total_layers))
    ax2.set_xticklabels(layer_labels, fontsize=9)
    ax2.set_ylabel("Avg Cross-Similarity", fontsize=10)
    ax2.set_title("Orthogonality by Layer (Lower = More Independent Primitives)", fontsize=11)
    ax2.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Highly orthogonal')
    ax2.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Independence threshold')
    ax2.legend(fontsize=8)
    
    plt.tight_layout()
    
    heatmap_path = f"brain_scan_{FILE_TAG}.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    print(f"   Saved: {heatmap_path}")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    
    avg_consistency_by_layer = consistency_matrix.mean(axis=0)
    combined = avg_consistency_by_layer - orthogonality_by_layer
    best_layer = np.argmax(combined)
    
    print("\n" + "=" * 65)
    print("   SUMMARY")
    print("=" * 65)
    
    print(f"\n   Layers scanned: {total_layers} (embedding + {n_layers} transformer layers)")
    print(f"\n   Average consistency by layer:")
    for layer_idx in range(total_layers):
        bar_len = int(max(0, avg_consistency_by_layer[layer_idx]) * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        marker = " ◄── BEST" if layer_idx == best_layer else ""
        print(f"   {layer_labels[layer_idx]:>4} {bar} {avg_consistency_by_layer[layer_idx]:.4f}  "
              f"(ortho: {orthogonality_by_layer[layer_idx]:.4f}){marker}")
    
    print(f"\n   Best layer for primitive structure: {layer_labels[best_layer]}")
    print(f"   (Consistency: {avg_consistency_by_layer[best_layer]:.4f}, "
          f"Orthogonality: {orthogonality_by_layer[best_layer]:.4f})")
    
    print(f"\n   Where each primitive peaks:")
    for p_idx, prim_name in enumerate(prim_names):
        peak = np.argmax(consistency_matrix[p_idx])
        val = consistency_matrix[p_idx, peak]
        print(f"   {prim_name:<16} peaks at {layer_labels[peak]:>4} ({val:.4f})")
    
    print()
    
    # Save data
    data_path = f"brain_scan_{FILE_TAG}_data.npz"
    np.savez(data_path,
             consistency_matrix=consistency_matrix,
             orthogonality_by_layer=orthogonality_by_layer,
             primitive_names=np.array(prim_names),
             layer_labels=np.array(layer_labels))
    print(f"   Raw data saved to {data_path}")
    print(f"   Heatmap saved to {heatmap_path}")
    print()


if __name__ == "__main__":
    run_experiment()
