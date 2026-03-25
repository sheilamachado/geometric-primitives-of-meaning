"""
SEMANTIC PRIMITIVES PROBE
=========================
Testing whether meaning has discoverable geometric structure.
Uses Ollama API - final layer embeddings only.

Model: nomic-embed-text (via Ollama)

Usage:
    py semantic_primitives_probe.py
"""

import numpy as np
import requests
import json
import sys
from itertools import combinations

# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"

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
# EMBEDDING EXTRACTION
# ============================================================

def get_embedding(text, model=MODEL):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": model, "input": text},
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        if "embeddings" in data:
            return np.array(data["embeddings"][0])
        elif "embedding" in data:
            return np.array(data["embedding"])
        else:
            print(f"Unexpected response format: {list(data.keys())}")
            return None
    except requests.exceptions.ConnectionError:
        print("\n[ERROR] Cannot connect to Ollama!")
        print("Make sure Ollama is running: ollama serve")
        print(f"And the model is available: ollama pull {model}")
        sys.exit(1)
    except Exception as e:
        print(f"Error getting embedding for '{text[:50]}...': {e}")
        return None


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
    print("   SEMANTIC PRIMITIVES PROBE")
    print("   Testing geometric structure of meaning")
    print("=" * 65)
    print(f"\n   Model: {MODEL}")
    print(f"   Primitives: {len(PRIMITIVES)}")
    total_pairs = sum(len(v) for v in PRIMITIVES.values())
    print(f"   Total sentence pairs: {total_pairs}")
    print(f"   Total embeddings needed: {total_pairs * 2}")
    print()

    # -- Step 1: Collect all embeddings --
    print("[1/4] Collecting embeddings...")
    primitive_vectors = {}
    all_embeddings = {}

    for prim_name, pairs in PRIMITIVES.items():
        print(f"  {prim_name}...", end=" ", flush=True)
        diff_vectors = []

        for i, (without, with_prim) in enumerate(pairs):
            emb_without = get_embedding(without)
            emb_with = get_embedding(with_prim)

            if emb_without is None or emb_with is None:
                print(f"  [SKIP pair {i}]", end="")
                continue

            all_embeddings[(prim_name, i, "without")] = emb_without
            all_embeddings[(prim_name, i, "with")] = emb_with

            diff = emb_with - emb_without
            diff_vectors.append(diff)

        primitive_vectors[prim_name] = diff_vectors
        print(f"({len(diff_vectors)} pairs)")

    embedding_dim = len(next(iter(all_embeddings.values())))
    print(f"\n   Embedding dimensionality: {embedding_dim}")

    # -- Step 2: Test CONSISTENCY --
    print("\n" + "=" * 65)
    print("[2/4] CONSISTENCY TEST")
    print("  Do 'with - without' vectors point the same direction")
    print("  across different sentence content?")
    print("  (Higher = more consistent direction = more real)")
    print("-" * 65)

    consistency_scores = {}
    for prim_name, vectors in primitive_vectors.items():
        if len(vectors) < 2:
            print(f"  {prim_name}: not enough pairs")
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

        avg_sim = np.mean(sims)
        std_sim = np.std(sims)
        consistency_scores[prim_name] = avg_sim

        bar_len = int(max(0, avg_sim) * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"  {prim_name:<16} {bar} {avg_sim:.4f} (±{std_sim:.4f})")

    # -- Step 3: Test ORTHOGONALITY --
    print("\n" + "=" * 65)
    print("[3/4] ORTHOGONALITY TEST")
    print("  Are different primitives pointing in DIFFERENT directions?")
    print("  (Lower = more independent = more distinct primitives)")
    print("-" * 65)

    mean_primitive_vectors = {}
    for prim_name, vectors in primitive_vectors.items():
        if vectors:
            mean_vec = np.mean(vectors, axis=0)
            norm = np.linalg.norm(mean_vec)
            if norm > 0:
                mean_primitive_vectors[prim_name] = mean_vec / norm

    prim_names = sorted(mean_primitive_vectors.keys())
    
    print(f"\n  {'':>16}", end="")
    for name in prim_names:
        print(f"  {name[:6]:>6}", end="")
    print()
    
    cross_sims = []
    for i, name_i in enumerate(prim_names):
        print(f"  {name_i:<16}", end="")
        for j, name_j in enumerate(prim_names):
            sim = cosine_similarity(
                mean_primitive_vectors[name_i],
                mean_primitive_vectors[name_j]
            )
            if i != j:
                cross_sims.append(sim)
            marker = "  ●●●●" if i == j else f"  {sim:>5.2f}"
            print(marker, end="")
        print()

    avg_cross = np.mean(cross_sims) if cross_sims else 0
    print(f"\n  Average between-primitive similarity: {avg_cross:.4f}")
    print(f"  (Perfect orthogonality = 0.00, identical = 1.00)")

    # -- Step 4: TRANSFERABILITY TEST --
    print("\n" + "=" * 65)
    print("[4/4] TRANSFERABILITY TEST")
    print("  Can a primitive vector learned from N-1 pairs")
    print("  predict the direction in the held-out pair?")
    print("-" * 65)

    transfer_scores = {}
    for prim_name, vectors in primitive_vectors.items():
        if len(vectors) < 3:
            print(f"  {prim_name}: not enough pairs for leave-one-out")
            continue

        loo_sims = []
        for hold_out_idx in range(len(vectors)):
            train_vecs = [v for k, v in enumerate(vectors) if k != hold_out_idx]
            mean_train = np.mean(train_vecs, axis=0)
            norm = np.linalg.norm(mean_train)
            if norm > 0:
                mean_train = mean_train / norm

            test_vec = vectors[hold_out_idx]
            test_norm = np.linalg.norm(test_vec)
            if test_norm > 0:
                test_vec = test_vec / test_norm
            sim = cosine_similarity(mean_train, test_vec)
            loo_sims.append(sim)

        avg_transfer = np.mean(loo_sims)
        transfer_scores[prim_name] = avg_transfer

        bar_len = int(max(0, avg_transfer) * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"  {prim_name:<16} {bar} {avg_transfer:.4f}")

    # -- SUMMARY --
    print("\n" + "=" * 65)
    print("   SUMMARY")
    print("=" * 65)

    avg_consistency = np.mean(list(consistency_scores.values())) if consistency_scores else 0
    avg_transfer = np.mean(list(transfer_scores.values())) if transfer_scores else 0

    print(f"""
   Average consistency:    {avg_consistency:.4f}
   Average orthogonality:  {avg_cross:.4f}  (lower = better)
   Average transferability: {avg_transfer:.4f}

   INTERPRETATION:
   ─────────────────────────────────────────────
   Consistency > 0.3:  Primitives are REAL directions
   Consistency > 0.5:  Primitives are STRONG directions
   Consistency > 0.7:  Primitives are DOMINANT directions

   Orthogonality < 0.3: Primitives are mostly INDEPENDENT
   Orthogonality < 0.1: Primitives are highly ORTHOGONAL

   Transferability > 0.3: Directions GENERALIZE to new content
   Transferability > 0.5: Directions are ROBUST and universal
   ─────────────────────────────────────────────
""")

    if avg_consistency > 0.3 and avg_cross < 0.3:
        print("   >>> The primitives appear to be REAL, INDEPENDENT")
        print("       directions in the model's semantic space.")
        print("       This supports the geometric structure hypothesis.")
    elif avg_consistency > 0.3:
        print("   >>> The primitives are real directions but NOT fully")
        print("       independent — they may share some structure.")
    elif avg_consistency > 0.1:
        print("   >>> Weak signal. Primitives show SOME directionality")
        print("       but the final-layer embeddings may be too compressed.")
        print("       This needs intermediate-layer probing to confirm.")
    else:
        print("   >>> No clear directional structure detected at this layer.")
        print("       This does NOT disprove the hypothesis — we're only")
        print("       seeing the final layer. The structure may exist in")
        print("       intermediate layers (Ng's reasoning circuits).")

    # Save with model-specific filenames
    tag = "nomic-embed-text"
    
    print(f"\n   Saving raw data to primitive_vectors_{tag}.npz...")
    save_data = {}
    for prim_name, vectors in primitive_vectors.items():
        if vectors:
            save_data[prim_name] = np.array(vectors)
    np.savez(f"primitive_vectors_{tag}.npz", **save_data)
    
    if mean_primitive_vectors:
        mean_data = {k: v for k, v in mean_primitive_vectors.items()}
        np.savez(f"mean_primitive_vectors_{tag}.npz", **mean_data)
    
    print(f"   Saved: primitive_vectors_{tag}.npz")
    print(f"   Saved: mean_primitive_vectors_{tag}.npz")
    print()
    
    return consistency_scores, avg_cross, transfer_scores


if __name__ == "__main__":
    run_experiment()
