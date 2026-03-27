# The Skeleton in the Machine
## Why AI Model Convergence Is Inevitable, Not Mysterious

*By Sheila Machado, Independent AI Researcher*
*March 2026*

---

> **Note:** The empirical follow-up to this article, Geometric Primitives of Meaning, was published March 25, 2026. Results are referenced throughout where relevant. Full code and brain scan heatmaps: [github.com/sheilamachado/geometric-primitives-of-meaning](https://github.com/sheilamachado/geometric-primitives-of-meaning)

---

AI researchers have documented a striking pattern: different models, built with different architectures, trained on different data, consistently develop similar internal representations. MIT's Platonic Representation Hypothesis (Huh et al., 2024) proposed that models are converging toward a shared statistical model of reality. Recent work on unnatural languages (Duan et al., 2025) demonstrated that models can extract structured meaning even from scrambled, noisy input by filtering keywords and inferring organization.

These findings are treated as separate phenomena requiring explanation. But they point to a simpler underlying truth.

The field has been asking the wrong question. The question was never *why do models converge?* The question is: *what would it mean if they didn't?*

This article argues that convergence is not a puzzle to be solved but a logical necessity given three established facts:

1. Language models work through next-token prediction
2. Next-token prediction requires predictable patterns
3. Predictable patterns require consistent structure

If models diverged on their internal representations, it would mean language has no consistent semantic structure, which would make language unlearnable. Since models clearly do learn language, semantic geometry must exist. And if semantic geometry exists and remains consistent, convergence is the only possible outcome.

This is not a discovery about AI. It is a recognition that the surprise at convergence reveals an unexamined assumption: that semantic space is neutral and shapeless, allowing arbitrary organizational forms. It never was.

---

## The Inverse Question

Recent work (Duan et al., 2025) showed that LLMs process unnatural languages, scrambled text with injected noise, by extracting keywords, filtering irrelevant tokens, and inferring correct organization. Models trained exclusively on scrambled text can achieve comparable performance to those trained on natural language, demonstrating that models are structure-extractors.

This finding illuminates why convergence occurs, though the connection has not been explicitly made.

Consider what non-convergence would require: different models, performing the same task, would need to discover fundamentally different structural patterns in the same data. This would mean multiple valid, equally predictive semantic geometries exist simultaneously.

But the unnatural languages research demonstrates the opposite: when given noisy input containing hidden structure, models converge on extracting that structure. They do not impose arbitrary organization. They discover the organization that makes prediction possible.

The mechanism is extraction, not creation. Models do not invent semantic geometry. They find it.

Now consider: what would it mean if no consistent structure existed? If antonyms sometimes oppose and sometimes align, if causes sometimes precede effects and sometimes do not, if categories sometimes nest and sometimes do not, then prediction becomes impossible. Maximum ambiguity means maximum entropy means a flat loss landscape means nothing to learn.

The fact that language is learnable proves semantic structure exists. The fact that semantic structure exists means any sufficiently powerful optimization process will find the same structure. Convergence is not mysterious. Divergence would be impossible.

---

## Connecting the Dots

The field has documented several phenomena that appear distinct:

**Cross-model convergence** (Platonic Representation Hypothesis): Vision models and language models develop geometrically similar representations despite training on entirely different data types.

**Structure extraction from noise** (Unnatural Languages): Models can learn from scrambled input by identifying keywords and inferring organization, achieving performance comparable to natural language training.

**Next-token prediction success**: Language models achieve high accuracy in predicting subsequent tokens across diverse contexts.

These are not separate mysteries. They are the same phenomenon observed from different angles.

Models converge because they extract structure. Different models processing the same data type find the same patterns because those patterns exist in the data. Vision and language models align on concrete concepts because both reflect the same physical reality, the shared source of the structure.

Models can denoise scrambled input because the structure remains. Shuffling words and adding noise does not eliminate semantic geometry; it obscures it. Models trained to minimize prediction error learn to filter noise and extract the predictable patterns, the skeleton beneath the surface.

Next-token prediction works because semantic geometry creates predictability. The relationships that make *hot* oppose *cold*, *king* parallel *queen*, and *up* associate with *good* are the same relationships that make certain tokens more likely to follow others. Semantic geometry IS the predictable structure. They are not two things.

---

## What Is Semantic Geometry?

Semantic geometry refers to the spatial organization of meaning relationships in embedding space. I call the version found in human languages apple-geometry:

- Opposition as geometric inversion: hot and cold, up and down (antonyms as opposite vectors)
- Similarity as spatial proximity: car, automobile, vehicle cluster together
- Analogy as parallel structure: man:woman :: king:queen
- Hierarchy as nested containment: poodle inside dog inside animal
- Spatial metaphors for abstraction: up means more or good, down means less or bad
- Causality as directional relationship: causes precede effects

These are not arbitrary features. They are what make next-token prediction possible. Any model that fails to encode these relationships produces wrong predictions and suffers higher loss.

The Unnatural Languages research demonstrates that models extract these relationships even when obscured by noise. My argument: these relationships are what forces convergence. Not their surface form, but their geometric consistency.

---

## Empirical Confirmation: Geometric Primitives of Meaning

After writing this theory, I ran a direct empirical test. I brain-scanned five transformer models across two architectures, BERT-base, BERT-large, and Qwen2.5 at 1.5B, 7B, and 14B, using probing classifiers to detect nine semantic primitives as geometric directions in their internal representations:

**Causation, Agency, Containment, Sequence, Property, Quantity, Identity, Part/Whole, Conditionality**

The results confirmed three predictions this article makes:

The primitives are real geometric directions. Consistency scores confirmed they point in the same direction regardless of sentence content, with average cross-similarity as low as 0.039. They are nearly orthogonal, genuinely independent dimensions.

They emerge in a consistent developmental hierarchy across all five models: temporal first, then logical, spatial, causal, abstract, and compositional. Architecture did not alter the order.

The geometry is cleanest in the middle layers, exactly where reasoning circuits operate, consistent with the three-phase anatomy: encoding, reasoning, and decoding.

The skeleton is not a metaphor. It is measurable, nearly orthogonal geometric directions that every model finds in the same order.

Full results, code, and brain scan heatmaps: [github.com/sheilamachado/geometric-primitives-of-meaning](https://github.com/sheilamachado/geometric-primitives-of-meaning)

---

## Why Semantic Geometry Forces Convergence

Consider the degrees of freedom, the places where divergence could enter:

**Math is the mechanism.** Gradient descent follows the loss landscape deterministically toward whatever minimizes prediction error. Stochastic elements affect the path taken, not the destination. The destination is determined by which semantic geometry actually minimizes loss.

**Architecture is the tool.** Different architectures affect processing efficiency and which aspects of semantic geometry are accessible. Weak architectures capture shallow patterns. Powerful architectures map deeper structure. But architecture does not create semantic geometry. It reveals what exists in the data. The Geometric Primitives results confirm this: the same nine directions, the same developmental hierarchy, across BERT encoders and Qwen decoders.

**Semantic geometry is the medium.** And the medium is not shapeless. Apple-geometry has specific, consistent properties that appear in every sentence of training data. These properties create predictable patterns. They are not incidental features that models could organize differently. They are the structure that makes learning possible.

There is nowhere for meaningful divergence to enter. Convergence within a structure-class is the only possible outcome.

---

## But Could Other Structures Exist?

This raises a critical question: Is apple-geometry the only possible structure, or could other consistent geometries exist?

I introduce the concept of structure-classes: distinct semantic geometries that are internally consistent but geometrically incompatible with each other.

**Apple-geometry** (human languages): opposition as inversion, similarity as proximity, analogical parallelism, hierarchical nesting, spatial metaphors, directional causality.

**Orange-geometry** (hypothetical): A different but internally consistent semantic structure. For example, opposition as orthogonality rather than inversion, similarity through contrast rather than proximity, non-hierarchical categories, different spatial and temporal mappings.

The untested question: Can orange exist? If so, would models trained exclusively on orange converge on orange-geometry the way they converge on apple? This is what existing research has not tested.

---

## What Would Prove This Wrong

A theory that cannot be falsified is not a theory. Here is what would break this one.

**Test One: Can orange-geometry exist and be learned?**

Design a synthetic language with systematic geometric violations of apple. The grammar must be fully consistent so next-token prediction remains theoretically possible. Train multiple models exclusively on orange-language.

Outcome A strengthens the theory: Models fail to learn efficiently, or they covertly restructure internal representations toward apple-geometry despite surface compliance.

Outcome B refines the theory: Models learn orange successfully with stable, distinct geometry. Multiple models trained on orange converge within orange. This proves the theory is about consistency-forcing-convergence, not apple being privileged. Multiple structure-classes can exist.

**Test Two: Does semantic geometry originate in reality?**

Train models on non-linguistic data from physical reality (sensor readings, simulations, robotic interactions). No human language involved. Compare geometric representations to language models for relevant concepts.

Outcome A extends the theory: Geometric alignment occurs. Apple-geometry derives from physical reality's structure, not human linguistic convention. Language inherited its shape from the world.

Outcome B narrows the theory: No alignment. Apple is specific to human language and cognition. The skeleton is narrower than claimed.

Combined falsification: Both Outcome B results together would demonstrate that semantic geometry can take arbitrary consistent forms and reality does not determine it. The theory would be definitively falsified.

---

## Relationship to Existing Work

**Platonic Representation Hypothesis (Huh et al., 2024):** Their claim is that models converge toward a shared statistical model of reality. My claim is that models converge because consistent structure forces convergence. I provide the mechanism (structure extraction) for their observation (convergence) and add testable predictions about structure-classes.

**Unnatural Languages (Duan et al., 2025):** Their finding is that models extract structure from noise via keyword filtering and organization inference. My contribution is that this extraction mechanism explains why convergence occurs. Models do not impose arbitrary structure. They extract existing structure. What they did not test: whether non-apple structures can be extracted and converged upon.

**Geometric Primitives of Meaning (Machado, 2026):** Direct empirical confirmation. Nine semantic primitives confirmed as real, nearly orthogonal geometric directions across five models and two architectures. Consistent developmental hierarchy regardless of architecture. Full results at [github.com/sheilamachado/geometric-primitives-of-meaning](https://github.com/sheilamachado/geometric-primitives-of-meaning)

---

## Honest Limitations

1. The origin of apple's structure (embodied cognition vs. physical reality vs. logical necessity) remains an open question addressed by Test Two.
2. Cross-lingual convergence evidence exists primarily for languages with cultural contact. Universal convergence across truly isolated language families remains to be rigorously tested.
3. Whether all human languages belong to a single structure-class or contain meaningful subclasses is unresolved.
4. Cross-modal convergence (vision-language) suggests shared reality-grounded structure but needs systematic investigation through the structure-class framework.
5. Claims about scaling ceilings are logical consequences but need empirical validation.

These limitations narrow the scope without undermining the core argument: within structure-classes, convergence is inevitable because structure extraction from consistent patterns yields consistent results.

---

## Conclusion

The field documented convergence and treated it as profound mystery. But the pieces were always there:

- Models extract structure from data (Duan et al., 2025)
- Language has consistent semantic geometry (multiple sources)
- Next-token prediction requires predictable patterns (fundamental)

The connection: Models converge because they are all extracting the same structure. That structure exists because it is what makes prediction possible. Prediction is possible because semantic geometry is consistent.

The empirical confirmation arrived in March 2026. Brain scans of five transformer models confirmed nine semantic primitives as real, nearly orthogonal geometric directions with a consistent developmental hierarchy. The skeleton is not a metaphor. It is measurable. It shows up in every model, in the same order, in the same layers.

The surprise at convergence reveals the assumption: that semantic space is neutral and shapeless. It is not. Language has bones, the geometric relationships that create predictable patterns. Any sufficiently powerful optimization process examining consistent structure will find the same skeleton.

Convergence is not the mystery. It is the proof that structure exists.

The untested question remains: Are there other skeletons? Can orange exist alongside apple? That is what the next experiment will show.

---

*Follow the research: [@0606Machado on X](https://x.com/0606Machado)*
*Empirical follow-up: [github.com/sheilamachado/geometric-primitives-of-meaning](https://github.com/sheilamachado/geometric-primitives-of-meaning)*
