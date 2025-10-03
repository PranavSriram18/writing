# Title (TODO)

## 1. Introduction

How do Transformers - the models that underpin modern LLMs - actually work under the hood? And how can we make them faster? These are central questions in modern AI research, particularly in the subfields of mechanistic interpretability, attention variant design, and sparsity. The goal of this article is to bridge the gulf between introductory articles on transformer architecture and the rapidly growing body of frontier developments on these subjects.

In particular, our (perhaps ambitious) thesis is: despite the diversity and apparent complexity of ideas in this space, **a small number of mental models and metaphors can equip a reader comfortable with the fundamentals to understand the research frontier**.

To this end, we hope to explore the following ideas in this and future articles. (Don't worry if many of these terms don't make sense yet!)

* Transformer models as *defining information flow through a graph*
* Residual streams as *fixed-bandwidth information highways*
* "Residual actors" as *collaborating actors with immediate and long-term goals*
* Vanilla Attention as a particular implementation of an *abstract interface for cross-stream causal communication*
* Attention Heads as *low-rank, additive updates* that write into subspaces of the residual stream
* A number of attention variants as **connectivity-preserving static or dynamic sparsification** of the underlying information-flow graph
* Kernelized attention as defining a factor graph mediating cross-stream communication


# 2) The Transformer as a Grid of Information Flow

Our core frame for this article will be to think of transformers in terms of information flowing through a grid. The two axes of this grid are time (tokens) and depth (layers). Each point `(t, l)` on the grid is the state of token `t` after layer `l`, which we will denote `x_{t,l}`.

![Transformer Grid](transformer-grid-figure)

**Rows as Layers**

A horizontal row in our grid corresponds to a transformer layer. Layers are the computational units we typically think about in deep learning: a model is a composition of several layers. Each transformer layer is comprised of three core operations:

1. Attention - the core focus of this article.
2. MLP - a shallow feedforward neural network.
3. Normalizers and regularizers, like dropout, RMSNorm, and others, which we will collectively refer to as "nonlinearities." While important, we will omit these from all equations and descriptions in this article to keep things uncluttered, as they don't change our core explanations.

**Columns as Residual Streams**

A vertical column corresponds to a single token being processed across layers. We call the `t`th column the **residual stream** for token `t`, a term popularized in [Anthropic's original Transformer Circuits paper](https://transformer-circuits.pub/2021/framework/index.html). A key frame we will adopt in this article is a shift from thinking about transformers as stacks of rows (layers), and instead as a series of parallel columns (residual streams).

**The Journey of a Single Token**

Given a sequence of input tokens `w_1, ..., w_T`, focus on how a single token `w_t` flows through its residual stream. 

1. The token enters its stream as the sum of its word **embedding vector** `e_t` and **positional embedding** `p_t`. 

2. Each layer computes an update that is **added** to the stream - hence the term *residual*. The token's representation evolves through a sequence of intermediate states: `x_{t,0}, x_{t,1}, …, x_{t,L}`.

3. At the final layer, the representation is multiplied by the unembedding matrix to produce logits over the vocabulary, which are then normalized into a probability distribution for the next token.

We can thus think of the residual stream as an **information highway**, mapping the current token through a progressive sequence of representations that culminate in a sufficient statistic for the distribution of the next token. Importantly, this highway has a **fixed bandwidth,** dictated by the dimensionality D of the residual stream state. 

**Residual Actors and Attention as an Interface**

Let's now unpack what happens inside a layer. We can think of the two core operations, namely Attention and the MLP, as representing *communication* with other streams, and  *computation* on an individual stream. 

```
# Attention: collaboration step - pull from previous streams at the same layer
x_{t,l+1} = x_{t,l} + Attend(x_{1,l}, x_{2,l}, ..., x_{t,l})

# MLP: solo step - compute locally on the updated state
x_{t,l+1} = x_{t,l+1} + fMLP(x_{t,l+1})
```

* The **collaboration step** uses attention to gather information from earlier streams `(u, l)` with `u ≤ t`. This is the actor listening to its peers, constrained by causality to only look backward.
* The **solo step** uses an MLP to compute purely on its own state, refining or transforming what it already has.

TODO (@me) - explain residual actors, collaboration, attention interface, goals. 

**The collaboration metaphor.** Imagine `T` residual actors working side by side. Each one is handed its token embedding at the bottom of the grid. Its job is to transform this vector, over `L` steps, into a state that can predict the next token. But actors are not solitary. Along the way they also leave behind signals that future actors can read. In this way, the actors collaborate: each has an individual goal but also helps others.

**The grid as a graph**

With this picture in mind, we can view the transformer as a graph:

(TODO - insert image here)

* Vertical edges `(t, l) → (t, l+1)` represent the evolution of a token's representation via additive updates between layers.

* Horizontal edges `(u, l) → (t, l)` represent information flow from earlier to later streams.

**Fixed bandwidth.** Each `x_{t,l}` is a D-dimensional vector. All updates write back into this same space. Capacity is finite, which forces the network to manage what information persists and what is overwritten over the course of a residual stream.

---

# 3) Anatomy of Causal Attention: QK and OV Circuits

To motivate attention from first principles, let's put ourselves in the shoes of a single residual actor at `(t, l)`. Our job is to enrich our own state with information from previous streams. We can break this task down into asking two fundamental questions:

**Where should we look?** Among all positions `u ≤ t`, which earlier actors are relevant to me?

**What information should I grab?** From each chosen source, what information should I import?

These roles are implemented by queries, keys, and values:

* **Key (k_u):** each earlier actor `(u, l)` emits a key vector that encodes *what kind of information it has*.
* **Query (q_t):** our actor emits a query vector that encodes *what kind of information it wants*.
* **Value (v_u):** each earlier actor also emits a value vector containing the *payload* it can provide.

```
# scores via dot products
score_u = dot(q_t, k_u)

# normalize into weights
a_{t,u} = softmax(score_u)

# weighted average of values
z_t = Σ_{u≤t} a_{t,u} * v_u

# projection and update
x_{t,l+1} = x_{t,l} + W_O · z_t
```

Key takeaways:

* **Separation of concerns.** Queries and keys decide *where* to look; values carry *what* to import; `W_O` projects the imported signal back into the residual space.

* **Additive integration.** The imported content is added to the residual state; nothing is overwritten outright.

In interpretability terms, this separation is often described as **QK circuits** (deciding *where* to look) and **OV circuits** (deciding *what* to write). This decoupling of routing from content is part of what makes attention powerful and analyzable.

**Computational Complexity**

Now we can see why causal attention is expensive. Consider generating a sequence of `T` tokens. The actor at position `t` must compute attention over all `u ≤ t` positions. Each position `t` involves:
- Computing query and key projections: `O(D²)` (matrix-vector multiplication with `D × D` weight matrices)
- Computing `t` dot products between query and keys: `O(tD)`  
- Weighted sum of `t` value vectors: `O(tD)`

So position `t` does `O(D² + tD)` work. Summing across all positions:

```
Total work = Σ_{t=1}^T O(D² + tD) 
           = O(TD²) + O(D · Σ_{t=1}^T t) 
           = O(TD²) + O(T²D)
           = O(T²D + TD²)
           = O(T²D)  [when T > D, which holds in modern applications]
```

Intuitively, this quadratic scaling makes sense: each residual actor does work proportional to its position in the sequence. Early actors do little; later actors do much more. The average workload grows linearly with sequence length, and we have `T` actors, yielding `O(T²D)` total complexity.

As a first approximation, this `O(T²D)` complexity is the central bottleneck in scaling transformers to long contexts, though as we'll see shortly, there is some nuance to this picture. Much of the attention variant literature aims to attack this `O(T²D)` term.

**Nuances on Complexity**

Interestingly, in a [talk](https://www.youtube.com/watch?v=rBCqOTEfxvg&t=1080s) shortly after the original Transformer paper, Łukasz Kaiser mentioned being nervous about the cost being quadratic in context length, before Noam Shazeer pointed out that D was significantly larger than T, so the `O(T²D)` term wasn't the bottleneck. Their application was language translation of sentences, so T was just ~70 in their context! It's striking to hear becuase in under a decade we've gone from translating sentences to thinking about how to push models to reason about corpuses of over a million tokens!

An important detail to keep in mind when discussing the complexity of attention is that
attention is highly parallel, so actual wall-clock time differs significantly from raw FLOP counts. An interesting frame for thinking about complexity in a world of increasing compute is: what is the complexity of an algorithm in the limit of infinite parallel compute? For a fascinating deep dive on this, see ["Attention is Logarithmic (Actually)"](https://supaiku.com/attention-is-logarithmic).

---

# 4) Attention Heads: Work-Partitioning and Low-Rank Updates

The standard framing of multi-head attention is about **work-partitioning**: keys, queries, and values are sliced along the embedding dimension, heads perform attention independently on their slices, the results are concatenated and then projected using W_O before being added to the residual stream. A key thing to notice is that concatenating the partial results then multiplying by W_O is equivalent to multiplying the partial results by shards of W_O and adding the results. 

A cleaner pseudocode sketch:

```
for each head h in {1..H}:
    q_t^h = x_{t,l} · W_Q^h        # head-specific query
    k_u^h = x_{u,l} · W_K^h        # head-specific key (for u ≤ t)
    v_u^h = x_{u,l} · W_V^h        # head-specific value
    scores^h = [ dot(q_t^h, k_u^h) for u ≤ t ]
    weights^h = softmax(scores^h)
    z_t^h = Σ_{u≤t} weights^h_u * v_u^h

# combine head outputs
z_t = concat(z_t^1, …, z_t^H)
x_{t,l+1} = x_{t,l} + W_O · z_t
```

This is the usual pipeline: partition the work across heads, then concatenate and project.

There is a second, equally correct way to read the last line that is often more illuminating - the **low-rank additive update** view. Concatenation followed by projection is equivalent to a sum of head-specific writes:

```
# equivalent form that exposes additivity by head
x_{t,l} ← x_{t,l} + Σ_h (W_O^h · z_t^h)
```

Each head writes into the residual through its own projection slice `W_O^h`.

The low-rank additive framing plays a significant role in mechanistic interpretability work. A few consequences:

* **Low-rank, subspace-targeted writes.** A head can only modify the residual within the column space of `W_O^h` - at most rank `d_h`. Heads are low-rank writers into (possibly different) subspaces of the highway.
* **Limited interaction between heads.** If two heads write largely into disjoint or orthogonal subspaces, later computation may treat their contributions as independent. Overlap enables interaction or interference. The geometry of `W_O` partitions bandwidth.
* **Implicit memory management.** Updates are additive and persistent. Information written by a head sticks around unless future layers actively overwrite or counter-write it. Since bandwidth is finite (dimension D), writing one thing necessarily crowds others. Some heads compress or move information, others cache patterns for downstream use, and some act as cleaners.

This dual framing - work-partitioning on the surface, low-rank additive updates under the hood - will matter when we adopt the graph view next. It lets us see heads not only as parallel searchers, but as specialized writers competing for limited highway capacity.

---

# 5) The Combinatorics of Attention-Based Information Flows

With the grid-as-graph picture established, we can now ask a deeper question: how can information travel from one position `(t1, l1)` to another `(t2, l2)`? The answer reveals surprising combinatorial richness.

Information moves through the graph by alternating between two types of edges:
* **Horizontal moves** (attention): `(τ, l) → (t, l)` where `τ < t`
* **Vertical moves** (residual): `(t, l) → (t, l+1)`

Even simple cases reveal exponential growth in path count. From `(i, L)` to `(j, L+1)` there are only two paths: Right then Up, or Up then Right. But from `(i, L)` to `(j, L+2)`, the number of possible paths grows quickly, including many interleavings of horizontal and vertical moves. In general, the number of distinct paths grows exponentially with both layer depth and token distance.

This graph framing emphasizes a key insight: transformers do not move information along a single fixed route. Instead, there is an explosion of possible pathways by which signals can travel and mix across the network. This redundancy suggests opportunity for optimization—not every edge may be necessary for effective communication.

---

# 6) Static Graph Sparsification

We have seen that there are exponentially many possible information pathways between positions `(t1, l1)` and `(t2, l2)`. This suggests redundancy: not every edge is necessary for effective communication. Can we **sparsify** the graph while preserving the essential routes of information flow?

This is the intuition behind many efficient attention variants. By restricting which edges exist in the graph, we can reduce computational cost while maintaining useful long-range connectivity.

**The Neighborhood Framework**

Let's introduce notation to make this concrete. Define `N(t, l)` as the **attention neighborhood** of position `(t, l)`: that is, the set of grid positions that token `t` can attend to. The actor at `(t, l)` computes attention only over positions in `N(t, l)`, ignoring all others.

Each sparsification pattern simply provides a different definition of `N(t, l)`.

**Receptive Field**

A critical concept for understanding these patterns is the **receptive field**: the set of input tokens that can influence a given position. Formally, the receptive field of position `(T, l)` is the set of token indices `i` such that there exists a path in the information flow graph from `(i, 0)` to `(T, l)`. Equivalently, it's the number of initial token positions that actor `(T, l)` can "see" through the network.

The receptive field determines how much context a position has access to, while the neighborhood `N(t, l)` determines the computational cost per layer. The art of efficient attention is maximizing receptive field while minimizing neighborhood size.

Let's examine the key variants and their trade-offs:

### Vanilla (Full) Attention

`N(t, l) = {(1, l), (2, l), ..., (t, l)}`

Every position attends to all earlier positions at the same layer (plus itself). This is the complete causal attention pattern—maximum connectivity, maximum cost.

**Receptive field:** `T` at all layers (perfect—every position can see the entire sequence)  
**Complexity:** `O(T² D)` as derived in Section 3

Vanilla attention provides the best possible receptive field but pays for it with quadratic scaling. Every token is directly accessible from every later position at each layer.

### Sliding Window Attention

`N(t, l) = {(max(1, t-w+1), l), ..., (t, l)}`

Each position attends only to its `w` most recent neighbors. The receptive field is bounded locally but expands linearly with depth: a token at layer `L` can indirectly access information from roughly `L × w` positions back.

**Receptive field:** `O(w · l)` at layer `l` (grows linearly with depth)  
**Complexity:** `O(T w D)` (linear in sequence length!)

This is a dramatic improvement: we've broken the quadratic barrier. However, we pay a price in connectivity. To establish a path from position `1` to position `T`, we need approximately `T/w` layers. Put differently, information propagates at a rate of `w` tokens per layer. For long sequences, this can require very deep networks or risk losing access to distant context.

The sliding window represents a fundamental trade-off: computational efficiency for limited receptive field growth.

### Dilated Attention

`N(t, l) = {(t, l), (t-d, l), (t-2d, l), (t-3d, l), ...}` (for positions ≥ 1)

Positions are sampled at regular intervals `d` (the dilation factor). Different layers can use different dilations. With carefully chosen dilation schedules across layers, the receptive field expands exponentially while keeping edge count linear per layer.

**Receptive field:** `O(d^l)` with exponentially increasing dilations across layers  
**Complexity:** `O(T D)` if we fix the neighborhood size (e.g., attend to `k` positions at each layer)

This is the key insight: by using dilation `d = 2` at layer 1, `d = 4` at layer 2, `d = 8` at layer 3, etc., we can reach position `1` from position `T` in only `O(log T)` layers. Each layer still does `O(TD)` work (constant-sized neighborhoods), giving us the best of both worlds: logarithmic depth to full receptive field with only linear cost per layer.

Dilated attention shows that we can have efficient computation *and* rapid receptive field growth—a dramatic improvement over sliding windows.

### Logarithmic Attention

`N(t, l) = {(t, l), (t-1, l), (t-2, l), (t-4, l), (t-8, l), ...}`

Or more generally `N(t, l) = {(t-k^p, l) : k ≥ 0}` for some `p > 0`. This ensures coverage at both short and long ranges with sublinear edge count. A token can attend to recent neighbors (granular, local information) and exponentially spaced distant neighbors (coarse, global information).

**Receptive field:** `T` even at a single layer (can reach all positions directly)  
**Complexity:** `O(T log(T) D)` since each position attends to `O(log t)` earlier positions

Logarithmic attention achieves full receptive field immediately while breaking quadratic scaling. The neighborhood size grows only logarithmically with position, giving us `O(T log T)` complexity—a middle ground between vanilla `O(T²)` and linear patterns like sliding windows. The trade-off is that distant tokens are accessible but only at exponentially coarser granularity.

### Random or Stochastic Masking

`N(t, l)` is a random subset of size `r` drawn from `{(1, l), ..., (t, l)}`

Which edges are present varies per sequence or batch. While any individual edge is unreliable, connectivity is preserved in expectation across the randomness. This trades deterministic paths for probabilistic coverage.

**Receptive field:** `T` in expectation (any position can be sampled)  
**Complexity:** `O(T r D)` where `r` is the fixed sample size

Random attention provides probabilistic full coverage at linear cost (if `r` is constant). The key insight is that while no single path is guaranteed, the aggregate effect across random samples approximates dense connectivity. This works surprisingly well in practice when `r` is chosen appropriately.

### Global Tokens

`N(t, l) = {(g_1, l), (g_2, l), ..., (g_k, l)} ∪ LocalPattern(t, l)`

Every position attends to a small set of designated global token positions `g_1, ..., g_k` (often the first few tokens) in addition to some local pattern. These global tokens act as information hubs, providing shortcut paths for long-range communication. Any two positions can exchange information with at most 2 hops through a global node.

**Receptive field:** `T` (all positions connect through global hubs)  
**Complexity:** `O(T D)` if `k` and the local pattern size are constant

Global tokens create a hub-and-spoke topology: instead of every position connecting to every other, they all connect to a few central hubs. This achieves full receptive field with linear complexity by accepting a 2-hop path between arbitrary positions. The global tokens become information bottlenecks that aggregate and broadcast context.

### Sink Tokens

`N(t, l) = {(s_1, l), (s_2, l), ..., (s_m, l)} ∪ LocalPattern(t, l)`

Similar to global tokens, but sink tokens `s_1, ..., s_m` are special positions designed to absorb attention mass when other edges are pruned. This provides a fallback destination that prevents attention weights from concentrating inappropriately on irrelevant positions when the desired target is outside `N(t, l)`.

**Receptive field:** Depends on `LocalPattern`, but sinks don't expand it  
**Complexity:** `O(T D)` if `m` and the local pattern size are constant

Sink tokens address a practical problem: when using sparse attention, the softmax must still normalize over *some* set of positions. If all relevant positions are pruned from `N(t, l)`, attention mass has nowhere meaningful to go and may concentrate on irrelevant nearby tokens. Sinks provide a designated "nowhere" that can safely absorb this mass without corrupting the information flow. They're less about expanding receptive field and more about maintaining attention pattern stability in sparse regimes.

**Key Insight**

All these patterns are **static**: `N(t, l)` is determined ahead of time based only on position indices, not on the actual content of the sequence. They represent fixed blueprints for trimming the graph without fully collapsing its connectivity. The art lies in choosing `N(t, l)` to preserve the critical information pathways while maximizing computational savings.