# Title (TODO)

## 1. Introduction

How do Transformers - the models that underpin modern LLMs - actually work under the hood? And how
can we make them faster? These are central questions in modern AI research, particularly in the
subfields of mechanistic interpretability, attention variant design, and sparsity. The goal of this
article is to bridge the gulf between introductory articles on transformer architecture and the
rapidly growing body of frontier developments on these subjects.

In particular, our (perhaps ambitious) thesis is: despite the diversity and apparent complexity of
ideas in this space, <span style="color: #007bff; font-weight: bold;">a small number of mental models and metaphors can equip a reader
comfortable with the fundamentals to understand the research frontier</span>.

To this end, we hope to explore the following ideas in this and future articles. (Don't worry if many
of these terms don't make sense yet!)

* Transformer models as <span style="color: #2ecc71; font-style: italic;">defining information flow through a graph</span>
* We'll introduce the notion of *Residual streams*, and how they can be thought  of as <span style="color: #2ecc71; font-style: italic;">fixed-bandwidth information highways</span>
* "Residual actors" as <span style="color: #2ecc71; font-style: italic;">collaborating actors with immediate and long-term goals</span>
* Vanilla Attention as a particular implementation of an <span style="color: #2ecc71; font-style: italic;">abstract interface for cross-stream
  causal communication</span>
* Attention Heads as <span style="color: #2ecc71; font-style: italic;">low-rank, additive updates</span> that write into subspaces of the
  residual stream
* A number of attention variants as <span style="color: #007bff; font-weight: bold;">connectivity-preserving static or dynamic
  sparsification</span> of the underlying information-flow graph
* Kernelized attention as defining a factor graph mediating cross-stream communication

---

## 2. Prerequisites and Notes on Style

**Prerequisites.** 
This article assumes you're comfortable with the basics of how Transformers work, particularly
causal self-attention. If you need a refresher, we recommend [Andrej Karpathy's video](PLACEHOLDER) and [3Blue1Brown's video](PLACEHOLDER) as excellent starting points.

**Scope.** 
We focus exclusively on causal, decoder-only transformers (like GPT-style models). Throughout this article, we use "Vanilla Attention" to refer to ordinary causal attention, the standard
self-attention mechanism used in these models.

**Inspiration.** 
This article is heavily inspired by [Anthropic's Mathematical Framework for
Transformer Circuits](PLACEHOLDER). One of the goals of this article is to provide a gentler onramp
to some of the deep insights expounded in that work.

**Philosophy.** 
Our emphasis is on building intuition rather than mathematical rigor or implementation details. To this end:
- We omit architectural and implementation details (like normalizers, regularizers, numerical
issues, etc.) that don't change the core story
- We liberally anthropomorphize (talking about actors "wanting" information, etc.)
- We often depict parallel computations as serial when it aids understanding.

**Notation**
Our working model will be a causal, decoder-only transformer with `L` layers, hidden dimension
`D`, and context length `T`. We'll denote input tokens by \(w_1, \ldots, w_T\), and use \(x_{t,l}\) to denote the representation of token \(t\) at layer \(l\). We use 1-indexing for both layers and tokens; \(x_{t,0}\) denotes the representation of the \(t\)th token before any transformer block but after token embedding and positional encoding.

---

# 3. The Transformer as a Grid of Information Flow

Our core frame for this article will be to think of transformers in terms of information flowing
through a grid. The two axes of this grid are time (tokens) and depth (layers). Each node `(t, l)`
on the grid represents the state of token `t` after layer `l`, which we denote \(x_{t,l}\).  

![Transformer Grid](transformer-grid-figure)

<span style="color: #007bff; font-weight: bold;">Rows as Layers</span>

A horizontal row in our grid corresponds to a transformer layer. Layers are the computational units
we typically think about in deep learning: a model is a composition of several layers. Each
transformer layer is comprised of three core operations:

1. Attention - the core focus of this article.
2. MLP - a shallow feedforward neural network.
3. Normalizers and regularizers, like LayerNorm, dropout, and others, which we will collectively refer
   to as "nonlinearities." While important, we will omit these from all equations and descriptions in
   this article to keep things uncluttered, as they don't change our core explanations.

<span style="color: #007bff; font-weight: bold;">Columns as Residual Streams</span>

A vertical column corresponds to a single token being processed across layers. We call the `t`th
column the <span style="color: #007bff; font-weight: bold;">residual stream</span> for token `t`, a term popularized in [Anthropic's original
Transformer Circuits paper](https://transformer-circuits.pub/2021/framework/index.html). A key frame
we will adopt in this article is a shift from thinking about transformers as stacks of rows (layers),
and instead as a series of parallel columns (residual streams).

<span style="color: #007bff; font-weight: bold;">The Journey of a Single Token</span>

Given a sequence of input tokens \(w_1, \ldots, w_T\), focus on how a single token \(w_t\) flows through its
residual stream. 

1. The token enters its stream as the sum of its word <span style="color: #007bff; font-weight: bold;">embedding vector</span> \(e_t\) and
   <span style="color: #007bff; font-weight: bold;">positional embedding</span> \(p_t\). 

2. Each layer computes an update that is <span style="color: #007bff; font-weight: bold;">added</span> to the stream - hence the term
   <span style="color: #2ecc71; font-style: italic;">residual</span>. The token's representation evolves through a sequence of intermediate states:
   \(x_{t,0}, x_{t,1}, \ldots, x_{t,L}\).

3. At the final layer, the representation is multiplied by the unembedding matrix to produce logits
   over the vocabulary, which are then normalized into a probability distribution for the next token.

We can thus think of the residual stream as an <span style="color: #007bff; font-weight: bold;">information highway</span>, mapping the current
token through a progressive sequence of representations that culminate in a sufficient statistic for
 the distribution of the next token. Importantly, this highway has a <span style="color: #007bff; font-weight: bold;">fixed bandwidth</span>, dictated by
 the dimensionality `D` of the residual stream state. 

<span style="color: #007bff; font-weight: bold;">Residual Actors and Attention as an Interface</span>

Let's now unpack what happens inside a layer. We can think of the two core operations, namely
Attention and the MLP, as representing <span style="color: #2ecc71; font-style: italic;">communication</span> with other streams, and
<span style="color: #2ecc71; font-style: italic;">computation</span> on an individual stream. 

```
# Attention: collaboration step - pull from previous streams at the same layer
z_{t,l} = x_{t,l} + Attend(x_{1,l}, x_{2,l}, ..., x_{t,l})

# MLP: solo step - compute locally on the post-attention state
x_{t,l+1} = z_{t,l} + MLP(z_{t,l})
```

* The <span style="color: #007bff; font-weight: bold;">collaboration step</span> uses attention to gather information from earlier streams
  \((u, l)\) with \(u \le t\). 

* The <span style="color: #007bff; font-weight: bold;">solo step</span> uses an MLP to compute purely on its own state, refining or
  transforming what it already has.

TODO (@me) - explain residual actors, collaboration, attention interface, goals. 

<span style="color: #007bff; font-weight: bold;">The collaboration metaphor.</span> Imagine `T` residual actors working side by side. Each one is
handed its token embedding at the bottom of the grid. Its job is to transform this vector, over `L`
steps, into a state that can predict the next token. But actors are not solitary. Along the way they
also leave behind signals that future actors can read. In this way, the actors collaborate: each has
an individual goal but also helps others.

<span style="color: #007bff; font-weight: bold;">The grid as a graph</span>

With this picture in mind, we can make concrete our framing of transformers as a graph.

(TODO - insert image here)

* Vertical edges \((t, l) \to (t, l+1)\) represent the evolution of a token's representation via additive
  updates between layers.

* Horizontal edges \((u, l) \to (t, l)\) represent information flow from earlier to later streams.


---

# 4. Anatomy of Causal Attention: QK and OV Circuits
We'll now recap how ordinary Attention works, albeit with an emphasis on (a) motivating it from first
principles, and (b) highlighting some aspects particularly salient to the frames we're developing. 

### 4.1 Attention Recap
Let's put ourselves in the shoes of a single residual actor at `(t, l)`. Our job in the attention step is to enrich our own
state with information from previous streams. We can break this task down into asking two fundamental
questions:

<span style="color: #2ecc71; font-style: italic;">Where should we look?</span> Among all nodes `u ≤ t`, which ones are relevant
to me?

<span style="color: #2ecc71; font-style: italic;">What information should I grab?</span> From each chosen source, what information should I
import?

These roles are implemented by queries, keys, and values:

* <span style="color: #007bff; font-weight: bold;">Key (\(k_u\))</span>: each earlier actor \((u, l)\) emits a key vector \(k_u\) that broadcasts
  <span style="color: #2ecc71; font-style: italic;">"this is the kind of information I have"</span>.

* <span style="color: #007bff; font-weight: bold;">Query (\(q_t\))</span>: we emit a query vector \(q_t\) that encodes <span style="color: #2ecc71; font-style: italic;">what kind of information we
  want</span>.

* <span style="color: #007bff; font-weight: bold;">Value (\(v_u\))</span>: each earlier actor also emits a value vector containing the actual
  <span style="color: #2ecc71; font-style: italic;">information payload</span> it provides if we select it.

* We use our query to score the relevance of each of the \(t\) keys \(k_1, k_2, \ldots, k_t\), and construct a
  <span style="color: #2ecc71; font-style: italic;">weighted average</span> of the associated values.

In pseudocode:

```
# scores each key by taking a dot product with our query
for u in {1, ..., t}:
    score_{t, u} = dot(q_t, k_u)

# normalize the scores to sum to 1 via softmax
for u in {1, ..., t}:
    a_{t,u} = exp(score_{t, u}) / Σ_{j≤t} exp(score_{t, j})

# create a weighted average of values based on attention scores
h_t = Σ_{u≤t} a_{t,u} * v_u

# multiply by another matrix W_O before adding to the residual stream
z_{t,l} = x_{t,l} + W_O · h_t
```

Note that while this pseudocode is meant to elucidate what's going on mathematically, it is NOT how
Attention is actually implemented in practice, primarily due to the fact that computations we're
depicting as serial here are actually carried out in parallel.

Key takeaways:

* <span style="color: #007bff; font-weight: bold;">Separation of concerns.</span> Queries and keys decide <span style="color: #2ecc71; font-style: italic;">where to read</span>; values and W_O
  determine <span style="color: #2ecc71; font-style: italic;">what to write</span>. In interpretability terms, this separation is described as
  <span style="color: #007bff; font-weight: bold;">QK and OV circuits</span>. 

* <span style="color: #007bff; font-weight: bold;">Linearity Modulo Attention Pattern.</span> The only source of nonlinearity comes from the softmax
  operation, which is part of the QK circuit (determining the attention pattern). If we fix the
  attention pattern, the entire attention operation becomes a linear function of its inputs.

* <span style="color: #007bff; font-weight: bold;">Additive integration.</span> The imported content is added to the residual state; nothing is
  overwritten outright.

### <span style="color: #007bff; font-weight: bold;">4.2 Computational Complexity</span>

Consider generating a sequence of \(T\) tokens. The
actor at node \(t\) must compute attention over all nodes \(u \le t\). Each node \(t\) involves:
- Computing query, key, and value given residual stream state: \(\mathcal{O}(D^2)\) (matrix-vector multiplication
  with \(D \times D\) weight matrices)
- Computing \(t\) dot products between query and keys: \(\mathcal{O}(tD)\)  
- Weighted sum of \(t\) value vectors: \(\mathcal{O}(tD)\)

So the actor at node \(t\) does \(\mathcal{O}(D^2 + tD)\) work. Summing across all nodes:

```
Total work = \(\sum_{t=1}^T \mathcal{O}(D^2 + tD)\) 
           = \(\mathcal{O}(TD^2) + \mathcal{O}\big(D \cdot \sum_{t=1}^T t\big)\) 
           = \(\mathcal{O}(TD^2) + \mathcal{O}(T^2D)\)
           = \(\mathcal{O}(T^2D)\)  [when \(T > D\), which holds in modern applications]
```

Intuitively, this quadratic scaling makes sense: each residual actor does work proportional to the
index of its node in the sequence. The average
workload grows linearly with sequence length, and we have \(T\) actors, yielding \(\mathcal{O}(T^2D)\) total
complexity.

As a first approximation, this \(\mathcal{O}(T^2D)\) complexity is the central bottleneck in scaling transformers
to long contexts, though as we'll see shortly, there is some nuance to this picture. Much of the
attention variant literature aims to attack this \(\mathcal{O}(T^2D)\) term. 

An important thing to note is that both the QK and OV circuits contribute to this quadratic cost: the
linear work per stream comes from both interacting with all previous keys (QK circuit) and from
(weighted) summing all previous values. Thus, any attempt to break the quadratic barrier must address
both QK and OV circuits. 

### <span style="color: #007bff; font-weight: bold;">4.3 Aside: Nuances on Complexity</span>

Interestingly, in a [talk](https://www.youtube.com/watch?v=rBCqOTEfxvg&t=1080s) shortly after the original
Transformer paper, Łukasz Kaiser mentioned being nervous about the cost being quadratic in context
length, before Noam Shazeer pointed out that D was significantly larger than T, so the `O(T²D)` term
wasn't the bottleneck. Their application was language translation of sentences, so T was just ~70 in
their context! It's striking to hear becuase in under a decade we've gone from translating sentences
to thinking about how to push models to reason about corpuses of over a million tokens!

Another important detail to keep in mind when discussing the complexity of attention is that attention is
highly parallel, so actual wall-clock time differs significantly from raw FLOP counts. An
interesting lens for thinking about complexity in a world of increasing compute is: what is the complexity of an algorithm in the limit of infinite parallel compute? For a fascinating deep dive on this, see
["Attention is Logarithmic (Actually)"](https://supaiku.com/attention-is-logarithmic).

---

# 5. Attention Heads: Work-Partitioning and Low-Rank Updates

The standard framing of multi-head attention is about <span style="color: #007bff; font-weight: bold;">work-partitioning</span>: keys, queries, and
values are sliced along the embedding dimension, heads perform attention independently on their
slices, the results are concatenated and then projected using W_O before being added to the residual
stream.

In pseudocode:

```
# concat-then-project formulation
# Let h_t^1, h_t^2, ..., h_t^H denote the outputs from each of H heads
# (each is a weighted average of values from that head)

h_t = concat(h_t^1, …, h_t^H)  # concatenate head outputs
z_{t,l} = x_{t,l} + W_O · h_t    # project and add to residual stream
```

A key thing to notice is that concatenating vectors then multiplying by a matrix is equivalent to
sharding the matrix, multiplying vectors by the corresponding shards independently, and adding the
results.

```
# independent-adds formulation
# W_O^h is the slice of W_O corresponding to head h
z_{t,l} = x_{t,l} + Σ_h (W_O^h · h_t^h)
```

Each head writes into the residual through its own projection slice `W_O^h`.

The low-rank additive framing plays a significant role in mechanistic interpretability work. A few
consequences:

* <span style="color: #007bff; font-weight: bold;">Low-rank, subspace-targeted writes.</span> A head can only modify the residual within the column
  space of `W_O^h` - at most rank `d_h`. Heads are low-rank writers into subspaces of the shared
  stream.

* <span style="color: #007bff; font-weight: bold;">Potentially Limited interaction between heads.</span> If two heads write largely into disjoint or
  orthogonal subspaces, later computation may treat their contributions as independent. Overlap
  enables interaction or interference. The geometry of `W_O` partitions bandwidth.

* <span style="color: #007bff; font-weight: bold;">Implicit memory management.</span> Updates are additive and persistent. Information written by a head
  sticks around unless future layers actively overwrite or counter-write it. Since bandwidth is finite
  (dimension D), writing one thing necessarily crowds others. Some heads compress or move information,
  others cache patterns for downstream use, and some act as cleaners.

---

# 6. The Combinatorics of Attention-Based Information Flows

With the information-flow graph picture established, we can now ask an interesting question: in how
many ways can information travel from one residual stream state \((t_1, l_1)\) to another \((t_2, l_2)\)? 

Recall that information moves through the graph by alternating between two types of edges:
* <span style="color: #007bff; font-weight: bold;">Horizontal moves</span> (attention): \((u, l) \to (t, l)\) where \(u < t\)
* <span style="color: #007bff; font-weight: bold;">Vertical moves</span> (residual): \((t, l) \to (t, l+1)\)

Let's look at a simple case. In how many ways can we travel from the first stream in one layer to the
last stream in the next layer, i.e. from \((1, l)\) to \((T, l+1)\)? There are a few types of paths: 
* All the way Right, then Up: \((1, l) \to (T, l) \to (T, l+1)\)
* Up, then all the way Right: \((1, l) \to (1, l+1) \to (T, l+1)\)
* Part way Right, Up, rest of the way Right: \((1, l) \to (k, l) \to (k, l+1) \to (T, l+1)\)

In the third case, there are \(T-2\) choices for \(k\) (namely \(k = 2, 3, \ldots, T-1\)), for a total of \(T\)
paths across all three cases. So even in a single layer network, there are already multiple paths
information from the first stream can take to reach the last stream.

More generally, any path from \((t, l)\) to \((t + p, l + q)\) requires a total of \(p\) horizontal moves
and \(q\) vertical moves. The number of ways to arrange these moves is the binomial coefficient
\(\binom{p+q}{p}\). By [Stirling's approximation](https://en.wikipedia.org/wiki/Stirling%27s_approximation), this grows exponentially with \(p + q\). In particular, as we
scale context length and depth, we quickly reach an astronomical number of paths from the beginning of
the first stream to the end of the final stream. This suggests possible redundancy: can we remove some
edges from our graph while still maintaining healthy information flow across streams?

---

# 7. Static Graph Sparsification

A natural idea based on the picture established so far is to sparsify the underlying information flow
graph, i.e. remove some edges, while preserving the ability for information to flow between any pair of
streams \(t_1\) and \(t_2\). Let's introduce notation to make this concrete. 

### 7.1 Neighborhoods and Receptive Fields
<span style="color: #007bff; font-weight: bold;">Neighborhoods</span>

Define \(N(t, l)\) as the <span style="color: #007bff; font-weight: bold;">attention neighborhood</span> of node \((t, l)\): that is, the set of nodes that the
actor at \((t, l)\) can attend to. The actor at \((t, l)\) computes attention only over nodes in
\(N(t, l)\), ignoring all others. In ordinary attention, we have \(N(t, l) = \{(1, l), (2, l), \ldots, (t, l)\}\), i.e. all previous nodes in the current layer. 

We'll see that a large number of efficient attention mechanisms boil down to simply defining `N(t, l)`
in different ways. In particular, these mechanisms **shrink** the neighborhood to some subset of the full ordinary neighborhood. Why does this help? We have the following observation:

Observation: if we fix neighborhood size to some constant `w`, the time complexity of generating `T` tokens is \(\mathcal{O}(TD^2 + TDw) = \mathcal{O}(TDw)\), assuming the second term still dominates. This is a 
factor of \(T/w\) saving over ordinary attention.

To see this, simply follow the derivation of time complexity in Section 3. For both the steps that contribute to the quadratic dependence on `T` - namely query-key interactions and 
value sums - we are now doing \(\mathcal{O}(wD)\) work for the `t`th stream instead of \(\mathcal{O}(tD)\).

<span style="color: #007bff; font-weight: bold;">Receptive Field</span>

Let's also make concrete the notion of "preserving information flow." We'll define the
<span style="color: #007bff; font-weight: bold;">receptive field</span> of node `(t, l)` as the set of input tokens that this node can "see" through the network. More
formally, it is the set of indices `i` such that there exists a path in the
information flow graph from node `(i, 0)` to node `(t, l)`.

In ordinary attention, the node `(t, l)` can "see" all tokens from 1 through `t`, 
because it receives information from all previous streams, so the receptive field is the full set `{1, ..., t}`. As we shrink neighborhoods, we will also shrink the receptive fields of some tokens. Thus, there is a tradeoff between neighborhood size and receptive field: smaller
neighborhoods yield lower attention cost, but also lower receptive field. 


### 7.2 Sliding Window Attention
In Sliding Window Attention, each actor attends only to its `w` most recent neighbors. In symbols, 
`N(t, l) = {(max(1, t-w+1), l), ..., (t, l)}`

**Time Complexity**
As we've established, since the neighborhood size is fixed to `w`, the time complexity of attention
will be `O(TD^2 + DTw)`

**Receptive Field**
Consider node `(t, 1)`. It can only see the `w` most recent tokens, i.e. tokens `t, t-1, ..., t-w+1`. If we go up a layer, the receptive field approximately doubles: `(t, 2)` can see back up to `(t-w+1, 1)`, which can see up to `(t-2*w+2, 0)`. Continuing in this manner, we see that the receptive field **grows linearly with depth**, i.e. the size of the receptive field of `(t, l)` is `O(lw)`. Put another way, we need about `T/w` layers to ensure the last stream
receives information from the first token. 

Sliding window attention thus gives us about a `T/w` complexity saving over ordinary attention, 
but at the cost of needing about `T/w` layers for information to propagate over the entire 
sequence. This is not great for long contexts, and so when sliding window attention is used in
practice, it's typically used in conjunction with ordinary attention (e.g. alternating layers, as in GPT OSS), as opposed to fully replacing it. 

### 7.3 Dilated Attention
Dilated attention is like sliding window attention but with "jumps." Instead of just looking at the last 
`w` nodes, we'll make jumps of length `d`, the dilation factor. In symbols, 
`N(t, l) = {(t, l), (t-d, l), (t-2d, l), (t-3d, l), ..., (t - (w-1)*d)}`

Consider what happens when we stack layers with dilation factors `1, w, w^2, ...` . In the first
layer, each node just talks to its closest `w` neighbors, as in sliding window. But in the second layer,
each node talks to `w` nodes, whose receptive fields are disjoint and each of size `w`, yielding a receptive field of size O(w^2). Continuing in this manner, we see that receptive field increases *exponentially* with depth, as opposed to linearly in sliding window attention. The time complexity is the same as in sliding window attention, but we now only need `log_{w}T` layers to establish full information flow, as opposed to `T/w` in sliding window attention.

### 7.4 Logarithmic Attention
Instead of using a fixed size jump inside a layer, consider what happens if we use an 
exponentially increasing jump size within a layer:
`N(t, l) = {(t, l), (t-1, l), (t-2, l), (t-4, l), (t-8, l), ... (t - 2^k)}`,
where `k = \floor{log_2(t)}`. 

The neighborhood size is now upper bounded by `log_{2}(T)`, implying a time complexity of
`O(TD^2 + DTlogT)`. We have the following nice observation: 

Claim: the receptive field of `(t, l)` where `l > log_{2}(t)` is the full set `{1, l}, ..., {t, l}`. In other words, we achieve full information flow within `log_{2}(t)` layers. 

Proof (sketch): the basic idea is that at any point we have the ability to jump right by any
power of 2. So to get from `(t, 1)` to `(t + d, l)`, simply follow the binary representation of `d`, i.e. write `d` as a sum of powers of 2, and make those jumps. 

### 7.5 Stochastic Masking
The idea here is to choose random subsets at each layer:
`N(t, l)` is a random subset of size `w` drawn from `{(1, l), ..., (t, l)}`.

As before, the time complexity is `O(TD^2 + TDw)`. Now, why would we expect randomly chosen
neighborhoods to yield good connectivity patterns? It turns out *random bipartite graphs are
expanders with high probability*. While a deep dive on this is beyond the scope of this article,
we'll briefly mention that:

1. The field of [spectral graph theory](https://web.stanford.edu/class/cs168/l/l11.pdf) quantifies
notions of "graphs with good information flow" we've been alluding to, via eigenvalues of matrices associated with the graph.

2. [Expanders](https://terrytao.wordpress.com/2011/12/02/245b-notes-1-basic-theory-of-expander-graphs/) are a special class of graphs that are sparse but preserve good information flow.

3. Random bipartite graphs, generated with appropriate hyperparameters, [are expanders with high
probability](https://theory.epfl.ch/courses/topicstcs/Lecture3.pdf).

### 7.6 Global Tokens & Sink Tokens
Global tokens can be used in conjunction with other static sparsification methods. The basic idea is to augment the neighborhood of each node with a common set of nodes called global tokens:
`N(t, l) = {(g_1, l), (g_2, l), ..., (g_k, l)} ∪ PrevNeighborhood(t, l)`

Sink tokens are a particular case, where the global tokens are the first `k` tokens of the sequence
for some `k`. This technique is used in [GPT-OSS](https://openai.com/index/introducing-gpt-oss/) in the sliding window attention layers. A theoretical basis for
global tokens is suggested in [taking a graph view](https://publish.obsidian.md/the-tensor-throne/Transformers+as+GNNs/Attention+sinks+from+the+graph+perspective), and sink tokens in
particular were advocated in the somewhat provocative post [Attention is Off By One](https://www.evanmiller.org/attention-is-off-by-one.html).

---

# 8. The Landscape of Efficient Attention Mechanisms

So far we've developed a mental framework for understanding what's going on in transformer models,
and used this framework to understand one particular family of efficient attention techniques as static sparsification of the underlying information flow graph. We'll now zoom out a bit, and briefly sketch the broader
landscape of efficient attention techniques, situating each within the framework we've developed.

### <span style="color: #007bff; font-weight: bold;">8.1 Static Sparsification</span>
This was the focus of the previous section. The core problem addressed by these techniques is the
quadratic cost involved with all query-key and attention weight-value interactions, and solutions
involve statically defining `N(t, l)` to reduce communication in both QK and OV circuits while
preserving receptive field growth. 

Notable techniques and papers: sliding windows and global
nodes ([Longformer](https://arxiv.org/abs/2004.05150)), dilations and block patterns ([BigBird](https://arxiv.org/abs/2007.14062), [H-Transformer-1D](https://arxiv.org/abs/2107.11906)), strided/block-sparse layouts ([Sparse Transformer](https://arxiv.org/abs/1904.10509)), star-shaped global hubs (Star-Transformer), and hierarchical dilations across layers (LongNet). (TODO - add links)

### <span style="color: #007bff; font-weight: bold;"> 8.2 Dynamic Sparsification and Routing</span>
Problem: static sparsification is <span style="color: #2ecc71; font-style: italic;">content-blind</span>, and involves potentially imbuing models with our imperfect structural priors about sequence modeling. An arguably
more "[bitter lesson](TODO -link)-pilled" idea is *dynamic* sparsity: let the model decide what edges
matter based on the content of the sequence being processed, thereby constructing `N(t, l)`
dynamically per-token.

Core challenge: How does an actor actually decide which nodes matter to it, without
scoring every previous key? Some ideas include: 
* LSH attention ([Reformer](https://arxiv.org/abs/2001.04451)): use [locality sensitive
hashing](https://en.wikipedia.org/wiki/Locality-sensitive_hashing) to place nearby bucket queries
and keys together, and only attend within buckets.
* [Routing Transformers] (https://arxiv.org/abs/2003.05997): learn cluster assignments or router tokens that group related positions.
* Approximate nearest neighbor methods: use ANN indices to retrieve top-k keys per query.
* [Deepseek Sparse Attention](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp): use two rounds. The first is a lightweight scoring mechanism
("lightning indexer") that is used to define `N(t, l)`, while the second does a full attention
score computation only on the neighborhood.

### <span style="color: #007bff; font-weight: bold;">8.3 Kernel and Low Rank Methods: Factorized Communication.</span>
Recall the standard algebraic formulation of attention:

```
Attn(Q, K, V) = Softmax( Mask(Q K^T) / √d ) · V
```

Many efficiency methods revisit this equation: they approximate the softmax kernel, approximate the
attention matrix by a low rank matrix, and/or reorder the multiplications so that K^TV is computed
first and QK^T is never materialized.

<span style="color: #007bff; font-weight: bold;">**Complementary graph view**</span>
From the perspective of information flow, these methods introduce intermediary nodes that mediate communication.
Instead of every node directly attending to every other, tokens first broadcast their values into a shared set of basis nodes, and later receive aggregated information back from them.
The result is a factor graph: communication happens in two hops, node → intermediary → node.

The factorized communication view unifies a range of specific techniques:
* Kernelized attention ([Performer](https://arxiv.org/abs/2009.14794), [Linear Transformer](https://arxiv.org/abs/2006.16236), [CosFormer](https://arxiv.org/abs/2202.08791)) – feature maps applied to
keys and queries effectively define intermediate nodes, that actors write to and read from based on
their keys and queries. 

* Low-rank and landmark methods ([Linformer](https://arxiv.org/abs/2006.04768), [Nyströmformer](https://arxiv.org/abs/2102.03902), [Perceiver IO](https://arxiv.org/abs/2107.14795)) – explicit landmark or latent nodes that summarize many streams.


### <span style="color: #007bff; font-weight: bold;"> 8.4 Graph Augmentation: Hubs, Highways, and Compression</span>
A number of efficient attention techniques can be viewed as augmenting a graph dominated by *local*
communication with *global* connectivity structure, such as long-range highways, summary blocks,
and global hubs. Examples include: 
* Caching past hidden states or compressing them into summaries ([Transformer XL](https://arxiv.org/abs/1901.02860), [Compressive Transformer](https://arxiv.org/abs/1911.05507)).
* Multi-scale graphs (hierarchical pyramids, blockwise attention with cross-block connectors).
* [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401), which can be viewed as augmenting our grid graph with off-grid nodes.


### <span style="color: #007bff; font-weight: bold;"> 8.5 Exact IO-aware kernels.</span>
In this article we've mostly focused on conceptual aspects of transformers, whereas efficient
real-world implementations require contending with the realities of modern hardware. Nevertheless,
the framework we've developed provides a helpful lens for understanding techniques like [Flash
Attention](https://arxiv.org/abs/2205.14135) and [Paged Attention](https://arxiv.org/abs/2309.06180). These can be viewed as defining **tilings** and **traversals** of the grid graph that compute attention in a *data-locality-aware* way, ensuring information moves efficiently across the hardware as well as across the model.


### <span style="color: #007bff; font-weight: bold;">8.6 KV-efficiency and head-sharing.</span>
A widely used idea in frontier models is to share keys and values across attention heads, while
keeping the queries separate. Key techniques include [Multi Query Attention](https://arxiv.org/abs/1911.02150) and [Grouped Query Attention](https://arxiv.org/pdf/2305.13245).

Some intuition for why we can get away with this comes from thinking about QK and OV circuits: it's the *combination* of queries and keys that determines where we look; thus, to ensure that different 
heads can look in different places, it suffices to vary just one of them across heads. Analogously
for the OV circuit - even with shared values, different heads can still write to different subspaces of the residual stream due to W_O.

---

# 9) Summary and Next Steps
