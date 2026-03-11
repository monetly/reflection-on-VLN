## A Reflection on My Understanding of VLN as an Interface Problem

Over the past period, my understanding of Vision-and-Language Navigation (VLN) has changed substantially.  
My initial view was to treat VLN primarily as a **representation learning problem**: if one could construct a sufficiently informative semantic representation from language and vision, then the downstream control behavior should improve accordingly. Formally, I was implicitly assuming that once a high-level representation
\[
z_t = f_\theta(o_t, x)
\]
is sufficiently expressive, where \(o_t\) denotes the current observation and \(x\) denotes the instruction, then a policy
\[
a_t \sim \pi_\phi(\cdot \mid z_t)
\]
should be able to produce reliable actions.

However, my recent experiments suggest that this assumption is too optimistic. In practice, VLN is not merely a problem of constructing a semantically meaningful \(z_t\), but rather a problem of whether the semantic information can be converted into an **executable and stable control interface**.

### 1. From representation-centered thinking to closed-loop thinking

My earlier attempts followed a fairly consistent logic: improve the semantic side, then expect the policy side to follow. Concretely, I explored several directions:
- injecting VLM-based semantic signals into NoMaD-like navigation pipelines,
- using arrow-annotated images to let a VLM directly choose actions,
- treating detected objects as direct IBVS targets,
- hand-defining anchors such as hallway or doorway regions.

All of these attempts can be interpreted as trying to define an intermediate variable
\[
g_t \in \mathcal{G},
\]
where \(\mathcal{G}\) is some semantic or geometric goal space, and then using \(g_t\) to guide low-level execution. The working assumption was that if \(g_t\) is “good enough” semantically, then it should also be useful for control.

What I gradually realized is that the actual system performance depends not on the semantic quality of \(g_t\) alone, but on whether \(g_t\) induces a well-conditioned control problem. In other words, the important quantity is not simply
\[
I(g_t; y_t),
\]
where \(y_t\) may denote the semantic target or instruction relevance, but rather whether \(g_t\) also provides sufficient information for action selection:
\[
I(g_t; a_t^\star \mid o_t),
\]
where \(a_t^\star\) denotes the action that would be desirable for successful execution at time \(t\).

A representation may be semantically meaningful while still being control-irrelevant.

### 2. Semantic correctness does not imply control correctness

This became clear across multiple experiments.

#### 2.1 VLM-guided navigation
When I tried to use a VLM to constrain a navigation model, the semantic signal was often correct at a coarse level, but too weak to impose a precise local constraint on the policy. If the policy requires a strong executable subgoal, then a vague semantic preference is not enough. This reveals a mismatch between two spaces:
- a **semantic relevance space**,
- and an **actionable control space**.

Formally, if the VLM outputs a score
\[
s(g_t \mid o_t, x),
\]
this score may correlate with semantic relevance, but it does not necessarily define a good control objective unless
\[
\arg\max_{g_t} s(g_t \mid o_t, x)
\]
also lies in a region that is reachable, stable, and beneficial for low-level execution.

#### 2.2 Token-level action selection
In the arrow-based setting, I attempted to map action selection directly to token probabilities, essentially using
\[
a_t = \arg\max_{a \in \mathcal{A}} p_\theta(a \mid o_t, x).
\]
This produced stronger constraints than vague semantic grounding, but the resulting controller was unstable. The main issue was that token probabilities are not naturally calibrated as control values; they are affected by language-model biases, tokenization artifacts, and prompt conventions. In particular, the first-token probability often does not behave like a physically meaningful action score.

So although this setting improves semantic selectivity, it does not resolve the fundamental issue that the output space is not aligned with closed-loop control.

#### 2.3 Object-centered IBVS
When I switched to object-based IBVS, the system gained a more direct geometric grounding. If an object target is represented by image features \(u_t\), then IBVS tries to minimize
\[
e_t = u_t - u^\star,
\]
where \(u^\star\) is the desired image configuration. This gives a clean local control law, but I found that the object itself is often a poor navigation target. An object may be semantically relevant, but:
- it may not lie in free space,
- its image center may not correspond to a traversable subgoal,
- and “moving toward the object” is not always the same as “moving toward a useful navigational state.”

Thus object-centric grounding improves local geometric consistency, but often fails as a general navigation abstraction.

### 3. The key issue is not representation quality alone, but interface executability

This led me to a more important perspective: the core problem is not only how to encode semantics, but how to define an intermediate interface
\[
g_t = \Gamma(o_t, x)
\]
such that the downstream policy
\[
a_t \sim \pi(\cdot \mid o_t, g_t)
\]
is both stable and effective.

What matters is not whether \(g_t\) is interpretable or elegant in isolation, but whether it satisfies at least the following properties:
1. **Reachability**: \(g_t\) corresponds to a state or subgoal that is locally attainable.
2. **Executability**: \(g_t\) induces a control problem with sufficiently low ambiguity.
3. **Task relevance**: moving toward \(g_t\) increases the probability of eventual task success.
4. **Persistence or manageability**: if used over long horizons, \(g_t\) should be representable in a form that can be updated, remembered, or discarded.

This reframing changed how I evaluate candidate intermediate representations. Instead of asking whether an anchor is semantically intuitive, I now ask whether it serves as a valid control interface.

### 4. Why OpenFrontier felt close to my earlier thinking

My reaction to OpenFrontier was particularly revealing. At first glance, it felt very close to what I had been trying to do: identify something in the image and move toward it. But the crucial difference is that OpenFrontier does not let the VLM choose among arbitrary objects. Instead, it first defines a set of navigationally meaningful candidates—frontiers—and then asks the semantic module to score them.

If we denote the candidate set by
\[
\mathcal{C}_t = \{c_t^{(1)}, \dots, c_t^{(K)}\},
\]
OpenFrontier effectively solves
\[
c_t^\star = \arg\max_{c \in \mathcal{C}_t} \Big( \alpha \cdot \text{semantic\_score}(c) + \beta \cdot \text{exploration\_utility}(c) \Big),
\]
where each \(c\) is already constrained to be a navigationally valid candidate.

This clarified for me that the decisive factor is often not the semantic scorer itself, but the design of the candidate space \(\mathcal{C}_t\). My earlier systems often skipped this step and tried to use semantic entities directly as control targets. That was a structural weakness.

### 5. Low-level policy is not an implementation detail

Another major shift in my understanding concerns the role of the low-level policy. Initially, I treated the policy mostly as a downstream component that should be able to exploit a sufficiently good semantic interface. However, I gradually realized that the low-level policy is not a passive recipient. Its structure determines what kinds of intermediate goals are even meaningful.

In practice, the full system is better described as:
\[
(o_t, x) \xrightarrow{\Gamma} g_t \xrightarrow{\pi} a_t \xrightarrow{\mathcal{T}} o_{t+1},
\]
where \(\mathcal{T}\) denotes the environment transition.  
The quality of \(g_t\) can only be judged relative to \(\pi\) and \(\mathcal{T}\). A semantically attractive \(g_t\) is useless if the induced transition dynamics are unstable, myopic, or ambiguous.

This is why I no longer view classic PointGoal-style workers or local navigation policies as “boring baselines.” They define the geometry of the interface itself. If \(\pi\) is fixed, then the meaningful question becomes: what class of \(g_t\) makes
\[
J(\pi, \Gamma) = \mathbb{E}\left[\sum_t r_t\right]
\]
largest under resource and observability constraints?

### 6. My current view of VLN

At this point, I no longer see VLN primarily as a benchmark for multimodal understanding. I see it more as an **embodied systems problem** in which semantics, candidate generation, local execution, and memory interact.

More specifically, I now think the central question is not

\[
\text{“How do we make the model understand the instruction?”}
\]

but rather

\[
\text{“How do we transform semantic information into an intermediate variable that can stably constrain low-level execution?”}
\]

That is, VLN is fundamentally an **interface design problem**:
\[
(o_t, x) \mapsto g_t \mapsto a_t,
\]
and the difficulty lies in designing \(g_t\) so that it is simultaneously semantically meaningful and control-relevant.

### 7. What I learned from this process

Perhaps the most important lesson for me is methodological rather than technical. For a long time, when a method did not work, I tended to assume that the issue lay in details—prompting, parameter settings, token choices, minor interface changes—rather than in the formulation itself. This led me to spend a long time trying to repair unstable approaches.

What I understand more clearly now is that if a method only works under fragile configurations, then the problem may not be one of tuning but of formulation. A useful research habit is therefore not only to optimize details, but also to ask whether the interface itself is structurally aligned with the task.

### Final statement

If I were to summarize my current understanding in one sentence, it would be this:

\[
\boxed{
\text{VLN is not primarily a representation problem; it is a problem of designing executable interfaces between semantic understanding and closed-loop control.}
}
\]

This is the main shift in my thinking. Going forward, I want to place less emphasis on finding ever more elegant semantic abstractions, and more emphasis on whether a proposed intermediate representation genuinely supports stable, low-cost, and task-relevant embodied execution.
