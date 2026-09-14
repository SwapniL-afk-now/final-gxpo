# SLED Algorithm

SLED (**Self Logits Evolution Decoding**) modifies the final-token distribution
using information from earlier transformer layers.

## 1. Obtain logits from multiple layers

At each decoding step, obtain logits from the final layer and selected early
layers:

$$
\ell_N, \ell_{n_1}, \ell_{n_2}, \ldots
$$

where $N$ is the final layer.

## 2. Convert logits to probabilities

$$
p_n = \operatorname{softmax}\left(\frac{\ell_n}{\tau}\right)
$$

where $\tau$ is the temperature.

## 3. Select the top-$k$ final-layer tokens

Choose the $k$ tokens with the highest final-layer logits:

$$
I_k = \operatorname{TopK}(\ell_N, k)
$$

## 4. Estimate latent knowledge from each early layer

For each selected early layer $n$, calculate the logit-evolution direction:

$$
v_n = \ell_n - \ell_N
$$

For every candidate token $i \in I_k$, compare this direction with the
gradient associated with selecting token $i$:

$$
g_i^{(n)} = p_n - e_i
$$

where $e_i$ is the one-hot vector for token $i$.

Compute the positive cosine alignment:

$$
\bar m_i^{(n)} =
\max\left(\operatorname{CosSim}(v_n, g_i^{(n)}), 0\right)
$$

Square the alignment scores:

$$
m_i^{(n)} = \left(\bar m_i^{(n)}\right)^2
$$

The latent distribution estimated by layer $n$ is:

$$
P_{\text{latent}}^{(n)}(i) =
\frac{m_i^{(n)}}{\sum_{j \in I_k}m_j^{(n)}}
$$

## 5. Combine the early-layer distributions

Weight each layer according to its total alignment:

$$
s_n =
\frac{\sum_{i \in I_k}m_i^{(n)}}
{\sum_m \sum_{i \in I_k}m_i^{(m)}}
$$

Then combine the layer-specific distributions:

$$
P_{\text{latent}} =
\sum_n s_n P_{\text{latent}}^{(n)}
$$

This gives SLED's estimate of the model's latent knowledge for the next token.

## 6. Modify the final logits

Apply one KL-gradient-like update to the final logits:

$$
\tilde{\ell}_{N,i} =
\ell_{N,i} -
\frac{\alpha}{\tau}
\left(p_{N,i} - P_{\text{latent},i}\right)
$$

Thus, tokens favored by the early layers are increased, while tokens favored
only by the final layer are decreased.

Tokens outside the top-$k$ set are usually suppressed:

$$
\tilde{\ell}_{N,i} = \eta \qquad \text{for } i \notin I_k
$$

where $\eta$ is a very negative value, such as $-1000$.

## 7. Sample the next token

The next token is sampled from the modified distribution:

$$
p_{\text{SLED}} =
\operatorname{softmax}\left(\frac{\tilde{\ell}_N}{\tau}\right)
$$

## Simplified pseudocode

```text
for each decoding step:
    final_logits = logits from final layer
    early_logits = logits from selected early layers

    final_probs = softmax(final_logits / temperature)
    top_tokens = top_k(final_logits, k)

    latent_distribution = 0

    for each early layer:
        direction = early_logits - final_logits

        for each token in top_tokens:
            target_gradient = early_probs - one_hot(token)
            alignment = cosine(direction, target_gradient)
            alignment = max(alignment, 0) ** 2

        layer_distribution = normalize(alignment)
        layer_weight = sum(alignment)
        latent_distribution += layer_weight * layer_distribution

    latent_distribution = normalize(latent_distribution)

    modified_logits = final_logits - (
        alpha / temperature
        * (final_probs - latent_distribution)
    )

    modified_logits[not_top_tokens] = -1000

    next_token = sample(modified_logits)
```

## Current vLLM configuration

Our current selected-layer experiment uses:

```text
Selected layers: [14, 18, 22, 26]
Evolution scale k: 10
Evolution rate alpha: 2.0
Suppressed-token value: -1000
```

The original SLED method uses all early layers. The selected-layer version is
an efficiency and ablation variant. SLED is a logit-space gradient update, not
a transformer residual skip connection.

## Reference

[SLED: Self Logits Evolution Decoding](https://arxiv.org/pdf/2411.02433)
