import math
import random

random.seed(67)

TEXT = "Betty Botter bought some butter but she said the butter's bitter.\nIf I put it in my batter it will make my batter bitter.\nBut a bit of better butter will make my batter better.\nSo she bought a bit of butter better than her bitter butter.\nAnd she put it in her batter and the batter was not bitter.\nSo it was better Betty Botter bought a bit of better butter.\n"

vocab_chars = sorted(set(TEXT))
vocab_size = len(vocab_chars)
CONTEXT_LEN = 8
EMBED_DIM = 16
NUM_LAYERS = 1
LEARN_RATE = 0.01

token_ids = [vocab_chars.index(c) for c in TEXT]
train_token_ids = token_ids[: int(len(token_ids) * 0.9)]
val_token_ids = token_ids[int(len(token_ids) * 0.9) :]


class ScalarValue:
    def __init__(self, data, parents=(), local_grads=()):
        self.data = float(data)
        self.grad = 0.0
        self.parents = parents
        self.local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, ScalarValue) else ScalarValue(other)
        return ScalarValue(self.data + other.data, (self, other), (1, 1))

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other = other if isinstance(other, ScalarValue) else ScalarValue(other)
        return ScalarValue(
            self.data * other.data, (self, other), (other.data, self.data)
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, exp):
        return ScalarValue(self.data**exp, (self,), (exp * self.data ** (exp - 1),))

    def __sub__(self, other):
        return self + (other * -1)

    def __neg__(self):
        return self * -1

    def __truediv__(self, other):
        if isinstance(other, ScalarValue):
            return self * (1.0 / other.data)
        return self * (1.0 / other)

    def log(self):
        return ScalarValue(math.log(self.data), (self,), (1 / self.data,))

    def exp(self):
        e = math.exp(self.data)
        return ScalarValue(e, (self,), (e,))

    def relu(self):
        return ScalarValue(max(0.0, self.data), (self,), (float(self.data > 0),))

    def backward(self):
        topo, seen = [], set()

        def dfs(node):
            if node not in seen:
                seen.add(node)
                for parent in node.parents:
                    dfs(parent)
                topo.append(node)

        dfs(self)
        self.grad = 1.0
        for node in reversed(topo):
            for parent, lg in zip(node.parents, node.local_grads):
                parent.grad += node.grad * lg


def make_weight_matrix(rows, cols):
    return [
        [ScalarValue(random.gauss(0, 0.08)) for _ in range(cols)] for _ in range(rows)
    ]


state_dict = {
    "wte": make_weight_matrix(vocab_size, EMBED_DIM),
    "wpe": make_weight_matrix(CONTEXT_LEN, EMBED_DIM),
    "lm": make_weight_matrix(vocab_size, EMBED_DIM),
}

for layer_idx in range(NUM_LAYERS):
    for proj in ("query", "key", "value", "out"):
        state_dict[f"layer{layer_idx}.{proj}"] = make_weight_matrix(
            EMBED_DIM, EMBED_DIM
        )
    state_dict[f"layer{layer_idx}.ffn1"] = make_weight_matrix(EMBED_DIM * 4, EMBED_DIM)
    state_dict[f"layer{layer_idx}.ffn2"] = make_weight_matrix(EMBED_DIM, EMBED_DIM * 4)

all_params = [p for w in state_dict.values() for row in w for p in row]


def linear(x, w):
    return [sum((wi * xi for wi, xi in zip(row, x)), ScalarValue(0.0)) for row in w]


def rms_norm(x):
    mean_sq = sum((xi * xi for xi in x), ScalarValue(0.0)) * (1 / len(x))
    inv_rms = (mean_sq + 1e-5) ** -0.5
    return [xi * inv_rms for xi in x]


def softmax(xs):
    max_val = max(x.data for x in xs)
    exps = [(x - max_val).exp() for x in xs]
    total = sum(exps, ScalarValue(0.0))
    return [e * (total**-1) for e in exps]


def transformer_forward(token_id, position, kv_keys, kv_vals):
    hidden = rms_norm(
        [
            te + pe
            for te, pe in zip(state_dict["wte"][token_id], state_dict["wpe"][position])
        ]
    )

    for layer_idx in range(NUM_LAYERS):
        residual = hidden
        normed_hidden = rms_norm(hidden)

        query = linear(normed_hidden, state_dict[f"layer{layer_idx}.query"])
        key = linear(normed_hidden, state_dict[f"layer{layer_idx}.key"])
        value = linear(normed_hidden, state_dict[f"layer{layer_idx}.value"])

        kv_keys[layer_idx].append(key)
        kv_vals[layer_idx].append(value)

        attn_logits = [
            sum(
                (query[j] * kv_keys[layer_idx][t][j] for j in range(EMBED_DIM)),
                ScalarValue(0.0),
            )
            * (EMBED_DIM**-0.5)
            for t in range(len(kv_keys[layer_idx]))
        ]

        attn_weights = softmax(attn_logits)

        context = [
            sum(
                (
                    attn_weights[t] * kv_vals[layer_idx][t][j]
                    for t in range(len(kv_vals[layer_idx]))
                ),
                ScalarValue(0.0),
            )
            for j in range(EMBED_DIM)
        ]

        attn_out = linear(context, state_dict[f"layer{layer_idx}.out"])
        hidden = [r + ao for r, ao in zip(residual, attn_out)]

        ffn_hidden = [
            xi.relu()
            for xi in linear(rms_norm(hidden), state_dict[f"layer{layer_idx}.ffn1"])
        ]
        ffn_out = linear(ffn_hidden, state_dict[f"layer{layer_idx}.ffn2"])
        hidden = [h + fo for h, fo in zip(hidden, ffn_out)]

    return linear(hidden, state_dict["lm"])


def compute_loss(seq_ids):
    n = min(CONTEXT_LEN, len(seq_ids) - 1)
    kv_keys = [[] for _ in range(NUM_LAYERS)]
    kv_vals = [[] for _ in range(NUM_LAYERS)]
    total_loss = ScalarValue(0.0)
    for pos in range(n):
        logits = transformer_forward(seq_ids[pos], pos, kv_keys, kv_vals)
        probs = softmax(logits)
        total_loss = total_loss + (probs[seq_ids[pos + 1]].log() * -1)
    return total_loss * (1 / n)


def generate_text(prompt="Betty Botter "):
    token_ids = [vocab_chars.index(c) for c in prompt]
    print(prompt, end="", flush=True)
    for _ in range(50):
        kv_keys = [[] for _ in range(NUM_LAYERS)]
        kv_vals = [[] for _ in range(NUM_LAYERS)]
        for pos, tok in enumerate(token_ids[-CONTEXT_LEN:]):
            logits = transformer_forward(tok, pos, kv_keys, kv_vals)
        next_tok = random.choices(
            range(vocab_size), weights=[p.data for p in softmax(logits)]
        )[0]
        token_ids.append(next_tok)
        print(vocab_chars[next_tok], end="", flush=True)
    print()


for step in range(501):
    if step % 100 == 0:
        val_loss = compute_loss(val_token_ids[: CONTEXT_LEN + 1]).data
        print(f"step {step}: val_loss={val_loss:.4f}")
        generate_text()

    start_idx = random.randint(0, len(train_token_ids) - CONTEXT_LEN - 1)
    loss = compute_loss(train_token_ids[start_idx : start_idx + CONTEXT_LEN + 1])
    loss.backward()

    for param in all_params:
        param.mean_sq_grad = 0.99 * getattr(param, "mean_sq_grad", 0.0) + 0.01 * (
            param.grad**2
        )
        param.data -= LEARN_RATE * param.grad / (param.mean_sq_grad**0.5 + 1e-8)
        param.grad = 0.0
