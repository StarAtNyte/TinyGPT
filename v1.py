import math
import random

random.seed(42)

TEXT = "Peter Piper picked a peck of pickled peppers. A peck of pickled peppers Peter Piper picked. If Peter Piper picked a peck of pickled peppers,Where's the peck of pickled peppers Peter Piper picked?"

vocab_chars = sorted(set(TEXT))
vocab_size = len(vocab_chars)
CONTEXT_LEN = 8
EMBED_DIM = 16
NUM_LAYERS = 1
LEARN_RATE = 0.01

print(f"Vocab size: {vocab_size}")

all_token_ids = [vocab_chars.index(c) for c in TEXT]
train_token_ids = all_token_ids[: int(len(all_token_ids) * 0.9)]
val_token_ids = all_token_ids[int(len(all_token_ids) * 0.9) :]


# AUTOGRAD ENGINE
class ScalarValue:
    def __init__(self, data, parents=(), local_grads=()):
        self.data = float(data)
        self.grad = 0.0
        self.parents = parents
        self.local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, ScalarValue) else ScalarValue(other)
        return ScalarValue(
            data=self.data + other.data,
            parents=(self, other),
            local_grads=(1.0, 1.0),
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other = other if isinstance(other, ScalarValue) else ScalarValue(other)
        return ScalarValue(
            data=self.data * other.data,
            parents=(self, other),
            local_grads=(other.data, self.data),
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, exponent):
        return ScalarValue(
            data=self.data**exponent,
            parents=(self,),
            local_grads=(exponent * self.data ** (exponent - 1),),
        )

    def __sub__(self, other):
        return self + (other * -1)

    def __neg__(self):
        return self * -1

    def __truediv__(self, other):
        return self * (1 / other)

    def log(self):
        return ScalarValue(
            data=math.log(self.data),
            parents=(self,),
            local_grads=(1 / self.data,),
        )

    def exp(self):
        return ScalarValue(
            data=math.exp(self.data),
            parents=(self,),
            local_grads=(math.exp(self.data),),
        )

    def relu(self):
        return ScalarValue(
            data=max(0.0, self.data),
            parents=(self,),
            local_grads=(1 if self.data > 0 else 0,),
        )

    def backward(self):
        topo_order = []
        visited = set()

        def dfs(node):
            if node not in visited:
                visited.add(node)
                for parent in node.parents:
                    dfs(parent)
                topo_order.append(node)

        dfs(self)

        self.grad = 1.0

        for node in reversed(topo_order):
            for parent, local_grad in zip(node.parents, node.local_grads):
                parent.grad += node.grad * local_grad


def make_weight_matrix(num_rows, num_cols):
    return [
        [ScalarValue(random.gauss(0, 0.08)) for _ in range(num_cols)]
        for _ in range(num_rows)
    ]


state_dict = {
    "wte": make_weight_matrix(vocab_size, EMBED_DIM),
    "wpe": make_weight_matrix(CONTEXT_LEN, EMBED_DIM),
    "lm": make_weight_matrix(vocab_size, EMBED_DIM),
}

for layer_idx in range(NUM_LAYERS):
    prefix = f"layer{layer_idx}"
    for proj_name in ("query", "key", "value", "out"):
        state_dict[f"{prefix}.{proj_name}"] = make_weight_matrix(EMBED_DIM, EMBED_DIM)

    state_dict[f"{prefix}.ffn1"] = make_weight_matrix(EMBED_DIM * 4, EMBED_DIM)
    state_dict[f"{prefix}.ffn2"] = make_weight_matrix(EMBED_DIM, EMBED_DIM * 4)

all_params = [
    scalar
    for weight_matrix in state_dict.values()
    for row in weight_matrix
    for scalar in row
]


def linear(input_vec, weight_matrix):
    return [
        sum((w_ij * x_j for w_ij, x_j in zip(weight_row, input_vec)), ScalarValue(0))
        for weight_row in weight_matrix
    ]


def rms_norm(input_vec):
    n = len(input_vec)
    mean_sq = sum((xi * xi for xi in input_vec), ScalarValue(0.0)) / n
    inv_rms = (mean_sq + 1e-5) ** -0.5
    return [xi * inv_rms for xi in input_vec]


def softmax(logit_list):
    max_val = max(sv.data for sv in logit_list)
    exp_vals = [(sv - max_val).exp() for sv in logit_list]
    total = sum(exp_vals, ScalarValue(0.0))
    return [e / total for e in exp_vals]


def transformer_forward(token_id, position, kv_cache_keys, kv_cache_values):
    token_emb = state_dict["wte"][token_id]
    pos_emb = state_dict["wpe"][position]
    hidden = rms_norm([te + pe for te, pe in zip(token_emb, pos_emb)])

    for layer_idx in range(NUM_LAYERS):
        pfx = f"later{layer_idx}"

        residual = hidden
        norm_hidden = rms_norm(hidden)

        query_vec = linear(norm_hidden, state_dict[f"{pfx}.query"])
        key_vec = linear(norm_hidden, state_dict[f"{pfx}.key"])
        val_vec = linear(norm_hidden, state_dict[f"{pfx}.value"])

        kv_cache_keys[layer_idx].append(key_vec)
        kv_cache_values[layer_idx].append(val_vec)

        num_past_tokens = len(kv_cache_keys[layer_idx])

        scale_factor = EMBED_DIM**-0.5
        attention_logits = [
            sum(
                (query_vec[j] * kv_cache_keys[layer_idx][t] for j in range(EMBED_DIM)),
                ScalarValue(0.0),
            )
            * scale_factor
            for t in range(num_past_tokens)
        ]

        attention_weights = softmax(attention_logits)

        context_vec = [
            sum(
                (
                    attention_weights[t] * kv_cache_values[layer_idx][t]
                    for t in range(num_past_tokens)
                ),
                ScalarValue(0.0),
            )
            for j in range(EMBED_DIM)
        ]

        attn_out = linear(context_vec, state_dict[f"{pfx}.out"])
        hidden = [h + ao for h, ao in zip(residual, attn_out)]

        ffn_input = rms_norm(hidden)
        ffn_expanded = [
            xi.relu() for xi in linear(ffn_input, state_dict[f"{pfx}.ffn1"])
        ]
        ffn_output = linear(ffn_expanded, state_dict[f"{pfx}.ffn2"])
        hidden = [h + fo for h, fo in zip(hidden, ffn_output)]

    logits = linear(hidden, state_dict["lm"])
    return logits

def compute_loss(token_id_sequence):
