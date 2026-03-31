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
            data = self.data * other.data,
            parents(self,other),
            local_grads = (other.data, self.data),
        )

    def __rmul__(self,other):
        return self.__mul__(other)

    def __pow__(self,exponent):
        return ScalarValue(
            data = self.data**exponent,
            parents = (self,),
            local_grads = (exponent* self.data **(exponent-1),),
        )

    def __sub__(self, other):
        return self + (other *-1)

    def __neg__(self):
        return self * -1

    def __truediv__(self, other):
        return self * (1 / other)

    def log(self):
        return ScalarValue(
            data = math.log(self.data),
            parents = (self,),
            local_grads = (1/self.data,),
        )

    def exp(self):
        return ScalarValue(
            data = math.exp(self.data),
            parents = (self,),
            local_grads = (math.exp(self.data),)
        )

    def relu(self):
        return ScalarValue(
            data = max(0.0,self.data),
            parents = (self,),
            local_grads = (1 if self.data > 0 else 0,)
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
