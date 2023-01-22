import jax
import jax.numpy as jnp
from jax import grad, value_and_grad
from jax import jit
from jax import random
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import pandas as pd
from amazing_printer import ap as aprint
from torch.utils.data import DataLoader
import seaborn as sns
from matplotlib import pyplot as plt
from fire import Fire
import seaborn as sns
from pathlib import Path


@dataclass
class Linear:
    inshape: int
    outshape: int

    def init(self, scale=1):
        key = jnp.array(np.random.rand(2), dtype=jnp.uint32)
        params = scale * random.normal(
            key, (self.outshape, self.inshape)
        ), scale * random.normal(key, (self.outshape, 1))
        return params

    def eval(self, params, X):
        W, b = params
        return ((W @ X.T) + b).T

    def __repr__(self):
        return f"--- Linear({self.inshape},{self.outshape}) ---"


@dataclass
class Relu:
    def init(self):
        self.params = 0.0
        return self.params

    def eval(self, params, x):
        return jax.nn.relu(x)

    def __repr__(self):
        return f"--- Relu() ---"


@dataclass
class Sigmoid:
    def init(self):
        self.params = 0.0
        return self.params

    def eval(self, params, x):
        return jax.nn.sigmoid(x)

    def __repr__(self):
        return "--- Sigmoid() ---"


@dataclass
class NeuralNet:
    layers: list
    params = []

    def init(self):
        for layer in self.layers:
            pm = layer.init()
            self.params.append(pm)

    def eval(self, params, X):
        for i, (layer, param) in enumerate(zip(self.layers, params)):
            if i == 0:
                yhat = layer.eval(param, X)
            else:
                yhat = layer.eval(param, yhat)
        return yhat

    def __repr__(self):
        return "\n".join(
            ["======= NNET ======="]
            + [str(lyr) for lyr in self.layers]
            + ["===================="]
        )


@dataclass
class MSE:
    def __init__(self, model):
        self.model = model
        self.loss = jit(self.loss_slow)
        self.lossgrad = jit(grad(self.loss_slow))
        
    def loss_slow(self, params, X, y):
        N = len(y)
        error = self.model.eval(params, X) - y
        error2 = error**2
        return jnp.sum(error2) / N


def train(mse, _X, _y, niters=200, lr=0.01, showpbar=True):
    if showpbar:
        pbar = tqdm(range(niters))
    else:
        pbar = range(niters)

    params = mse.model.params

    for i in pbar:
        ll, gg = mse.loss(params, _X, _y), mse.lossgrad(params, _X, _y)
        if showpbar:
            pbar.set_postfix({"loss": f"{ll:.3f}"})

        params = jax.tree_map(
            lambda p, g: p - lr * g , params, gg
        )

    mse.model.params = params
    mse.oldgrad = gg
    return ll, params


def main(imgfile, EPOCHS=5, lr=0.01, NBATCHES=10):
    # planted model
    np.random.seed(20)

    mylayers = [
        Linear(2, 3), Sigmoid()
    ]

    nn = NeuralNet(layers=mylayers)
    nn.init()
    print(nn)


    X = np.random.rand(10000,2)
    b = np.ones((3,1))
    W = np.array([[1,2],[3,4],[5,6]])
    y = jax.nn.sigmoid(((W @ X.T) + b).T)
    print(y.shape)
    yhat = nn.eval(nn.params,X)
    print(yhat.shape)
    BATCH_SIZE = int(y.shape[0]/NBATCHES)


    def mycollate(x):
        x, y = zip(*x)
        x = np.array(x)
        y = np.array(y)
        return x, y

    dl = DataLoader(
        [(X[i, :], y[i]) for i in range(len(y))],
        batch_size=BATCH_SIZE,
        collate_fn=mycollate,
        shuffle=True,
        pin_memory=False,
    )


    # define mse on model
    mse = MSE(model=nn)

    pbar = tqdm(range(EPOCHS))
    XX, yy = next(iter(dl))
    # __grad =  mse.lossgrad(mse.model.params,XX,yy)
    lastloss=np.inf
    lossdiff=0
    for epoch in pbar:
        # if lossdiff>0:lr = lr * 1.05 * (1+np.sin(np.pi*epoch/EPOCHS))
        # else: lr = lr / 1.05 / (1+np.sin(np.pi*epoch/EPOCHS))
        lr = lr * (1.005 + 0.003*np.sin(np.pi * epoch/EPOCHS))
        for batch in range(NBATCHES):
            XX, yy = next(iter(dl))
            __loss, __params = train(
                mse,
                XX,
                yy,
                niters=1,
                lr=lr,
                showpbar=False,
            )
            lossdiff = lastloss - __loss    
            lastloss = __loss   
            pbar.set_postfix({"loss": f"{__loss:.25f}"})

        print("\b"*len(str(mse.model.params[0])),end='')
        print(mse.model.params[0],end='')

if __name__ == "__main__":
    Fire(main)
    
