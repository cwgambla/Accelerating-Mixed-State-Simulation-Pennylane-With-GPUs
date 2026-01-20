# Utilizing Mix State GPU Acceleration With Pennylane
In this tutorial, we will utilize jax to enable Pennylane to run circuits utilizing a GPU backend. As this code will show, this will massively speed up the runtime of mixed state calculations.

### Implementation
First we need to import out libraries.


```python
import time
import jax
import jax.numpy as jnp
import pennylane as qml
from pennylane import numpy as np
```

### Benchmarking
To measure how effective GPU acceleration is, we first need something to compare it against. Below is a run of the mill pennylane circuit. This circuit is running utilizing the default.mixed device backend, which allows the use of mixed state calculations, and more importantly, has extensive support for creating noisy simulations.  We will be using this backend for the remainder of this tutorial. As a bit of little add on, slight phase damping has been added to demonstrate an example of noise be implemented on GPU acceleration.

However, this circuit will run without utilizing GPU acceleration. The circuit will run and be timed to see how long it runs. This will be our baseline that we will test against our GPU circuit. 


```python
wires = 10
dev = qml.device("default.mixed", wires=wires, shots=10000)

#control circuit
@qml.qnode(dev)
def noisy_circuit_np(p):
    qml.Hadamard(wires=0)
    for i in range(wires - 1):
        qml.CNOT(wires=[i, i+1])
    qml.PhaseDamping(p, wires=0)
    return qml.expval(qml.PauliZ(wires-1))


p_val = np.array(0.2, requires_grad=True)
start = time.time()
res = noisy_circuit_np(p_val)
duration = time.time() - start

print(f"Standard NumPy Run (10000 shots): {duration:.4f}s")
print(f"Expectation Value: {res}")
```

    Standard NumPy Run (10000 shots): 0.9210s
    Expectation Value: 0.0024


However, for purposes that will become clear later, we are going to run this circuit and time its speed again. Notice how the time to run the circuit remains nearly the same.


```python
p_val = np.array(0.2, requires_grad=True)
start = time.time()
res = noisy_circuit_np(p_val)
duration = time.time() - start

print(f"Standard NumPy Run (10000 shots): {duration:.4f}s")
print(f"Expectation Value: {res}")
```

    Standard NumPy Run (10000 shots): 0.9273s
    Expectation Value: 0.0032



```python
#Do not worry about this line
#This line is was to enable compatability with x_64 architecture
#It will vary from machine to machine what will be needed here
jax.config.update("jax_enable_x64", True)
```

Now we will utilize GPU to accelerate out circuit. The circuit uses jax.jit, and implementing the circuit with a jax interface. In addtion, instead of passing through a numpy array with the probabilities, we utilize the jnp.array(), which is much faster. Notice how the backend stays the same. However, there is a compilation performance penalty the first time running the circuit as jax moves all the neccessary calculations to the GPU. This will hurt initial performance.


```python
wires = 10
dev = qml.device("default.mixed", wires=wires, shots=10000)

@jax.jit
@qml.qnode(dev, interface="jax")
def noisy_circuit_jax(p):
    qml.Hadamard(wires=0)
    for i in range(wires - 1):
        qml.CNOT(wires=[i, i+1])
    qml.PhaseDamping(p, wires=0)
  
    return qml.expval(qml.PauliZ(wires-1))

p_jax = jnp.array(0.2)
start = time.time()
res = noisy_circuit_jax(p_jax).block_until_ready()
print(f"JAX GPU Run (10000 shots): {time.time() - start:.4f}s")
print(f"Expectation Value: {res}")
```

    JAX GPU Run (10000 shots): 0.7936s
    Expectation Value: -0.009000000000000001


The performance, while better, is still not the kind of performance that we are looking for. However, when we run the circuit a second time, we see a massive boost in performance.


```python
p_jax = jnp.array(0.2)
start = time.time()
res = noisy_circuit_jax(p_jax).block_until_ready()
print(f"JAX GPU Run (10000 shots): {time.time() - start:.4f}s")
print(f"Expectation Value: {res}")
```

    JAX GPU Run (10000 shots): 0.1337s
    Expectation Value: -0.0016


The circuit runs nearly 5x times faster than if it was not accelerated by the GPU. After initial compilation, this circuit will continue to run at the increased speed. This is very useful for when repeat circuit runs are necessary, such as with VQE or QAOA simulation.

This GPU speed is now possible because default.mixed is now has a compatible backend with JAX, allowing massive speedups for qubit simulations. Below is the link to the documentation for default.mixed.

https://pennylane.ai/devices/default-mixed

Thank you to the teams at Pennylane/Xanadu for making this possible, and for all you help in making this tutorial happen. Hopefully this tutorial helps you in your journey with QIS. 

### Version Notes
Below are system details for what the notebook was run on:

Python version: 3.10.17

JAX version: 0.6.0

PennyLane version: 0.42.3

Qiskit version: 1.2.4

Qiskit-aer version: 0.15.1

GPU used: NVIDIA GeForce RTX 4070
