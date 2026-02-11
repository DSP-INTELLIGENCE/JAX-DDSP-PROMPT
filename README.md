# JAX-DDSP-PROMPT

### JAX DSP PROMPT
Create a plan to implement in jax numpy: jax_<filename>.py
- no if/else in jax
- jax array are immutable 
- no dicts, classes, pytrees in jax dsp
- state is always a tuple
- store dsp parameters in state
- pass time varying parameters fo functions
- parameters are always passed to a function never a thing, packed or a tuple
- tick per sample 
- all parameters should be clamped to the limits of the dsp
- modulation and envelopes are always arrays everything else can be scalars
- lax.scan over tick if necessary
- vmap batches over lax.scan if needed
- you can not do typecasting inside jax ever (float,int) 
- you can never use python math ever it must always uise jax.numpy

