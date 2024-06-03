# pynnlib: Python Neural Network Library

A library I use as a **submodule** for my other projects.


> [!IMPORTANT]
> Developed in free time, **my current** choices:
> - Can be integrated as a submodule only
> - Not stable API: this library must not constraint the different applications that use it
> - New functionalities are integrated step by step
> - Only basic comments. Documentation won't be published (handwritten)
> - Only basic and non optimized inference sessions (slow) are integrated into this open-source project
> - Execution providers for pytorch and tensorRT only: Nvidia, cpu (partial)
> - Coding rules are simplified a lot
> - No systematic validation tests and not in this repo

<br/>

# Install this library

## As a library in an untracked project
Unzip the code in a directory name `pynnlib`
or clone at the root of a project
```
git clone https://github.com/herlegon/pynnlib.git
```
## As a submodule in a git project
```
git submodule init
git submodule add https://github.com/herlegon/pynnlib.git
```

<br/>

# How I use it
> [!CAUTION]
> The following examples are not validated and use a default session for the inference. Do not expect fast inference.


## Open a model

```python
from pynnlib import (
    nnlib,
    NnModel,
)

model: NnModel = nnlib.open(model_filepath)
```


## Convert a model to a TensorRT model
...and save the engine

```python
from pynnlib import (
    nnlib,
    NnModel,
    TrtModel,
    ShapeStrategy,
)

model: NnModel = nnlib.open(model_filepath)
trt_model: TrtModel = nnlib.convert_to_tensorrt(
    model=model,
    shape_strategy=shape_strategy,
    fp16=fp16,
    # optimization_level=opt_level, # Not yet supported
    opset=opset,
    device=device,
    out_dir="output_dir",
)
```

## Perform an inference
with a default inference session.<br/>
PyTorch to transfer an image (bgr, np.float32) to the execution provider (device) using pageable memory.


```python
from pynnlib import (
    nnlib,
    NnModel,
    NnModelSession,
)
# ...

# Open an image, datatype must be np.float32
in_img: np.ndarray = load_image(img_filepaths)

# Open a model
model: NnModel = nnlib.open(model_filepath)

# Create a session
session: NnModelSession = nnlib.session(model)
session.initialize(
    device=device,
    fp16=fp16,
)

# Perform inference
out_img: np.ndarray = session.process(in_img)

# ...

```

## Use a custom inference session

```python
from pynnlib import (
    nnlib,
    NnFrameworkType,
)

# PyTorch
nnlib.set_session_constructor(
    NnFrameworkType.PYTORCH,
    PyTorchCuPySession
)

# TensorRT
nnlib.set_session_constructor(
    NnFrameworkType.TENSORRT,
    TensorRtCupySession
)

```
