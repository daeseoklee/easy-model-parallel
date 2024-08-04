# easy-model-parallel

A minimal and simple-to-use library to implement model (tensor or pipeline) parallelism based on pytorch. This is a project created primarily to educate myself and has many limitations. However, feel free to contact me for any questions or requests.

## Design

### Outline 
The multiprocessing and inter-process communication are based on `torch.multiprocessing` and `torch.distributed`. 

Users write codes as if they are in a single process, but the same codes get to run parallelly across multiple processes, each having different `rank` parameter. At the same time, tensors, modules and operations are stored only in a designated process (and associated device). This is achieved by wrapping the tensors, modules and operations inside their shallow wrappers named `DeviceManagedTensor`, `DeviceManagedModule` and `DeviceManagedOperation`. These all have the property `data_device` (here, "device" is actually a pair of process rank and accelerator), which determines where the actual data is stored. On the processes not corresponding to `data_device`, the objects only hold placeholder data, which still play the important role of maintaining equivalent computation graphs across processes, so that gradients are sent across processes properly during back-propagation.


### Key properties 

- Computations are performed asynchronously, meaning a computation in one process does not block computations in the other processes, unless there is a need for data transfer.  
- The computation graph of each pipline remains in the device in which the forward-pass is performed (this is in contrast to what happens when you invoke `tensor.to(device)`). In other words, tensors are transferred across devices in a detached form but they keep the source information to use in backward. 


## How to use 

### Installation 
```
pip3 install --user pipx
pipx install uv

# Inside an existing conda environment
bash scripts/update_requirements.sh # This can be updated 
```

### Key concepts 

#### Device functor

A *Device functor* (`DeviceFunctor` object) turns tensors, modules and operations into `DeviceManagedTensor`, `DeviceManagedModule` and `DeviceManagedOperation` objects respectively. 

Device functor application can be performed with the following syntax: 
```
import torch
import emp

def main(devices):
    F = emp.DeviceFunctor(devices)
    device1 = F.allocate_device()
    device2 = F.allocate_device()
    
    x = ... 
    f = ...

    with F.under(device1):
        x = F(x) # torch.Tensor => DeviceManagedtensor
    with F.under(device2):
        f = F(f) # torch.nn.Module => DeviceManagedModule
        op = F(lambda a, b: ...) # Callable => DevicedManagedOperation
```

#### Inter-process data transfer

*Inter-process data transfers* happen when either a `DeviceManagedModule` or  a `DeviceManagedOperation` is called with the `DeviceManagedTensor` arguments whose `data_device` is different from its own `data_device`. A new data-switched `DeviceManagedTensor` is created internally, whose data is passed to module/operation.  

Continuing the preceding code block, 
```
    y = f(x) # Internally, the data of `x` is transferred from device 1 to device 2
```

#### Back-propagation 

Continuing the preceding code block,  
```
    y.backward() 
```
This handles back-propagation, which involves transferring the gradient of `x` from device 2 to device 1. As a result, `.grad` is filled for `x.tensor` at device 1 and the parameters in `f.module.parameters()` at device 2.  

#### Process launch

The preceding function `main` can be launched by:
```
if __name__ == "__main__:
    emp.launch(main, ["cuda:0", "cuda:1"])
```

### Examples 

- See `examples/ex1.py` to get a sense of how this library works. 
- See `examples/ex2.py` to learn how to implement model-parallel training. 


## Limitations 
- Currently, each process may use only one device. 
- The input/output signature of modules/operations to which device functors can be applied is limited (this could be easily fixed).
- No comprehesive test has   been performed.  