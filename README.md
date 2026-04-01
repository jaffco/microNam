# MicroNAM
Header-only [NAM](https://neuralampmodeler.com) inference with zero dependencies.

## Using

1. `#include "MicroNAM.h"` in your C++ project
2. Generate weights header with `nam_to_header.py <model.nam> <model.h>` and `#include` it
3. Example code below for instantiation in app:
```cpp
MicroNAM::NanoNet<1> mNamInstance; // audio block size 1
mNamInstance.load_weights(mModelWeights);
float inputBuffer[1];
float outputBuffer[1];
mNamInstance.forward(inputBuffer, outputBuffer);
```



## Awknowledgements

MicroNAM is based on an implementation shared by [nadavb](https://github.com/nadavb9001) in an [Electrosmith Daisy forum thread](https://forum.electro-smith.com/t/running-nam-on-embedded-hardware/9053/7?u=kidproquo).