# Bazel CUDA Example
An example implementation of a Bazel workspace using CUDA libraries as an external Bazel target.

## Running the Example
Run the following:
```
bazel run //src:main
```
This should output something similar to the following:
```
$ bazel run //src:main 
INFO: Analyzed target //src:main (1 packages loaded, 1152 targets configured).
INFO: Found 1 target...
Target //src:main up-to-date:
  bazel-bin/src/main
INFO: Elapsed time: 0.317s, Critical Path: 0.13s
INFO: 2 processes: 1 internal, 1 linux-sandbox.
INFO: Build completed successfully, 2 total actions
INFO: Build completed successfully, 2 total actions
Found 1 CUDA device(s).
Device number: 0
  Device name: NVIDIA GeForce GTX 1070 Ti
  Compute capability: 6.1
  Device clock rate (MHz): 1683
  Device memory (GB): 8.51096
  Memory clock rate (MHz): 2002
  Memory bus width (bits): 256
  Memory bandwidth (GB/s): 256.256
```
