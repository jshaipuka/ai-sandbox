# Building Decord in macOS for ARM64

The project illustrates how to build Decord in macOS for ARM64 architecture.

## Prerequisites

1. macOS (Tested with macOS Sequoia 15.5)
2. Homebrew (tested with Homebrew 4.5.8)
3. Miniconda (tested with conda 24.9.2)

## Build Steps

1. `brew install ffmpeg@5`
2. You may see the following output:
   ```text
   ffmpeg@5 is keg-only, which means it was not symlinked into /opt/homebrew,
   because this is an alternate version of another formula.
   
   If you need to have ffmpeg@5 first in your PATH, run:
     fish_add_path /opt/homebrew/opt/ffmpeg@5/bin
   
   For compilers to find ffmpeg@5 you may need to set:
     set -gx LDFLAGS "-L/opt/homebrew/opt/ffmpeg@5/lib"
     set -gx CPPFLAGS "-I/opt/homebrew/opt/ffmpeg@5/include"
   
   For pkg-config to find ffmpeg@5 you may need to set:
     set -gx PKG_CONFIG_PATH "/opt/homebrew/opt/ffmpeg@5/lib/pkgconfig"
   ```
   If you see it, run _all the four_ suggested commands.
3. `conda create -n decord-sandbox python=3.9 pip`
4. `conda activate decord-sandbox`
5. `conda install conda-forge::numpy=2.0.2`
6. Then we will follow the instructions
   from [here](https://github.com/georgia-tech-db/eva-decord?tab=readme-ov-file#mac-os). This is the Decord fork that
   supports newer macOS versions. We assume that Xcode Command Line Tools and CMake are already installed. If not,
   follow the instructions in the README.md to install them. Then do the following:
    1. `git clone --recursive https://github.com/georgia-tech-db/eva-decord.git`
    2. `cd eva-decord`
    3. `mkdir build && cd build`
    4. `cmake .. -DCMAKE_BUILD_TYPE=Release`
    5. `make`
    6. `cp ./libdecord.dylib ../python`
    7. `cd ../python`
    8. `pip install -e .`, this will install the locally built package to the `decord-sandbox` Conda environment.
       Alternatively, you can run `export PYTHONPATH=(pwd)` from there, this will allow other Python modules to see the
       built package (then you may need additional steps in PyCharm to run the project).
7. Now, in this project directory run `python main.py`.
