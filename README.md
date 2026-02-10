# Linearised-Modons
Julia scripts for studying the linear instabilities of modons.

## Scripts

- `linearised_modons.jl` - defines a function to solve the modon evolution problem. GeophysicalFlows.jl is used for the nonlinear case, and the linearised QG model for the linearised stability problem
- `linearised_QG.jl` - defines the linearised QG model using FourierFlows.jl to simulate the linear and nonlinear evolution of perturbations around a modon background state
- `run.jl` - runs the model for the specified parameters, looping through cases as required
- `make_frames.m` - converts the data files to frames and compiles these into mp4 movie files using ffmpeg

## FFMPEG

Ffmpeg is required for `make_frames.m`. This should be locally installed to a directory `ffmpeg/`.

Source: https://github.com/BtbN/FFmpeg-Builds/releases

To install ffmpeg-master-latest-linux64-gpl.tar.xz:

```
wget https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz
tar -xf ffmpeg-master-latest-linux64-gpl.tar.xz
```
