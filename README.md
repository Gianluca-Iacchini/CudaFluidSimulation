# CudaFluidSimulation

This repository contains the code for a 2D Eulerian fluid simulation.
The simulation is divided into two moduels, one for a pure CPU simulation, and one for a GPGPU simulation using CUDA.


## Requirements
CUDA and OpenGL are required to launch the simulation

## Building
The application was developed on Windows using Visual Studio 2022. To run the application on a Windows machine simply clone the repo and open the .sln file.
The code was not tested on Linux but should work out of the box; however compilation is required using nvcc.

## Usage
The default simulation is CPU only. To enable GPGPU simulation define a `GPU_SIM=1` macro or launch the application with the additional flag `-D GPU_SIM=1`.

Left Mouse Button: Add force and dye at mouse location.

Escape: Exit the application.

## Samples

<p align="center">
<img src="https://user-images.githubusercontent.com/34865358/212757748-0e72857f-4a13-4ca8-80ea-d3b3357b728c.png" width=30% height=30%>
<img src="https://user-images.githubusercontent.com/34865358/212757347-4e42e4b7-e192-4152-a82e-12872bb80d4b.png" width=45% height=45%>
</p>
