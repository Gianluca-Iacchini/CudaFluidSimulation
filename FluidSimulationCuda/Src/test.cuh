#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"
#include "cudart_platform.h"

#include <iostream>


#include <glad/glad.h>

void cudaInit(size_t x, size_t y, int scale, GLuint texture);
void computeField(float dt, int x1pos, int y1pos, int x2pos, int y2pos, bool isPressed);
void cudaExit();