#pragma once


#include <iostream>


#include "../../Include/glad/glad.h"

void cudaInit(size_t x, size_t y, int scale, GLuint texture);
void computeField(float dt, int x1pos, int y1pos, int x2pos, int y2pos, bool isPressed);
void cudaExit();