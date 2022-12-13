#pragma once


#include <iostream>
#include "../../Include/helper_math.h"

void cudaInit(size_t x, size_t y, int scale);
void computeField(uchar4* result, float dt, int x1pos, int y1pos, int x2pos, int y2pos, bool isPressed);
void cudaExit();