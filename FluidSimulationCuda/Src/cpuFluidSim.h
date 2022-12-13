#pragma once

#include <math.h>
#include "../../Include/glad/glad.h"
#include <stdint.h>


void init(int width, int height, int scale);
void on_frame(uint32_t* data, float dt, bool isPressed);
void on_mouse_button(int x, int y);