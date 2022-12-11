#pragma once

#include <math.h>
#include <glad/glad.h>



void init(int width, int height, int scale);
void on_frame(GLuint& texture, float dt, bool isPressed);
void on_mouse_button(int x, int y);