#pragma once

#include <math.h>
#include <glad/glad.h>



void init(int width, int height, int scale, GLuint& texture);
void on_frame(float dt, float mouseX, float mouseY, bool isPressed);
void on_mouse_button(int x, int y);