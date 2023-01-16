#pragma once

#include <math.h>
#include <glad/glad.h>



void c_fluidSimInit(int width, int height, int scale, GLuint& texture);
void c_OnSimulationStep(float dt, float mouseX, float mouseY, bool isPressed);
void c_fluidSimFree();
double* c_getAverageTimes();