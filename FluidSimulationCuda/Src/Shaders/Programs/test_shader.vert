#version 330 core

layout (location=0) in vec2 aPos;

out vec2 TCoords;


void main()
{   
    TCoords = aPos;
    gl_Position = vec4(aPos, 1, 1.0);
}