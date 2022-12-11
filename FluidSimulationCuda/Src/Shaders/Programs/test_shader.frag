#version 330 core

in vec2 TCoords;


uniform sampler2D dyeTexture;

out vec4 FragColor;

void main()
{
	vec2 ccolor = (TCoords / 2f) + 0.5f;
	FragColor = texture(dyeTexture, ccolor);
}