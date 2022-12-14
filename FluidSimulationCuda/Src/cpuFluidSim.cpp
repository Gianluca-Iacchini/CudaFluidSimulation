#include "cpuFluidSim.h"
#include <algorithm>
#include <vec2.h>
#include <helper_math.h>
#include <thread>


#define CLAMP(val, minv, maxv) fminf(maxv, fmaxf(minv, val))
#define MIX(v0, v1, t) v0 * (1.f - t) + v1 * t 

float deltTime = 0.f;
float _timePassed = 0.f;

int w = 512;
int h = 512;

int nx = 256;
int ny = 256;

int iterations = 5;
float vorticity = 0.35f;
float velocity_diffusion = 0.8f;
float density_diffusion = 0.8f;
float aDecay = 1.2f;

vec2f mousePos;
vec2f lastMousePos;

vec2f* old_velocity;
vec2f* new_velocity;

vec3f* old_color;
vec3f* new_color;

uchar4* pixels;

float* old_pressure;
float* new_pressure;
float* divergence;

float* abs_curl;

GLuint cpuTexture = -1;

#define FOR_EACH_CELL for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++)

vec3f _colorArray[7];
static vec3f _currentColor;

void init(int width, int height, int scale, GLuint& texture) {

    _colorArray[0] = { 1.0f, 0.0f, 0.0f };
    _colorArray[1] = { 0.0f, 1.0f, 0.0f };
    _colorArray[2] = { 1.0f, 0.0f, 1.0f };
    _colorArray[3] = { 1.0f, 1.0f, 0.0f };
    _colorArray[4] = { 0.0f, 1.0f, 1.0f };
    _colorArray[5] = { 1.0f, 0.0f, 1.0f };
    _colorArray[6] = { 1.0f, 0.5f, 0.3f };

    int idx = rand() % 7;
    _currentColor = _colorArray[idx];

    cpuTexture = texture;

    w = width;
    h = height;

    nx = width / scale;
    ny = height / scale;


    old_velocity = new vec2f[nx * ny];
    new_velocity = new vec2f[nx * ny];

    old_color = new vec3f[nx * ny];
    new_color = new vec3f[nx * ny];
    

    pixels = new uchar4[nx * ny];

    old_pressure = new float[nx * ny];
    new_pressure = new float[nx * ny];

    divergence = new float[nx * ny];
    abs_curl = new float[nx * ny];



    FOR_EACH_CELL{
    old_color[x + y * nx] = vec3f {0.0f, 0.0f, 0.0f};
    old_velocity[x + y * nx] = vec2f{0.0f, 0.0f};
    }
}

vec3f interpolate(vec3f* grid, vec2f p) {
    float x1 = (int)p.x;
    float y1 = (int)p.y;
    float x2 = (int)p.x + 1;
    float y2 = (int)p.y + 1;

    vec3f p0, p1, p2, p3;

    p0 = grid[int(CLAMP(y1, 0.0f, ny - 1)) * nx + int(CLAMP(x1, 0.0f, nx - 1))];
    p1 = grid[int(CLAMP(y1, 0.0f, ny - 1)) * nx + int(CLAMP(x2, 0.0f, nx - 1))];
    p2 = grid[int(CLAMP(y2, 0.0f, ny - 1)) * nx + int(CLAMP(x1, 0.0f, nx - 1))];
    p3 = grid[int(CLAMP(y2, 0.0f, ny - 1)) * nx + int(CLAMP(x2, 0.0f, nx - 1))];


    float tx = (p.x - x1) / (x2 - x1);
    float ty = (p.y - y1) / (y2 - y1);

    vec3f u1 = MIX(p0, p1, tx);
    vec3f u2 = MIX(p2, p3, tx);

    return MIX(u1, u2, ty);
}


vec2f interpolate(vec2f* grid, vec2f p) {
    float x1 = (int)p.x;
    float y1 = (int)p.y;
    float x2 = (int)p.x + 1;
    float y2 = (int)p.y + 1;

    vec2f v0, v1, v2, v3;

    v0 = grid[int(CLAMP(y1, 0.0f, ny - 1)) * nx + int(CLAMP(x1, 0.0f, nx - 1))];
    v1 = grid[int(CLAMP(y1, 0.0f, ny - 1)) * nx + int(CLAMP(x2, 0.0f, nx - 1))];
    v2 = grid[int(CLAMP(y2, 0.0f, ny - 1)) * nx + int(CLAMP(x1, 0.0f, nx - 1))];
    v3 = grid[int(CLAMP(y2, 0.0f, ny - 1)) * nx + int(CLAMP(x2, 0.0f, nx - 1))];


    float tx = (p.x - x1) / (x2 - x1);
    float ty = (p.y - y1) / (y2 - y1);

    vec2f u1 = MIX(v0, v1, tx);
    vec2f u2 = MIX(v2, v3, tx);

    return MIX(u1, u2, ty);
}

void advect_color(float dt, float aDecay) {
    float decay = 1.0f / (1.0f + aDecay * dt);
    FOR_EACH_CELL{
        vec2f pos = vec2f{float(x), float(y)} - dt * old_velocity[x + y * nx];
        vec3f p = interpolate(old_color, pos);
        p.x = fminf(1.0f, pow(p.x, 1.005f) * decay);
        p.y = fminf(1.0f, pow(p.y, 1.005f) * decay);
        p.z = fminf(1.0f, pow(p.z, 1.005f) * decay);
        new_color[x + y * nx] = p;
    }
    std::swap(old_color, new_color);
}

void advect_velocity(float dt, float aDecay) {
    float decay = 1.0f / (1.0f + aDecay * dt);
    FOR_EACH_CELL{
        vec2f pos = vec2f{float(x), float(y)} - dt * old_velocity[x + y * nx];
        new_velocity[x + y * nx] = interpolate(old_velocity, pos) * decay;
    }
    std::swap(old_velocity, new_velocity);
}

void diffuse_vel(float dt, float vDiffusion)
{
    float vAlpha = vDiffusion * vDiffusion / dt;
    float vBeta = 4.0f + vAlpha;

    for (int i = 0; i < 5; i++)
    {
        vec2f uL, uR, uB, uT, uC;
        FOR_EACH_CELL
        {
            // perfoms one iteration of jacobi method (diffuse method should be called 20-50 times per cell)
            uL = old_velocity[int(CLAMP(y, 0, ny - 1)) * nx + int(CLAMP(x - 1, 0, nx - 1))];
            uR = old_velocity[int(CLAMP(y, 0, ny - 1)) * nx + int(CLAMP(x + 1, 0, nx - 1))];
            uB = old_velocity[int(CLAMP(y - 1, 0, ny - 1)) * nx + int(CLAMP(x, 0, nx - 1))];
            uT = old_velocity[int(CLAMP(y + 1, 0, ny - 1)) * nx + int(CLAMP(x, 0, nx - 1))];
            uC = old_velocity[y * nx + x];

            new_velocity[y * nx + x] = (uT + uB + uL + uR + uC * vAlpha) * (1.f / vBeta);
        }
        std::swap(old_velocity, new_velocity);
    }
}

void diffuse_color(float dt, float dDiffusion)
{
    float dAlpha = dDiffusion * dDiffusion / dt;
    float dBeta = 4.0f + dAlpha;

    for (int i = 0; i < 5; i++)
    {
        vec3f dL, dR, dB, dT, dC;
        FOR_EACH_CELL
        {
            // perfoms one iteration of jacobi method (diffuse method should be called 20-50 times per cell)
            dL = old_color[int(CLAMP(y, 0, ny - 1)) * nx + int(CLAMP(x - 1, 0, nx - 1))];
            dR = old_color[int(CLAMP(y, 0, ny - 1)) * nx + int(CLAMP(x + 1, 0, nx - 1))];
            dB = old_color[int(CLAMP(y - 1, 0, ny - 1)) * nx + int(CLAMP(x, 0, nx - 1))];
            dT = old_color[int(CLAMP(y + 1, 0, ny - 1)) * nx + int(CLAMP(x, 0, nx - 1))];
            dC = old_color[y * nx + x];

            new_color[y * nx + x] = (dL + dR + dB + dT + dC * dAlpha) * (1.f / dBeta);
        }
    }
}

void diffuse(float dt, float dDiffusion, float vDiffusion) {
    std::thread vel{ diffuse_vel, dt, vDiffusion};
    std::thread col{ diffuse_color, dt, dDiffusion};
    vel.join();
    col.join();
}

void pressure_iteration() {

    memset(old_pressure, 0, nx * ny * sizeof(float));

    FOR_EACH_CELL{
        float dx = old_velocity[int(CLAMP(x+1, 0, nx-1)) + nx * int(CLAMP(y, 0, ny-1))].x - old_velocity[int(CLAMP(x - 1, 0, nx-1)) + nx * int(CLAMP(y, 0, ny-1))].x;
        float dy = old_velocity[int(CLAMP(x, 0, nx-1)) + nx * int(CLAMP(y+1, 0, ny-1))].y - old_velocity[int(CLAMP(x, 0, nx-1)) + nx * int(CLAMP(y-1, 0, ny-1))].y;
        divergence[x + y * nx] = dx + dy;
        old_pressure[x + y * nx] = 0.0f;
    }

    for (int k = 0; k < iterations; k++) {
        FOR_EACH_CELL{
            float sum = -divergence[x + y * nx]
                + old_pressure[int(CLAMP(x + 1, 0, nx - 1)) + nx * int(CLAMP(y, 0, ny - 1))]
                + old_pressure[int(CLAMP(x - 1, 0, nx - 1)) + nx * int(CLAMP(y, 0, ny - 1))]
                + old_pressure[int(CLAMP(x, 0, nx - 1)) + nx * int(CLAMP(y+1, 0, ny - 1))]
                + old_pressure[int(CLAMP(x, 0, nx - 1)) + nx * int(CLAMP(y-1, 0, ny -1))];
            new_pressure[x + y * nx] = 0.25f * sum;
        }
        std::swap(old_pressure, new_pressure);
    }
}

float curl(int x, int y) {
    float cL = old_velocity[int(CLAMP(x - 1, 0, nx - 1)) + nx * int(CLAMP(y, 0, ny - 1))].y;
    float cR = old_velocity[int(CLAMP(x + 1, 0, nx - 1)) + nx * int(CLAMP(y, 0, ny - 1))].y;
    float cT = old_velocity[int(CLAMP(x, 0, nx - 1)) + nx * int(CLAMP(y + 1, 0, ny - 1))].x;
    float cB = old_velocity[int(CLAMP(x, 0, nx - 1)) + nx * int(CLAMP(y - 1, 0, ny - 1))].x;

    return 0.5f * (cR - cL - cT + cB);
}

void vorticity_confinement(float dt) {
    FOR_EACH_CELL{
        abs_curl[x + y * nx] = fabsf(curl(x, y));
    }
 
        FOR_EACH_CELL{
            vec2f direction;

            float cL = abs_curl[int(CLAMP(x - 1, 0, nx - 1)) + nx * int(CLAMP(y, 0, ny - 1))];
            float cR = abs_curl[int(CLAMP(x + 1, 0, nx - 1)) + nx * int(CLAMP(y, 0, ny - 1))];
            float cB = abs_curl[int(CLAMP(x, 0, nx - 1)) + nx * int(CLAMP(y - 1, 0, ny - 1))];
            float cT = abs_curl[int(CLAMP(x, 0, nx - 1)) + nx * int(CLAMP(y + 1, 0, ny - 1))];
            
            direction.x = cT - cB;
            direction.y = cR - cL;
            
            vec2f force = 0.5f * direction;
            force = force * (1.f / (length(force) + 0.0001));
            force = force * curl(x, y) * vorticity;
            force = force * (-1.0f);

            new_velocity[x + y * nx] = old_velocity[x + y * nx] + force * dt;
    }

    std::swap(old_velocity, new_velocity);
}


void apply_color_and_force(float dt, float mouseX, float mouseY)
{
    mouseY = h - 1 - mouseY;

    mousePos = vec2f{ mouseX * 1.0f * nx / w, mouseY * 1.0f * ny / h };

    _timePassed += deltTime;

    // apply gradient to color
    int roundT = int(_timePassed) % 7;
    int ceilT = int((_timePassed)+1) % 7;
    float w = _timePassed - int(_timePassed);
    _currentColor = _colorArray[roundT] * (1 - w) + _colorArray[ceilT] * w;



    FOR_EACH_CELL{

        float eU = expf(-((x - lastMousePos.x) * (x - lastMousePos.x) + (y - lastMousePos.y) * (y - lastMousePos.y)) / 10);
        vec2f uF = (lastMousePos - mousePos) * dt * 500 * eU;

        float eC = expf(-((x - mousePos.x) * (x - mousePos.x) + (y - mousePos.y) * (y - mousePos.y)) / 10);
        old_color[x + nx * y] += _currentColor * eC;
        old_velocity[x + y * nx] = old_velocity[x + y * nx] + uF;
    }
}


void fluid_simulation_step(float dt, float mouseX, float mouseY, bool isPressed) {

    //std::thread t(&pressure_iteration);
    advect_velocity(dt, aDecay);
    advect_color(dt, aDecay);


    if (isPressed)
    {
        apply_color_and_force(dt, mouseX, mouseY);
    }



    vorticity_confinement(dt);

    diffuse(dt, density_diffusion, velocity_diffusion);


    pressure_iteration();

   // t.join();

    FOR_EACH_CELL{
    old_velocity[x + y * nx].x -= 0.5f * (old_pressure[int(CLAMP(x + 1, 0, nx - 1)) + nx * int(CLAMP(y, 0, ny - 1))] - old_pressure[int(CLAMP(x - 1, 0, nx - 1)) + nx * int(CLAMP(y, 0, ny - 1))]);
    old_velocity[x + y * nx].y -= 0.5f * (old_pressure[int(CLAMP(x, 0, nx - 1)) + nx * int(CLAMP(y + 1, 0, ny - 1))] - old_pressure[int(CLAMP(x, 0, nx - 1)) + nx * int(CLAMP(y - 1, 0, ny - 1))]);
    }

    // zero out stuff at bottom
    for (int y = 0; y <= 10; y++)
    {
        for (int x = 0; x < nx; x++)
        {
            old_color[x + y * nx] = vec3f{ 0.0f, 0.0f, 0.0f };
            old_velocity[x + y * nx] = vec2f{ 0.0f, 0.0f };
        }
    }

    lastMousePos = mousePos;
}

void on_frame(float dt, float mouseX, float mouseY, bool isPressed) {

    deltTime = dt;

    fluid_simulation_step(dt, mouseX, mouseY, isPressed);

    // density field to pixels
    FOR_EACH_CELL{
        float R = old_color[y * nx + x].x;
        float G = old_color[y * nx + x].y;
        float B = old_color[y * nx + x].z;

        pixels[y * nx + x] = make_uchar4(fminf(255.0f, 255.0f * R), fminf(255.0f, 255.0f * G), fminf(255.0f, 255.0f * B), 255);
    }

    // upload pixels to texture

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, nx, ny, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
}

void on_mouse_button(int x, int y) {

}
