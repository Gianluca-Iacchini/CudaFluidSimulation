#include <cpuFluidSim.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <vec2.h>
#include <helper_math.h>



#define CLAMP(val, minv, maxv) fminf(maxv, fmaxf(minv, val))
#define MIX(v0, v1, t) v0 * (1.f - t) + v1 * t 

//void draw_circle(float x, float y, float r, int n = 100) {
//    vec2f pos[n];
//    for (int i = 0; i < n; i++) {
//        float angle = 2.0f * 3.14159f * i / n;
//        pos[i] = vec2f{ x, y } + r * polar(angle);
//    }
//    draw(pos, n, GL_LINE_LOOP);
//}



int w = 512;
int h = 512;

int nx = 256;
int ny = 256;

int iterations = 5;
float vorticity = 10.0f;

float2 mousePos;
float2 lastMousePos;

float2* old_velocity;
float2* new_velocity;

float* old_density;
float* new_density;

uchar4* pixels;

float* old_pressure;
float* new_pressure;
float* divergence;

float* abs_curl;

#define FOR_EACH_CELL for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++)



void init(int width, int height, int scale) {


    w = width;
    h = height;

    nx = width / scale;
    ny = height / scale;


    old_velocity = new float2[nx * ny];
    new_velocity = new float2[nx * ny];

    old_density = new float[nx * ny];
    new_density = new float[nx * ny];
    

    pixels = new uchar4[nx * ny];

    old_pressure = new float[nx * ny];
    new_pressure = new float[nx * ny];

    divergence = new float[nx * ny];
    abs_curl = new float[nx * ny];



    FOR_EACH_CELL{
    old_density[x + y * nx] = 0.0f;
    old_velocity[x + y * nx] = float2{0.0f, 0.0f};
    }
}

float interpolate(float* grid, float2 p) {
    float x1 = (int)p.x;
    float y1 = (int)p.y;
    float x2 = (int)p.x + 1;
    float y2 = (int)p.y + 1;

    float p0, p1, p2, p3;

    p0 = grid[int(CLAMP(y1, 0.0f, ny - 1)) * nx + int(CLAMP(x1, 0.0f, nx - 1))];
    p1 = grid[int(CLAMP(y1, 0.0f, ny - 1)) * nx + int(CLAMP(x2, 0.0f, nx - 1))];
    p2 = grid[int(CLAMP(y2, 0.0f, ny - 1)) * nx + int(CLAMP(x1, 0.0f, nx - 1))];
    p3 = grid[int(CLAMP(y2, 0.0f, ny - 1)) * nx + int(CLAMP(x2, 0.0f, nx - 1))];


    float tx = (p.x - x1) / (x2 - x1);
    float ty = (p.y - y1) / (y2 - y1);

    float u1 = MIX(p0, p1, tx);
    float u2 = MIX(p2, p3, tx);

    return MIX(u1, u2, ty);
}


float2 interpolate(float2* grid, float2 p) {
    float x1 = (int)p.x;
    float y1 = (int)p.y;
    float x2 = (int)p.x + 1;
    float y2 = (int)p.y + 1;

    float2 v0, v1, v2, v3;

    v0 = grid[int(CLAMP(y1, 0.0f, ny - 1)) * nx + int(CLAMP(x1, 0.0f, nx - 1))];
    v1 = grid[int(CLAMP(y1, 0.0f, ny - 1)) * nx + int(CLAMP(x2, 0.0f, nx - 1))];
    v2 = grid[int(CLAMP(y2, 0.0f, ny - 1)) * nx + int(CLAMP(x1, 0.0f, nx - 1))];
    v3 = grid[int(CLAMP(y2, 0.0f, ny - 1)) * nx + int(CLAMP(x2, 0.0f, nx - 1))];


    float tx = (p.x - x1) / (x2 - x1);
    float ty = (p.y - y1) / (y2 - y1);

    float2 u1 = MIX(v0, v1, tx);
    float2 u2 = MIX(v2, v3, tx);

    return MIX(u1, u2, ty);
}

void advect_density(float dt) {
    FOR_EACH_CELL{
        float2 pos = make_float2(x, y) - dt * old_velocity[x + y * nx];
        new_density[x + y * nx] = interpolate(old_density, pos);
    }
    std::swap(old_density, new_density);
}

void advect_velocity(float dt) {
    FOR_EACH_CELL{
        float2 pos = make_float2(x, y) - dt * old_velocity[x + y * nx];
        new_velocity[x + y * nx] = interpolate(old_velocity, pos);
    }
    std::swap(old_velocity, new_velocity);
}

void diffuse_density(float dt) {
    float diffusion = dt * 100.01f;
    FOR_EACH_CELL{
        float sum =
            diffusion * (
             old_density[int(CLAMP(x - 1, 0, nx - 1)) + nx * int(CLAMP(y, 0, ny - 1))]
            + old_density[int(CLAMP(x + 1, 0, nx - 1)) + nx * int(CLAMP(y, 0, ny - 1))]
            + old_density[int(CLAMP(x, 0, nx - 1)) + nx * int(CLAMP(y - 1, 0, ny - 1))]
            + old_density[int(CLAMP(x, 0, nx - 1)) + nx * int(CLAMP(y + 1, 0, ny - 1))]
            )
            + old_density[x + y * nx];
        new_density[x + y * nx] = 1.0f / (1.0f + 4.0f * diffusion) * sum;
    }
    std::swap(old_density, new_density);
}

void diffuse_velocity(float dt) {
    float viscosity = dt * 0.000001f;
    FOR_EACH_CELL{
        float2 sum =
            viscosity * (
             old_velocity[int(CLAMP(x-1, 0, nx-1)) + nx * int(CLAMP(y, 0, ny-1))]
            + old_velocity[int(CLAMP(x + 1, 0, nx-1)) + nx * int(CLAMP(y, 0, ny-1))]
            + old_velocity[int(CLAMP(x, 0, nx-1)) + nx * int(CLAMP(y-1, 0, ny-1))]
            + old_velocity[int(CLAMP(x, 0, nx-1)) + nx * int(CLAMP(y+1, 0, ny-1))]
            )
            + old_velocity[x + y * nx];
        new_velocity[x + y * nx] = 1.0f / (1.0f + 4.0f * viscosity) * sum;
    }
    std::swap(old_velocity, new_velocity);
}

void project_velocity() {

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

    FOR_EACH_CELL{
        old_velocity[x + y * nx].x -= 0.5f * (old_pressure[int(CLAMP(x + 1, 0, nx - 1)) + nx * int(CLAMP(y, 0, ny - 1))] - old_pressure[int(CLAMP(x - 1, 0, nx - 1)) + nx * int(CLAMP(y, 0, ny - 1))]);
        old_velocity[x + y * nx].y -= 0.5f * (old_pressure[int(CLAMP(x, 0, nx - 1)) + nx * int(CLAMP(y + 1, 0, ny - 1))] - old_pressure[int(CLAMP(x, 0, nx - 1)) + nx * int(CLAMP(y -1, 0, ny - 1))]);
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
            float2 direction;
            //direction.y = abs_curl[int(CLAMP(x, 0, nx-1)) + nx * int(CLAMP(y - 1, 0, ny-1))] - abs_curl[int(CLAMP(x, 0, nx - 1)) + nx * int(CLAMP(y + 1, 0, ny - 1))];
    
            //direction.y = abs_curl[int(CLAMP(x+1, 0, nx - 1)) + nx * int(CLAMP(y, 0, ny - 1))] - abs_curl[int(CLAMP(x-1, 0, nx - 1)) + nx * int(CLAMP(y, 0, ny - 1))];
            float cL = abs_curl[int(CLAMP(x - 1, 0, nx - 1)) + nx * int(CLAMP(y, 0, ny - 1))];
            float cR = abs_curl[int(CLAMP(x + 1, 0, nx - 1)) + nx * int(CLAMP(y, 0, ny - 1))];
            float cB = abs_curl[int(CLAMP(x, 0, nx - 1)) + nx * int(CLAMP(y - 1, 0, ny - 1))];
            float cT = abs_curl[int(CLAMP(x, 0, nx - 1)) + nx * int(CLAMP(y + 1, 0, ny - 1))];
            direction.x = cT - cB;
            direction.y = cR - cL;
            float2 force = 0.5f * direction;
            force = force / (length(force) + 0.0001);
            force = force * curl(x, y) * vorticity;
            force = force * (-1.0f);


            //direction = vorticity / (length(direction) + 1e-5f) * direction;

            //if (x < nx / 2) direction *= 0.0f;
            new_velocity[x + y * nx] = old_velocity[x + y * nx] + force * dt;
            //new_velocity[x + y * nx] = old_velocity[x + y * nx] + dt * curl(x, y) * direction;
    }

    std::swap(old_velocity, new_velocity);
}

void add_density(int px, int py, int r = 10, float value = 0.5f) {
    for (int y = -r; y <= r; y++) for (int x = -r; x <= r; x++) {
        float d = sqrtf(x * x + y * y);
        float u = smoothstep(float(r), 0.0f, d);
        old_density[int(CLAMP(px+x, 0, nx - 1)) + nx * int(CLAMP(py+y, 0, ny - 1))] += u * value;
    }
}

float randf(float a, float b) {
    float u = rand() * (1.0f / RAND_MAX);
    return lerp(a, b, u);
}

float sign(float x) {
    return
        x > 0.0f ? +1.0f :
        x < 0.0f ? -1.0f :
        0.0f;
}

void fluid_simulation_step(float dt, bool isPressed) {
    //FOR_EACH_CELL{
    //    if (x > nx * 0.5f) continue;

    //    float r = 10.0f;
    //    old_velocity(x, y).x += randf(-r, +r);
    //    old_velocity(x, y).y += randf(-r, +r);
    //}

    advect_density(dt);
    advect_velocity(dt);

    if (isPressed)
    {
        FOR_EACH_CELL{
            
            float e = expf(-((x - lastMousePos.x) * (x - lastMousePos.x) + (y - lastMousePos.y) * (y - lastMousePos.y)) / 5);
            float2 uF = (lastMousePos - mousePos) * dt*5000 * e;


            old_velocity[x + y * nx] = old_velocity[x + y *nx] + uF;
        }
    }

    // fast movement is dampened
    FOR_EACH_CELL{
        old_velocity[x + y * nx] *= 0.99f;
    }

        // fade away
        FOR_EACH_CELL{
            old_density[x + y * nx] *= 0.9f;
    }


    vorticity_confinement(dt);

    diffuse_velocity(dt);


    project_velocity();


    diffuse_density(dt);



    // zero out stuff at bottom
    for (int y = 0; y <= 10; y++)
    {
        for (int x = 0; x < nx; x++)
        {
            old_density[x + y * nx] = 0.0f;
            old_velocity[x + y * nx] = float2{ 0.0f, 0.0f };
        }
    }
}


uint32_t swap_bytes(uint32_t x, int i, int j) {
    union {
        uint32_t x;
        uint8_t bytes[4];
    } u;
    u.x = x;
    std::swap(u.bytes[i], u.bytes[j]);
    return u.x;
}

uchar4 rgba32(uint32_t r, uint32_t g, uint32_t b, uint32_t a) {
    r = clamp(r, 0u, 255u);
    g = clamp(g, 0u, 255u);
    b = clamp(b, 0u, 255u);
    a = clamp(a, 0u, 255u);
    return make_uchar4(r, g, b, a);//(a << 24) | (b << 16) | (g << 8) | r;
}

uchar4 rgba(float r, float g, float b, float a) {
    return rgba32(r * 256, g * 256, b * 256, a * 256);
}

void on_frame(GLuint& texture, float dt, bool isPressed) {



    fluid_simulation_step(dt, isPressed);

    // density field to pixels
    FOR_EACH_CELL{
        float f = old_density[x + y * nx];
        f = log2f(f * 0.25f + 1.0f);
        float f3 = f * f * f;
        float r = 1.5f * f;
        float g = 1.5f * f3;
        float b = f3 * f3;
        pixels[x + y * nx] = rgba(r, g, b, 1.0);
    }

    // upload pixels to texture

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, nx, ny, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
}

void on_mouse_button(int x, int y) {
    y = h - 1 - y;
    lastMousePos = mousePos;
    mousePos = float2{ x * 1.0f * nx / w, y * 1.0f * ny / h };
    add_density(mousePos.x, mousePos.y, 5, 300.0f);
}
