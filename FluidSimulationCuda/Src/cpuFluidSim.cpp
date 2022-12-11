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

template <typename T>
struct Grid {
    T* values;
    int nx, ny;

    Grid() {}
    Grid(int nx, int ny) : nx(nx), ny(ny) {
        values = new T[nx * ny];
    }

    ~Grid() {
        delete[] values;
    }

    void swap(Grid& other) {
        std::swap(values, other.values);
        std::swap(nx, other.nx);
        std::swap(ny, other.ny);
    }

    const T* data() const {
        return values;
    }

    int idx(int x, int y) const {


        //x = clamp(x, 0, nx - 1);
        //y = clamp(y, 0, ny - 1);

        // wrap around
        x = (x + nx) % nx;
        y = (y + ny) % ny;

        return x + y * nx;
    }

    T& operator () (int x, int y) {
        return values[idx(x, y)];
    }

    const T& operator () (int x, int y) const {
        return values[idx(x, y)];
    }
};

int w = 512;
int h = 512;

int nx = 256;
int ny = 256;

float dt = 0.02f;
int iterations = 5;
float vorticity = 10.0f;

float2 mouse;

float2* old_velocity;
float2* new_velocity;

Grid<float> old_density;
Grid<float> new_density;

uchar4* pixels;

Grid<float> old_pressure;
Grid<float> new_pressure;
Grid<float> divergence;

Grid<float> abs_curl;

#define FOR_EACH_CELL for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++)

template <typename T>
void initGrid(Grid<T>& grid, int nx, int ny)
{
    grid.nx = nx;
    grid.ny = ny;
    grid.values = new T[nx * ny];
}

void init(int width, int height, int scale) {


    w = width;
    h = height;

    nx = width / scale;
    ny = height / scale;

    //initGrid(old_velocity, nx, ny);
    //initGrid(new_velocity, nx, ny);
    old_velocity = new float2[nx * ny];
    new_velocity = new float2[nx * ny];


    initGrid(old_density, nx, ny);
    initGrid(new_density, nx, ny);
    

    pixels = new uchar4[nx * ny];
    //initGrid(pixels, nx, ny);

    initGrid(old_pressure, nx, ny);
    initGrid(new_pressure, nx, ny);
    initGrid(divergence, nx, ny);

    initGrid(abs_curl, nx, ny);

    FOR_EACH_CELL{
    old_density(x, y) = 0.0f;
    old_velocity[x + y * nx] = float2{0.0f, 0.0f};
    }
}

template <typename T>
T interpolate(const Grid<T>& grid, float2 p) {
    int ix = floorf(p.x);
    int iy = floorf(p.y);
    float ux = p.x - ix;
    float uy = p.y - iy;
    return lerp(
        lerp(grid(ix + 0, iy + 0), grid(ix + 1, iy + 0), ux),
        lerp(grid(ix + 0, iy + 1), grid(ix + 1, iy + 1), ux),
        uy
    );
}


float2 interpolate(float2* grid, float2 p) {
    int ix = floorf(p.x);
    int iy = floorf(p.y);
    float ux = p.x - ix;
    float uy = p.y - iy;
    return lerp(
        lerp(grid[int(CLAMP(ix, 0, nx-1)) + nx * int(CLAMP(iy, 0, ny-1))], grid[int(CLAMP(ix+1, 0, nx-1)) + nx * int(CLAMP(iy, 0, ny-1))], ux),
        lerp(grid[int(CLAMP(ix, 0, nx-1)) + nx * int(CLAMP(iy+1, 0, ny-1))], grid[int(CLAMP(ix+1, 0, nx-1)) + nx * int(CLAMP(iy+1, 0, ny-1))], ux),
        uy
    );
}

void advect_density() {
    FOR_EACH_CELL{
        float2 pos = make_float2(x, y) - dt * old_velocity[x + y * nx];
        new_density(x, y) = interpolate(old_density, pos);
    }
    old_density.swap(new_density);
}

void advect_velocity() {
    FOR_EACH_CELL{
        float2 pos = make_float2(x, y) - dt * old_velocity[x + y * nx];
        new_velocity[x + y * nx] = interpolate(old_velocity, pos);
    }
    std::swap(old_velocity, new_velocity);
}

void diffuse_density() {
    float diffusion = dt * 100.01f;
    FOR_EACH_CELL{
        float sum =
            diffusion * (
            +old_density(x - 1, y + 0)
            + old_density(x + 1, y + 0)
            + old_density(x + 0, y - 1)
            + old_density(x + 0, y + 1)
            )
            + old_density(x + 0, y + 0);
        new_density(x, y) = 1.0f / (1.0f + 4.0f * diffusion) * sum;
    }
    old_density.swap(new_density);
}

void diffuse_velocity() {
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

    memset(old_pressure.values, 0, nx * ny * sizeof(float));

    FOR_EACH_CELL{
        float dx = old_velocity[int(CLAMP(x+1, 0, nx-1)) + nx * int(CLAMP(y, 0, ny-1))].x - old_velocity[int(CLAMP(x - 1, 0, nx-1)) + nx * int(CLAMP(y, 0, ny-1))].x;
        float dy = old_velocity[int(CLAMP(x, 0, nx-1)) + nx * int(CLAMP(y+1, 0, ny-1))].y - old_velocity[int(CLAMP(x, 0, nx-1)) + nx * int(CLAMP(y-1, 0, ny-1))].y;
        divergence(x, y) = dx + dy;
        old_pressure(x, y) = 0.0f;
    }

    for (int k = 0; k < iterations; k++) {
        FOR_EACH_CELL{
            float sum = -divergence(x, y)
                + old_pressure(x + 1, y + 0)
                + old_pressure(x - 1, y + 0)
                + old_pressure(x + 0, y + 1)
                + old_pressure(x + 0, y - 1);
            new_pressure(x, y) = 0.25f * sum;
        }
        old_pressure.swap(new_pressure);
    }

    FOR_EACH_CELL{
        old_velocity[x + y * nx].x -= 0.5f * (old_pressure(x + 1, y + 0) - old_pressure(x - 1, y + 0));
        old_velocity[x + y * nx].y -= 0.5f * (old_pressure(x + 0, y + 1) - old_pressure(x + 0, y - 1));
    }
}

float curl(int x, int y) {
    return
        old_velocity[int(CLAMP(x, 0, nx-1)) + nx * int(CLAMP(y + 1, 0, ny-1))].x - old_velocity[int(CLAMP(x, 0, nx-1)) + nx * int(CLAMP(y - 1, 0, ny-1))].x +
        old_velocity[int(CLAMP(x-1, 0, nx-1)) + nx * int(CLAMP(y, 0, ny-1))].y - old_velocity[int(CLAMP(x+1, 0, nx-1)) + nx * int(CLAMP(y, 0, ny-1))].y;
}

void vorticity_confinement() {
    FOR_EACH_CELL{
        abs_curl(x, y) = fabsf(curl(x, y));
    }

        FOR_EACH_CELL{
            float2 direction;
            direction.x = abs_curl(x + 0, y - 1) - abs_curl(x + 0, y + 1);
            direction.y = abs_curl(x + 1, y + 0) - abs_curl(x - 1, y + 0);

            direction = vorticity / (length(direction) + 1e-5f) * direction;

            if (x < nx / 2) direction *= 0.0f;

            new_velocity[x + y * nx] = old_velocity[x + y * nx] + dt * curl(x, y) * direction;
    }

    std::swap(old_velocity, new_velocity);
}

void add_density(int px, int py, int r = 10, float value = 0.5f) {
    for (int y = -r; y <= r; y++) for (int x = -r; x <= r; x++) {
        float d = sqrtf(x * x + y * y);
        float u = smoothstep(float(r), 0.0f, d);
        old_density(px + x, py + y) += u * value;
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

void fluid_simulation_step() {
    //FOR_EACH_CELL{
    //    if (x > nx * 0.5f) continue;

    //    float r = 10.0f;
    //    old_velocity(x, y).x += randf(-r, +r);
    //    old_velocity(x, y).y += randf(-r, +r);
    //}

        // dense regions rise up
    FOR_EACH_CELL{
            old_velocity[x + y * nx].y += (old_density(x, y) * 20.0f - 5.0f) * dt;
    }

    add_density(mouse.x, mouse.y, 10, 0.5f);

    //// fast movement is dampened
    //FOR_EACH_CELL{
    //    old_velocity(x, y) *= 0.999f;
    //}

        // fade away
        FOR_EACH_CELL{
            old_density(x, y) *= 0.85f;
    }


    vorticity_confinement();

    //diffuse_velocity();
    advect_velocity();

    project_velocity();


    //diffuse_density();
    advect_density();


    // zero out stuff at bottom
    FOR_EACH_CELL{
        if (y < 10) {
            old_density(x, y) = 0.0f;
            old_velocity[x + y * nx] = float2{0.0f, 0.0f};
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

void on_frame(GLuint& texture) {



    fluid_simulation_step();

    // density field to pixels
    FOR_EACH_CELL{
        float f = old_density(x, y);
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
    mouse = float2{ x * 1.0f * nx / w, y * 1.0f * ny / h };
    add_density(mouse.x, mouse.y, 10, 300.0f);
}
