#pragma once

#include <math.h>
struct vec2f {
    float x, y;

    vec2f& operator *= (float b){ x *= b  ; y *= b  ; return *this; }
    vec2f& operator += (vec2f b){ x += b.x; y += b.y; return *this; }
    vec2f& operator -= (vec2f b){ x -= b.x; y -= b.y; return *this; }
};

vec2f operator + (vec2f a, vec2f b){ return vec2f{a.x + b.x, a.y + b.y}; }
vec2f operator - (vec2f a, vec2f b){ return vec2f{a.x - b.x, a.y - b.y}; }
vec2f operator + (         vec2f b){ return vec2f{    + b.x,     + b.y}; }
vec2f operator - (         vec2f b){ return vec2f{    - b.x,     - b.y}; }
vec2f operator * (float a, vec2f b){ return vec2f{a   * b.x, a   * b.y}; }
vec2f operator * (vec2f a, float b){ return vec2f{a.x * b  , a.y * b  }; }
float dot        (vec2f a, vec2f b){ return float(a.x * b.x+ a.y * b.y); }

struct vec3f {
    float x, y, z;

    vec3f& operator *= (float b) { x *= b; y *= b; z *= b; return *this; }
    vec3f& operator += (vec3f b) { x += b.x; y += b.y; z += b.z; return *this; }
    vec3f& operator -= (vec3f b) { x -= b.x; y -= b.y; z -= b.z; return *this; }
};

vec3f operator + (vec3f a, vec3f b) { return vec3f{ a.x + b.x, a.y + b.y, a.z + b.z }; }
vec3f operator - (vec3f a, vec3f b) { return vec3f{ a.x - b.x, a.y - b.y, a.z - b.z }; }
vec3f operator + (vec3f b) { return vec3f{ +b.x,     +b.y,      +b.z }; }
vec3f operator - (vec3f b) { return vec3f{ -b.x,     -b.y,      -b.z }; }
vec3f operator * (float a, vec3f b) { return vec3f{ a * b.x, a * b.y, a * b.z }; }
vec3f operator * (vec3f a, float b) { return b * a; }
float dot(vec3f a, vec3f b) { return float(a.x * b.x + a.y * b.y + a.z * b.z); }


vec2f polar(float radians){
    return vec2f{cosf(radians), sinf(radians)};
}

float length(vec2f a){
    return sqrtf(dot(a, a));
}

vec2f normalize(vec2f a){
    return 1.0f/sqrtf(dot(a, a)) * a;
}

vec2f normalize(vec2f a, float eps){
    return 1.0f/(eps + sqrtf(dot(a, a))) * a;
}

vec2f v2f(float x, float y){
    return vec2f{x, y};
}

template <typename T, typename U>
T lerp(const T &a, const T &b, const U &u){
    return (U(1) - u)*a + u*b;
}

template <typename T>
const T& clamp(const T &x, const T &a, const T &b){
    return fminf(b, fmaxf(x, a));
}

template <typename T, typename U>
T smoothstep(const T &a, const T &b, const U &u){
    T t = clamp((u - a)/(b - a), U(0), U(1));
    return t*t*(U(3) - U(2)*t);
}
