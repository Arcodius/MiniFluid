#include <algorithm>
#include <cmath>
#include <math.h>

float normalize(float value, float minValue, float maxValue) {
    if (maxValue <= minValue)
        return 0.0f;

    return std::clamp(
        (value - minValue) / (maxValue - minValue),
        0.0f,
        1.0f
    );
}

// Convert a float to 8-bit color
uint8_t toByte(float x) {
    return static_cast<uint8_t>(
        std::lround(std::clamp(x, 0.0f, 1.0f) * 255.0f)
    );
}

// Convert pixel coordinate to simulation coordinate
float pix2sim(int pix, int display_res, int sim_res) {
    return (pix + 0.5f) / display_res * sim_res - 0.5f;
}