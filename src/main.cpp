#include <string.h>
#include <math.h>
#include <stdlib.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include "fluid_solver.h"

std::vector<uint8_t> gFrameBuffer;
SDL_Window* gSDLWindow;
SDL_Renderer* gSDLRenderer;
SDL_Texture* gSimTexture;
static int gDone;
const int WINDOW_WIDTH = 1024 / 2;
const int WINDOW_HEIGHT = 1024 / 2;
const int SIM_WIDTH = 128;
const int SIM_HEIGHT = 128;

FluidSolver2D solver = FluidSolver2D(SIM_WIDTH, SIM_HEIGHT, 1.0f / SIM_WIDTH);

bool SMOOTH_RENDERING = true;

SDL_FRect calDstRect(int outputWidth, int outputHeight, int simWidth, int simHeight) {
    const float scaleX = static_cast<float>(outputWidth) / simWidth;
    const float scaleY = static_cast<float>(outputHeight) / simHeight;
    const float scale = std::min(scaleX, scaleY);
    const float width  = simWidth  * scale;
    const float height = simHeight * scale;

    SDL_FRect dst_rect;
    dst_rect.x = (outputWidth  - width)  * 0.5f;
    dst_rect.y = (outputHeight - height) * 0.5f;
    dst_rect.w = width;
    dst_rect.h = height;
    return dst_rect;
}

bool update() {
    SDL_Event e;
    if (SDL_PollEvent(&e)) {
        if (e.type == SDL_EVENT_QUIT) {
            return false;
        }
        if (e.type == SDL_EVENT_KEY_UP && e.key.key == SDLK_ESCAPE) {
            return false;
        }
        if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_S) {
            SMOOTH_RENDERING = !SMOOTH_RENDERING;
            return true;
        }
    }
    return true;
}

void render(Uint64 aTicks) {
    // project data to buffer frame
    auto [nx, ny] = solver.getSize();
    for (int j = 0, c = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i, ++c) {
            const float value = solver.density(i, j);
            const uint8_t byteValue = toByte(normalize(value, 0.0f, 1.0f));
            gFrameBuffer[c * 4 + 0] = byteValue;
            gFrameBuffer[c * 4 + 1] = byteValue;
            gFrameBuffer[c * 4 + 2] = byteValue;
            gFrameBuffer[c * 4 + 3] = 255;
        }
    }

    if (!SDL_UpdateTexture(gSimTexture, nullptr, gFrameBuffer.data(), SIM_WIDTH * 4 * sizeof(uint8_t))) {
        SDL_Log("Failed to update texture: %s", SDL_GetError());
    }

    int out_width = 0, out_height = 0;
    SDL_GetCurrentRenderOutputSize(gSDLRenderer, &out_width, &out_height);
    SDL_FRect dst_rect = calDstRect(out_width, out_height, SIM_WIDTH, SIM_HEIGHT);
    SDL_SetRenderDrawColor(gSDLRenderer, 0, 0, 0, 255);
    SDL_RenderClear(gSDLRenderer);
    SDL_RenderTexture(gSDLRenderer, gSimTexture, nullptr, &dst_rect); // use all texture
    SDL_RenderPresent(gSDLRenderer); // Future need: change to SDL_LockTexture
}

void loop() {
    if (!update()) {
        gDone = 1;
    } else {
        render(SDL_GetTicks());
    }
}

int main(int argc, char** argv) {
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS)) {
        return -1;
    }

    gFrameBuffer = std::vector<uint8_t>(SIM_WIDTH * SIM_HEIGHT * 4); // ABGR8888
    gSDLWindow = SDL_CreateWindow("MiniFluid", WINDOW_WIDTH, WINDOW_HEIGHT, 0);
    gSDLRenderer = SDL_CreateRenderer(gSDLWindow, nullptr);
    gSimTexture = SDL_CreateTexture(gSDLRenderer, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STREAMING, SIM_WIDTH, SIM_HEIGHT);
    if (!gSDLWindow || !gSDLRenderer || !gSimTexture){
        SDL_Log("Initialization failed: %s", SDL_GetError());
    }
    if (!SDL_SetTextureScaleMode(
        gSimTexture, 
        SMOOTH_RENDERING ? SDL_SCALEMODE_LINEAR : SDL_SCALEMODE_NEAREST)) {
        SDL_Log("Failed to set scale mode: %s", SDL_GetError());
    }

    gDone = 0;
    while(!gDone) {
        loop();
    }

    SDL_DestroyTexture(gSimTexture);
    SDL_DestroyRenderer(gSDLRenderer);
    SDL_DestroyWindow(gSDLWindow);
    SDL_Quit();

    return 0;
}