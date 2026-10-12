// The rollout without any binding overhead, to separate the language from the FFI.
#include <cstdio>
#include <chrono>
#include "equity.hpp"
int main() {
    const int hole[2] = {12, 25};
    const int board[3] = {3, 17, 40};
    const int M = 300000;
    double sink = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < M; ++i)
        sink += pokerbot::equity_vs_random(hole, 2, board, 3, 40, (uint64_t)i);
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("  40-sample rollout, pure C++       %6.2f us   (sink %.1f)\n", dt / M * 1e6, sink);
    return 0;
}
