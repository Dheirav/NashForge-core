// Kuhn poker: three cards, one betting round, and an analytically known value.
//
// Here only as an acceptance test for MCCFR. The game value to the first player
// under optimal play is -1/18, so a solver that converges to it has its regret
// matching, external sampling and strategy averaging all correct. Comparing a
// port against our own Python would only show the translation was faithful;
// this shows the algorithm is right.
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "equity.hpp"   // Rng

namespace pokerbot {

class KuhnPoker {
public:
    static constexpr int num_players = 2;

    struct State {
        int8_t cards[2] = {-1, -1};
        std::string history;
        bool dealt = false;
    };

    State initial_state() const { return State{}; }

    bool is_chance(const State& s) const { return !s.dealt; }
    bool is_final_street(const State&) const { return false; }   // one street; pruning may apply

    bool is_terminal(const State& s) const {
        const std::string& h = s.history;
        return h == "pp" || h == "bb" || h == "bp" || h == "pbb" || h == "pbp";
    }

    State sample_chance(const State& s, Rng& rng) const {
        State next = s;
        // Two distinct cards from three.
        int8_t deck[3] = {0, 1, 2};
        for (int i = 0; i < 2; ++i) {
            const int j = i + static_cast<int>(rng.below(static_cast<uint32_t>(3 - i)));
            const int8_t t = deck[i]; deck[i] = deck[j]; deck[j] = t;
        }
        next.cards[0] = deck[0];
        next.cards[1] = deck[1];
        next.dealt = true;
        return next;
    }

    int current_player(const State& s) const {
        return static_cast<int>(s.history.size()) % 2;
    }

    std::vector<int8_t> legal_actions(const State&) const {
        return {0, 1};                      // 0 = pass/check/fold, 1 = bet/call
    }

    State next_state(const State& s, int8_t action) const {
        State next = s;
        next.history.push_back(action == 0 ? 'p' : 'b');
        return next;
    }

    double utility(const State& s, int player) const {
        const std::string& h = s.history;
        const int opponent = 1 - player;
        const bool wins = s.cards[player] > s.cards[opponent];

        if (h == "bp") return player == 0 ? 1.0 : -1.0;     // p1 folded to a bet
        if (h == "pbp") return player == 1 ? 1.0 : -1.0;    // p0 folded
        if (h == "pp") return wins ? 1.0 : -1.0;            // showdown for the antes
        return wins ? 2.0 : -2.0;                           // "bb" or "pbb"
    }

    using Key = std::string;
    void begin_iteration(Rng&) {}          // one card each; nothing to share across branches
    static std::string key_to_string(const Key& key) { return key; }
    static Key key_from_string(const std::string& key) { return key; }

    std::string information_set(const State& s, int player) const {
        return std::to_string(static_cast<int>(s.cards[player])) + "|" + s.history;
    }
};

}  // namespace pokerbot
