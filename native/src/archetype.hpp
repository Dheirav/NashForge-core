// A scripted opponent inside the training game, for best-response solves.
//
// The solver plays against itself and converges toward an equilibrium, which
// is the right target against opponents who play the full game and leaves
// chips against the field's stations and nits (the calibrated panel of
// 22 September 2026 puts every set at 55 to 62 percent against them, an
// equilibrium's margin rather than an exploiter's). Here the opponent's
// action at every one of its nodes comes from a fixed policy instead of the
// node's strategy, so what the traversal learns is the best response to that
// policy: a strategy that bets thin for value into a caller and never bluffs
// it, because it has been told the caller calls. This mirrors
// chipzen/archetypes.py decision for decision, on the abstract game's state:
// equity against a random hand (the native sampler, 200 runouts), the pot's
// price, the raises so far this street, and the same parameter names, so a
// row calibrated in Python to a real bot is the row the solver trains against.
// A best response is exploitable in turn, so what comes out is played only
// behind a confident read, never as the default.
#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "equity.hpp"
#include "nolimit.hpp"

namespace pokerbot {

struct ArchetypeParams {
    double open_eq = 0.5, limp_eq = 0.5, threebet_eq = 0.6, fold_margin = 0.0, raise_eq = 0.6,
           raise_p = 0.5, bluff_p = 0.0, call_p = 0.5, defend_eq = 0.5, defend3_eq = 0.5, open_frac = 0.5;
};

class ScriptedOpponent {
public:
    ScriptedOpponent(ArchetypeParams params, int big_blind, int samples = 200)
        : p_(params), big_blind_(big_blind), samples_(samples) {}

    /// The abstract raise nearest to what the Python archetype would send.
    /// chipzen/archetypes.py raises to the arena's minimum raise plus
    /// `fraction` of the pot after the call, so its raise-by is the minimum
    /// raise more than the fraction says; read as a fraction of the pot after
    /// the call, that is `fraction + min_raise_by / pot_after_call`. (The
    /// first exploiter's loss on 22 September was blamed on this sizing at
    /// first; the cause was the card-blind opponent, see nolimit_game.hpp's
    /// deal. The sizing match is still right, and measured secondary.)
    int8_t raise_like_python(const State& s, int player, const std::vector<int8_t>& legal,
                             double fraction, bool allin) const {
        if (allin && has(legal, ALL_IN)) return ALL_IN;
        const int opp = 1 - player;
        const int to_call = std::max(s.committed[static_cast<size_t>(opp)] - s.committed[static_cast<size_t>(player)], 0);
        const int pot_after_call = s.contributions[0] + s.contributions[1] + to_call;
        // The arena's minimum raise is the last raise's size, which when facing
        // a bet is what there is to call (the bettor raised from our level to
        // theirs), and a big blind otherwise.
        const int min_raise_by = std::max(to_call, big_blind_);
        const double effective = pot_after_call > 0 ? fraction + static_cast<double>(min_raise_by) / pot_after_call : fraction;
        return raise_to(legal, effective, false);
    }

    /// The action this opponent takes as `player` in `s`, among `legal`.
    int8_t act(const State& s, int player, const std::vector<int8_t>& legal, Rng& rng) const {
        const int opp = 1 - player;
        const int to_call = std::max(s.committed[static_cast<size_t>(opp)] - s.committed[static_cast<size_t>(player)], 0);
        const int pot = s.contributions[0] + s.contributions[1];
        const bool facing = to_call > 0;
        const double price = facing ? static_cast<double>(to_call) / (pot + to_call) : 0.0;
        const bool preflop = s.board_n == 0;
        const bool can_raise = std::any_of(legal.begin(), legal.end(), [](int8_t a) { return a >= RAISE_HALF; });
        const int raises = raises_this_street(s.history);
        const double u = unit(rng);
        int cards[2] = {s.hole[static_cast<size_t>(player)][0], s.hole[static_cast<size_t>(player)][1]};
        int board[5];
        for (int i = 0; i < s.board_n; ++i) board[i] = s.board[static_cast<size_t>(i)];
        const double e = equity(cards, board, s.board_n, rng);
        const bool is_short = s.stacks[static_cast<size_t>(player)] + to_call <= 12 * big_blind_;

        if (preflop) {
            if (raises == 0) {
                if (can_raise && e >= p_.open_eq) return raise_like_python(s, player, legal, p_.open_frac, false);
                if (e >= p_.limp_eq || !facing) return passive(legal);
                return fold(legal);
            }
            if (can_raise && e >= p_.threebet_eq && u > p_.call_p * 0.5)
                return raise_like_python(s, player, legal, 1.0, is_short || raises >= 2);
            const double defend = raises >= 2 ? p_.defend3_eq : p_.defend_eq;
            if (e >= defend && e >= price + p_.fold_margin) return passive(legal);
            return fold(legal);
        }
        if (facing) {
            if (e < price + p_.fold_margin) return fold(legal);
            if (can_raise && e >= p_.raise_eq && u > p_.call_p) return raise_like_python(s, player, legal, 1.0, is_short);
            return passive(legal);
        }
        if (can_raise && ((e >= p_.raise_eq && u < p_.raise_p) || u < p_.bluff_p)) return raise_like_python(s, player, legal, 0.66, false);
        return passive(legal);
    }

private:
    static double unit(Rng& rng) { return (rng.next() >> 11) * (1.0 / 9007199254740992.0); }

    /// Equity against a random hand, memoised on the cards. The deal is fixed
    /// within an iteration and the policy is asked at every node the opponent
    /// reaches, preflop a dozen times for the same two cards; without this the
    /// solve ran at 1.9 ms an iteration against 0.2 for self-play. Thread-local
    /// so the worker threads do not share it, eight entries because a deal has
    /// at most four boards per seat.
    double equity(const int* cards, const int* board, int board_n, Rng& rng) const {
        struct Entry { uint64_t key; double value; };
        thread_local Entry cache[8] = {};
        thread_local int next = 0;
        uint64_t key = (static_cast<uint64_t>(cards[0]) << 6) | static_cast<uint64_t>(cards[1]);
        for (int i = 0; i < board_n; ++i) key = (key << 6) | static_cast<uint64_t>(board[i]);
        key = (key << 3) | static_cast<uint64_t>(board_n);
        key |= 1ULL << 63;                           // never the zero of an empty slot
        for (const Entry& e : cache) if (e.key == key) return e.value;
        const double value = equity_vs_random(cards, 2, board, board_n, samples_, rng.next());
        cache[next] = {key, value};
        next = (next + 1) & 7;
        return value;
    }

    static int raises_this_street(const std::string& history) {
        const size_t slash = history.rfind('/');
        int n = 0;
        for (size_t i = slash == std::string::npos ? 0 : slash + 1; i < history.size(); ++i)
            if (history[i] >= '2') ++n;
        return n;
    }

    static bool has(const std::vector<int8_t>& legal, int8_t a) {
        return std::find(legal.begin(), legal.end(), a) != legal.end();
    }
    static int8_t passive(const std::vector<int8_t>& legal) { return has(legal, CHECK_CALL) ? CHECK_CALL : legal.front(); }
    static int8_t fold(const std::vector<int8_t>& legal) { return has(legal, FOLD) ? FOLD : passive(legal); }

    /// The legal sized raise nearest to `fraction` of the pot, or the shove.
    static int8_t raise_to(const std::vector<int8_t>& legal, double fraction, bool allin) {
        if (allin && has(legal, ALL_IN)) return ALL_IN;
        int8_t best = -1;
        double gap = 1e9;
        for (int8_t a : legal) {
            if (a < RAISE_HALF || a == ALL_IN) continue;
            const double d = std::fabs(raise_fraction(a) - fraction);
            if (d < gap) { gap = d; best = a; }
        }
        if (best >= 0) return best;
        if (has(legal, ALL_IN)) return ALL_IN;
        return passive(legal);
    }

    ArchetypeParams p_;
    int big_blind_;
    int samples_;
};

}  // namespace pokerbot
