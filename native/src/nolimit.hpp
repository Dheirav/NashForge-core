// Heads-up no-limit Hold'em betting, ported from games/nolimit.py.
//
// This is the file that most needs to be right. Betting is implemented twice in
// this project and has diverged twice: `training/fitness.py` once sized a
// pot-fraction raise off the pot BEFORE the call while the traversal game sized
// it after, making every engine raise about 20% too small; and the traversal
// game once failed to end a hand at an all-in, so it kept asking check/call from
// players holding nothing and keyed 19.7% of decision nodes differently from the
// engine. Both produced plausible wrong numbers rather than errors.
//
// `tests/test_native.py` drives this and the Python over the enumerated betting
// tree and compares chips at every node, which is what
// `tests/test_betting_equivalence.py` already does for the other pair.
#pragma once
#include <cstdint>
#include <array>
#include <cmath>
#include <string>
#include <string_view>
#include <vector>

namespace pokerbot {

enum Action : int8_t { FOLD = 0, CHECK_CALL = 1, RAISE_HALF = 2,
                       RAISE_POT = 3, RAISE_TWO = 4, ALL_IN = 5 };

constexpr int CHANCE = -1;
constexpr int NUM_STREETS = 4;
constexpr std::array<int, 4> STREET_BOARD_SIZE = {0, 3, 4, 5};
//: Raise actions from smallest to largest. A taper drops the smallest first.
constexpr std::array<int8_t, 4> RAISE_ACTIONS = {RAISE_HALF, RAISE_POT, RAISE_TWO, ALL_IN};

inline double raise_fraction(int action) {
    switch (action) {
        case RAISE_HALF: return 0.5;
        case RAISE_POT:  return 1.0;
        case RAISE_TWO:  return 2.0;
        default:         return 0.0;
    }
}

/// How many raise sizes are available at each raise depth.
///
/// An int N in Python means the uniform schedule; here that is a vector of N
/// fours. `{4, 1}` is four sizes for the opening bet and only all-in for the
/// raise, which is what makes a deeper betting tree affordable.
struct RaiseSchedule {
    std::vector<int> sizes;
    /// The actions themselves, one list per raise depth, when the schedule
    /// names its sizes instead of counting them from the largest end. Empty
    /// means counted, which is what every solve before 23 September used.
    /// Keeping the largest k is a choice: at a two-wide level it leaves
    /// "2x pot or all-in", and the pot-sized re-raise that is the ordinary
    /// three-bet cannot be made at all. Naming costs nothing, since the tree
    /// grows with how many sizes a level holds and not with which.
    std::vector<std::vector<int8_t>> chosen;

    static RaiseSchedule uniform(int depth) {
        RaiseSchedule s;
        s.sizes.assign(static_cast<size_t>(depth), 4);
        return s;
    }
    int depth() const {
        return static_cast<int>(chosen.empty() ? sizes.size() : chosen.size());
    }
    /// Raise actions legal at `at_depth`: as named, or largest kept when tapering.
    std::vector<int8_t> at(int at_depth) const {
        if (at_depth >= depth()) return {};
        if (!chosen.empty()) return chosen[static_cast<size_t>(at_depth)];
        int keep = sizes[static_cast<size_t>(at_depth)];
        if (keep < 0) keep = 0;
        if (keep > 4) keep = 4;
        return std::vector<int8_t>(RAISE_ACTIONS.begin() + (4 - keep), RAISE_ACTIONS.end());
    }
};

struct State {
    std::array<std::array<int8_t, 2>, 2> hole{};
    std::array<int8_t, 5> board{};
    int8_t board_n = 0;
    bool dealt = false;
    /// Action digits with '/' street separators. This IS the information-set
    /// key's public half, so it must read exactly as the Python's does.
    std::string history;
    std::array<int32_t, 2> contributions{};
    std::array<int32_t, 2> committed{};
    std::array<int32_t, 2> stacks{};
    int8_t street = 0;
};

class NoLimitHoldem {
public:
    NoLimitHoldem(int starting_stack, int small_blind, int big_blind,
                  RaiseSchedule schedule)
        : starting_stack_(starting_stack), small_blind_(small_blind),
          big_blind_(big_blind), schedule_(std::move(schedule)) {}

    State initial_state() const {
        State s;
        s.contributions = {small_blind_, big_blind_};
        s.committed = {small_blind_, big_blind_};
        s.stacks = {starting_stack_ - small_blind_, starting_stack_ - big_blind_};
        return s;
    }

    /// A view of this street's action digits, not a copy: measured at 4,052
    /// calls per iteration, the copy was a fifth to a third of the iteration.
    static std::string_view street_actions(const std::string& history) {
        const size_t slash = history.rfind('/');
        const std::string_view all(history);
        return slash == std::string::npos ? all : all.substr(slash + 1);
    }

    bool all_in(const State& s) const {
        return std::min(s.stacks[0], s.stacks[1]) == 0
            && s.committed[0] == s.committed[1];
    }

    bool street_closed(const State& s) const {
        const std::string_view acts = street_actions(s.history);
        if (acts.size() < 2 || acts.back() != char('0' + CHECK_CALL)) return false;
        if (s.committed[0] == s.committed[1]) return true;
        // A call that could not cover still closes: the caller is all-in and
        // the uncalled excess is returned at showdown.
        return std::min(s.stacks[0], s.stacks[1]) == 0;
    }

    bool is_terminal(const State& s) const {
        if (!s.dealt) return false;
        if (s.history.find(char('0' + FOLD)) != std::string::npos) return true;
        if (s.street >= NUM_STREETS) return true;
        if (all_in(s) && s.board_n == STREET_BOARD_SIZE[3]) return true;
        return false;
    }

    int current_player(const State& s) const {
        if (!s.dealt) return CHANCE;
        if (all_in(s)) return CHANCE;           // run the board out; nobody acts
        if (street_closed(s)) return CHANCE;    // deal the next street
        const std::string_view acts = street_actions(s.history);
        const int first = s.street == 0 ? 0 : 1;   // preflop the small blind acts first
        return (first + static_cast<int>(acts.size())) % 2;
    }

    std::vector<int8_t> legal_actions(const State& s) const {
        const std::string_view acts = street_actions(s.history);
        int raises = 0;
        for (char c : acts) {
            const int a = c - '0';
            if (a == RAISE_HALF || a == RAISE_POT || a == RAISE_TWO || a == ALL_IN) ++raises;
        }
        const bool facing = s.committed[0] != s.committed[1];
        const int last = acts.empty() ? -2 : acts.back() - '0';

        std::vector<int8_t> available;
        if (last == ALL_IN) {
            available = {FOLD, CHECK_CALL};     // nothing left to raise with
        } else {
            if (facing) available.push_back(FOLD);
            available.push_back(CHECK_CALL);
            for (int8_t a : schedule_for(s.street).at(raises)) available.push_back(a);
        }

        const int player = current_player(s);
        if (player < 0) return available;
        const int to_call = std::abs(s.committed[0] - s.committed[1]);
        if (s.stacks[static_cast<size_t>(player)] <= to_call) {
            std::vector<int8_t> capped;
            for (int8_t a : available)
                if (a == FOLD || a == CHECK_CALL) capped.push_back(a);
            return capped;
        }
        return available;
    }

    int cost(const State& s, int player, int action) const {
        const int opponent = 1 - player;
        const int to_call = s.committed[static_cast<size_t>(opponent)]
                          - s.committed[static_cast<size_t>(player)];
        if (action == FOLD) return 0;
        if (action == CHECK_CALL)
            return std::min(std::max(to_call, 0), s.stacks[static_cast<size_t>(player)]);
        if (action == ALL_IN) return s.stacks[static_cast<size_t>(player)];
        return raise_cost(s, player, raise_fraction(action));
    }

    int raise_cost(const State& s, int player, double fraction) const {
        const int opponent = 1 - player;
        const int to_call = s.committed[static_cast<size_t>(opponent)]
                          - s.committed[static_cast<size_t>(player)];
        const int pot = s.contributions[0] + s.contributions[1] + to_call;
        // nearbyint, not round: Python's round() is banker's rounding, so
        // round(2.5) is 2. std::round would give 3 and every half-pot raise into
        // an odd pot would differ by a chip, silently.
        const int raise_by = static_cast<int>(std::nearbyint(fraction * pot));
        return std::min(std::max(to_call, 0) + std::max(raise_by, big_blind_),
                        s.stacks[static_cast<size_t>(player)]);
    }

    State apply(const State& s, int action) const {
        const int player = current_player(s);
        return commit(s, player, cost(s, player, action),
                      static_cast<char>('0' + action));
    }

    State commit(const State& s, int player, int pay, char recorded) const {
        State next = s;
        next.contributions[static_cast<size_t>(player)] += pay;
        next.committed[static_cast<size_t>(player)] += pay;
        next.stacks[static_cast<size_t>(player)] -= pay;
        next.history.push_back(recorded);
        if (next.street == NUM_STREETS - 1 && street_closed(next))
            next.street = NUM_STREETS;
        return next;
    }

    /// Reveal the next street: board grows, per-street commitments reset.
    State advance_street(const State& s, const int8_t* board, int board_n) const {
        State next = s;
        for (int i = 0; i < board_n; ++i) next.board[static_cast<size_t>(i)] = board[i];
        next.board_n = static_cast<int8_t>(board_n);
        next.history.push_back('/');
        next.committed = {0, 0};
        next.street = static_cast<int8_t>(s.street + 1);
        return next;
    }

    int starting_stack() const { return starting_stack_; }

    /// One schedule per street, preflop to river, used in place of the single
    /// schedule when set. LBR's traced bets can put a large share of a 70bb rung's
    /// leak in two preflop sizes the tree lacks (an overbet, and a small
    /// re-raise where the taper offers only 2x and all-in); adding a size to
    /// every street multiplies the tree by 4 to 12, adding it preflop only by
    /// 1.4 to 1.9. The history key is unchanged: the action codes are the same,
    /// only which are legal differs by street.
    void set_street_schedules(std::vector<RaiseSchedule> per_street) {
        if (!per_street.empty() && per_street.size() != static_cast<size_t>(NUM_STREETS))
            throw std::invalid_argument("street schedules: one per street, preflop to river");
        street_schedules_ = std::move(per_street);
    }
    const RaiseSchedule& schedule_for(int street) const {
        return street_schedules_.empty() ? schedule_ : street_schedules_[static_cast<size_t>(street)];
    }

private:
    int starting_stack_;
    int small_blind_;
    int big_blind_;
    RaiseSchedule schedule_;
    std::vector<RaiseSchedule> street_schedules_;
};

}  // namespace pokerbot
