# Push-or-fold check: results/cfr/ladder169l_short

Small blind shoves or folds, big blind calls or folds, 169 hands a side, solved by fictitious play on Monte Carlo equities. The solver's raises of any size count as shoves here.

## 12bb rung: cap2_12bb.pkl

| | equilibrium | solver |
|---|---|---|
| small blind shoves (share of hands) | 61% | 40% (all-in only 9%, limps 43%) |
| big blind calls a shove | 36% | 40% |
| hands where the two agree, weighted | shove 74%, call 94% | |
| small blind value vs a best-responding big blind (bb/hand) | -0.022 (equilibrium -0.020) | -0.233 |

Solver folds where the equilibrium shoves (41): 53s 55 64s 66 74s 75s 77 86s 87o 95s 98s 98o 99 T5s T6s T8o T9s T9o TT J8o J9o JTs JTo Q2s Q4s Q8s QJo QQ K4s K6s K7o K9s KTs KK A3s A4s A9s A9o AKs AKo
Solver shoves where the equilibrium folds (3): 62s 84s Q7o
Solver folds to a shove where the equilibrium calls (3): K4s K5s K7o
Solver calls a shove where the equilibrium folds (7): 98s T8s T9s J7s J8s Q6s K3s

