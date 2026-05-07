
# wordle

A fast solver for wordle written in Python using numba.

For the maximum information policy it takes:

- less than 3 seconds to compute the optimal next word
- 30 minutes to compute the optimal starting word considering all solutions

Test conducted on MacBook Pro (13-inch, M1, 2020)

## Installation

From the repository root, create a virtual environment and install in editable mode (PowerShell):

```powershell
python -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Layout:

- `allowed_words/` — `words_answers.txt` and `words_guesses.txt`
- `experiments_csv/` — generated or committed experiment CSV outputs
- `src/wordle_solver_strategy/` — library (`import wordle_solver_strategy`), demos, and experiment scripts

Console entry points (after install) include `wordle-demo-solver`, `wordle-demo-game`, `wordle-bs-plays`, `wordle-bs-tiles`, and `wordle-exp-*` scripts for batch experiments (see `pyproject.toml` `[project.scripts]`).

## Interactive Solver

Run `wordle-demo-solver` (interactive stdin loop; same behavior as the former `demos/demo_solver.py`).

```shell
    Optimal starting guess: reast
    -----------------
    Enter guess,code: reast,10000
    Next guess: courd
    -----------------
    Enter guess,code: courd,20010
    Next guess: crick
    -----------------
    Enter guess,code: crick,22200
    Answer is crimp
```

## Simulate a game

Run `wordle-demo-game` for a short scripted demo (non-interactive), or from Python:

```python
    from wordle_solver_strategy import Game, MaxInfoAgent

    g = Game(word="crimp", verbose=True)
    agent = MaxInfoAgent(answers, guesses, mode="standard")
    final_guess, _ = agent.play(g)
    print(final_guess)
```

## Game Modes

This package supports both standard and hard mode.
In hard mode, any revealed hints must be used in subsequent guesses.

Use the `mode` parameter of the various Agent and Solver classes to change modes e.g.

```python
    agent = MaxInfoAgent(answers, guesses, mode="standard")
```

## BS Detector

Don't trust how fast someone solved wordle? Check their results by seeing all remaining words, ranked by common usage.

Run either of:

- `wordle-bs-plays`
- `wordle-bs-tiles`

## Performance and Optimal Starting Word

I consider an optimal start word to be the one that solves the most answers and takes the fewest plays on average.

The optimal starting word depends on the policy used and the game mode. The table below outlines the optimal starting word and performance
characteristics of each policy on the original wordle answer/guess list.

| Policy    | Mode     | Optimal Starting Word | Mean Guesses       | Failed Words                                                                                                                                                                                            |
|-----------|----------|-----------------------|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| MaxInfo   | Standard | 'reast'               | 3.600431965442765  | 'goner' 'hatch' 'jaunt' 'found' 'waste' 'taunt' 'catch' 'dilly' 'boxer'                                                                                                                                 |
|           | Hard     | 'reast'               | 3.639635732870772  |                                                                                                                                                                                                         |
| MaxSplits | Standard | 'trace'               | 3.6207343412527    |                                                                                                                                                                                                         |
|           | Hard     | 'salet'               | 3.6259211096662334 | 'pound' 'hatch' 'watch' 'found' 'cover' 'blank' 'boxer' 'wight'                                                                                                                                         |
| MaxPrune  | Standard | 'laten'               | 4.439469320066335  | 'sissy' 'awake' 'blush' ... 'judge' 'rower' 'shave'                                                                                                                                                     |
|           | Hard     | 'leant'               | 3.8034934497816595 | 'dolly' 'mover' 'piper' 'water' 'foist' 'bound' 'sense' 'viper' 'rarer' 'waver' 'wreak' 'flake' 'wound' 'baste' 'tight' 'biddy' 'happy' 'fleck' 'mossy' 'hound' 'blame' 'vaunt' 'match' 'catty' 'rower' |

## Second Guess Strategy Experiment

Generate a capped second-guess strategy for a specific first word. This script
groups answers by the first feedback code and assigns the best second guess per
group using an information-theoretic objective.

Examples:

```shell
wordle-exp-second-guess-strategy crane
wordle-exp-second-guess-strategy crane --max-num-second-word-in-strategy 8
```

Optional speed control:

```shell
wordle-exp-second-guess-strategy crane --max-num-second-word-in-strategy 8 --top-n-candidates 500
```

The output CSV is saved to:

`experiments_csv/second_guess_strategy_for_{first_word}_max_{k}.csv`

### Notes

Failed words are often due to "lookalikes". For example with the word `hatch` the solver will check `match`, `batch`, `patch` and `latch` first and ultimately fail.

## Policies

### Maximum Information

The policy plays the word from the guess list that maximises the expected information content revealed about the solution, until there
is one solution remaining. Played words are not repeated as they would not reveal new information.

Let the outcome from making a guess, i.e. the code received, be a discrete random variable $X$. This random variable has 243 possible outcomes (5 letters each with 3 feedback states i.e. $3^5$). The following are examples of outcomes:

- ⬜️⬜️⬜️⬜️⬜️ - no matches,
- 🟩⬜️⬜️⬜️⬜️ - only first letter matched,
- 🟩⬜️⬜️🟨⬜️ - exact match and a partial match.

For a guessed word, $G$, the probability of an outcome is given by a Categorical distribution i.e.

$$P(X | G) \sim Cat(k, p)$$

The expected information content in the outcomes of $P(X | G)$ is given by entropy, which is defined as

$$H(X|G) = -\sum_{i=1}^{243} P(X=x_i | G) \log(P(X=x_i | G))$$

where $P(X=x_i | G)$ is the probability of outcome $i$. The value of $P(X=x_i | G)$ is  the proportion of answers that fall in outcome $i$ when playing the guess word. Therefore,
$$H(X|G) = -\sum_{i=1}^{243} P(X=x_i | G) \log(P(X=x_i | G))$$

$$H(X | G)  = - \sum_{i=1}^{243} \frac{|S_i|}{|S|} \log \left (\frac{|S_i|}{|S|} \right )$$

The word that most evenly divides the answer pool into the 243 bins necessitates the greatest number of bins and thus has highest entropy.

### Maximum Splits

This policy plays the word from the guess list that results in the largest number of outcomes.

### Maximum Prune

This policy plays the word that reduces the remaining answers as much as possible.

## Appendix - Installation Notes

### numba on Apple M1

numba requires llvmlite, which in turn requires llvm version 11. The default installed version of llvm is likely more recent than version 11.

1. Install llvm version 11

    ```shell
    arch -arm64 brew install llvm@11
    ```

2. Install llvmlite by pointing to old llvm version

    ```shell
    LLVM_CONFIG="/opt/homebrew/Cellar/llvm@11/11.1.0_3/bin/llvm-config" arch -arm64 pip install llvmlite

    pip install numba
    ```

### scipy on Apple M1

```shell
    brew install openblas
    pip install --no-cache --no-use-pep517 pythran cython pybind11 gast
    OPENBLAS="$(brew --prefix openblas)" pip install --no-cache --no-binary :all: --no-use-pep517 scipy
```

## Appendix - Information Gain Equivalance

The Maximum Information policy is equivalent to the Information Gain decision policy used by tree learning algorithms such as ID3, C4.5 and CART.

A decision node consists of a guess word and the leaf nodes correspond to
each of the 243 possible outcomes that the answers can be placed into.

Under the information gain policy one must select the word which results in the greatest information gain, which is defined as:

$$IG(S,G) = H(S) - H(S|G)$$

where $H(S)$ is the entropy of set $S$, which is the set of all remaining answers and

$$H(S | G)  = \sum_{i=1}^{243} \frac{|S_i|}{|S|} * H( S_i )$$

is the conditional entropy due to the creation 243 splits, by playing word $G$, with each split containing a set of answers called $S_i$.

Expanding

$$H( S_i ) = - \sum_{j=1}^{|S_i|} P(j) \log(P(j))$$

where $P(j)$ is the probability of answer $j$ in $S_i$ and is equal to $\frac{1}{|S_i|}$ since we treat answers as categories and the answers are unique.

We can simplify as follows:

$$H( S_i ) = - \sum_{j=1}^{|S_i|} \frac{1}{|S_i|} \log(\frac{1}{|S_i|})$$

$$H( S_i ) = - \frac{1}{|S_i|} \sum_{j=1}^{|S_i|} \log(\frac{1}{|S_i|})$$

$$H( S_i ) = - \frac{1}{|S_i|} * |S_i| * \log(\frac{1}{|S_i|})$$

$$H( S_i ) = - \log(\frac{1}{|S_i|})$$

Therefore
$$H(S | G)  = - \sum_{i=1}^{243} \frac{|S_i|}{|S|} - \log(\frac{1}{|S_i|})$$

The equivalence can be seen through expanding the logs of both approaches and dropping constant terms. First the criteria from the wordle policy .

$$H(X|G) = -\sum_{i=1}^{243} P(X=x_i | G) \log(P(X=x_i | G))$$

$$H(X|G) = -\sum_{i=1}^{243}  \frac{|S_i|}{|S|} \log(\frac{|S_i|}{|S|})$$

$$H(X|G) = -\sum_{i=1}^{243}  \frac{|S_i|}{|S|} \left(\log(|S_i|) - \log(|S|) \right)$$

$$H(X|G) = -\sum_{i=1}^{243}  \frac{|S_i|}{|S|} \log(|S_i|)$$

then the decision tree case

$$H(S | G)  = - \sum_{i=1}^{243} \frac{|S_i|}{|S|} - \log(\frac{1}{|S_i|})$$

$$H(S | G)  = - \sum_{i=1}^{243} \frac{|S_i|}{|S|} - \left(\log(1) - \log(|S_i|) \right)$$

$$H(S | G)  = - \sum_{i=1}^{243} \frac{|S_i|}{|S|} \log(|S_i|)$$
