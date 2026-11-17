"""Run edge-case sentences through app.analyze() (no Gradio launch) and print one line each.

Columns: id | DistilBERT label+conf | ANFIS label | agree? | deciding rule | raw firing |
         low-coverage? | pos/neg/negation values | [TRUNC if >128 tokens] [NOEVID if pos=neg=0] "snippet"
Usage: python test_cases.py
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app import analyze  # noqa: E402  (loads DistilBERT + ANFIS; demo.launch() is guarded)

CASES = [
    "the film was not bad at all",
    "i don't think this is a good movie",
    "it is never boring",
    "the acting was great but the plot was terrible",
    "the plot was terrible but the acting was great",
    "oh great, another two hours of my life wasted",
    "yeah, sure, best movie ever",
    "absolutely fantastic",
    "awful",
    "the movie is two hours long",
    "i watched it on tuesday with my brother",
    "not good, not bad, just okay",
    "it's not that it wasn't funny",
    "WORST MOVIE EVER!!!",
    "amazing 😍😍",
    "well-made and well-acted",
    "the cinematography was breathtaking, the score was haunting, and the performances "
    "were superb throughout",
    "",
    "a",
    "this movie is a masterpiece of boredom",
    "la película fue excelente",
    "the film is bad, really bad, so bad that it is good",
    "not only funny but also deeply moving",
    "the movie was fine and " * 60 + "but the ending was awful",
]


def snippet(s, n=30):
    s = s.replace("\n", " ")
    return repr(s if len(s) <= n else s[:n] + "...")


def main():
    n_err = n_diff = n_low = n_noev = n_trunc = 0
    for i, s in enumerate(CASES, 1):
        try:
            a = analyze(s)
            bert = "POS" if a["label"] else "NEG"
            anf = "POS" if a["anfis_label"] else "NEG"
            agree = a["label"] == a["anfis_label"]
            n_diff += not agree
            n_low += a["low_coverage"]
            n_noev += a["no_evidence"]
            n_trunc += a["truncated"]
            trunc = (" TRUNC" if a["truncated"] else "") + (" NOEVID" if a["no_evidence"] else "")
            print(f"{i:>2} | {bert} {a['conf']:.0%} | ANFIS {anf} | {'agree' if agree else 'DIFF '} | "
                  f"{a['rule']} | raw {a['raw']:.3f} | {'LOW-COV' if a['low_coverage'] else 'ok     '} | "
                  f"pos {a['pos']:.3f} neg {a['neg']:.3f} negn {a['negation']}{trunc} | {snippet(s)}")
        except Exception as e:  # keep going on any failure
            print(f"{i:>2} | ERROR {type(e).__name__}: {e} | {snippet(s)}")
            n_err += 1
    print(f"\n{len(CASES)} cases | ANFIS disagrees on {n_diff} | low coverage on {n_low} | errors {n_err} "
          f"| no lexicon evidence on {n_noev} | truncated {n_trunc}")


if __name__ == "__main__":
    main()
