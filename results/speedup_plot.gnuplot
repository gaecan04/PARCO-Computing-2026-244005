set terminal pngcairo size 1400,900 font "Arial,12"
set output "/mnt/c/Users/gaeca/.vscode/cli/results/speedup_all_matrices.png"

set datafile separator comma
set key outside
set style data histograms
set style histogram cluster gap 2
set style fill solid 1.00 border -1
set grid ytics

set title "Sequential vs Best Parallel Implementations"
set ylabel "Execution Time (ms)"

set xtics rotate by -45

plot "/mnt/c/Users/gaeca/.vscode/cli/results/speedup_comparison.csv" using 3:xtic(1) title columnhead(2)
