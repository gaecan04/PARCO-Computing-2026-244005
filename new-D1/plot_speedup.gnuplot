set terminal pngcairo size 1400,800 noenhanced font "Arial,12"
set output "speedup_plot.png"
set title "Speedup Comparison (Sequential = 1)"
set style data histograms
set style histogram cluster gap 2
set style fill solid border -1
set boxwidth 0.9
set grid ytics
set ylabel "Speedup"
set yrange [0:*]
plot "speedup_data.dat" using 2:xtic(1) title "Sequential" linecolor rgb "#4e79a7", \
     "" using 3 title "MVM_parallel" linecolor rgb "#f28e2b", \
     "" using 4 title "MVM_parallel_atomic" linecolor rgb "#e15759", \
     "" using 5 title "MVM_parallel_sellc" linecolor rgb "#76b7b2"
