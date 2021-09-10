set terminal svg size 350,262 font 'Times,10'
set output 'plot.svg'

set key autotitle columnhead
set datafile separator ","
#set title 'L1-Loss for Testing'
set xrange [0:150]
set yrange [0:14]
set xlabel "number of epochs"
set ylabel "L1 Loss Values"

plot 'stage_1.csv' using 2:3 title 'Stage 1' with lines ,\
'stage_2.csv' using 2:3 title 'Stage 2' with lines ,\
'stage_3.csv' using 2:3 title 'Stage 3' with lines ,\
'stage_4.csv' using 2:3 title 'Stage 4' with lines
