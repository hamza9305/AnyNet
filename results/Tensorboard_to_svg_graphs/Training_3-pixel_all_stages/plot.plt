set terminal svg size 350,262 font 'Times,10'
set output 'plot.svg'

set key autotitle columnhead
set datafile separator ","
#set title '3-Pixel Error'
set xrange [0:150]
set yrange [0:0.8]
set xlabel "number of epochs"
set ylabel "3-Pixel Error Values"

plot 'stage_1.csv' using 2:3 title 'Stage 1' with lines ,\
'stage_2.csv' using 2:3 title 'Stage 2' with lines ,\
'stage_3.csv' using 2:3 title 'Stage 3' with lines ,\
'stage_4.csv' using 2:3 title 'Stage 4' with lines
