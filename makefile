build_openmp:
	gcc advection2D.c -o advection2D --std=c99 -lm -fopenmp

build:
	gcc advection2D.c -o advection2D --std=c99 -lm

plot:
	./advection2D
	gnuplot plot_final
	gnuplot plot_initial
	gnuplot plot_avg
	imv initial.png
	imv final.png
	imv avg.png
