build:
	gcc advection2D.c -o advection2D --std=c99 -lm

build_openmp:
	gcc advection2D.c -o advection2D --std=c99 -lm -fopenmp

plot:
	./advection2D
	gnuplot plot_final
	imv final.png
