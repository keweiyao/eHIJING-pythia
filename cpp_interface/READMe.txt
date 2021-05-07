You will need to have GSL installed and the support of c++11
simply compile in this folder with 

g++ --std=c++11  main.cpp eHIJING.cpp hcubature.cpp $(gsl-config --cflags --libs) -pthread -o 

Then, to run single quark test on the collinear higher-twist (mode 0)
    run 0 1.0 131
where the second and third parameters are the Kfactor and the atomic number.

To run single quark test on the GLV / Generalized higher twist (mode 1)
    run 1 1.0 131
where the second and third parameters are the Kfactor and the atomic number. This will take a minute or two to generate a table, depending on how much cores you have. Currently, I have not implemented the read table function yet. So, each time you run it, it computes and writes a new table. In the future, it will be able to simply read old table.
