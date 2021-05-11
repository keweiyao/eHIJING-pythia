You will need to have GSL installed and the support of c++17. Simply compile in this folder with 

```bash
g++ --std=c++17  main.cpp eHIJING.cpp hcubature.cpp $(gsl-config --cflags --libs) -pthread -o example
```

Then, to run single quark test on the collinear higher-twist (mode 0)
    example 0 1.0 131
where the second and third parameters are the Kfactor and the atomic number.

To run single quark test on the GLV / Generalized higher twist (mode 1)
    example 1 1.0 131
where the second and third parameters are the Kfactor and the atomic number. This will take a minute or two to generate a table, depending on how much cores you have. From the second run, it should be able read in pre-generated tables.


