#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "implementacoes.h"

int paralela(int n, double *a, double *b, double *x_new, double epsilon, int max_iteracoes, double *erros) {
    double (*A)[n] = (double (*)[n]) a;
    double *x_old = malloc(n * sizeof(double));
    int iteracao;
    double soma;
    double erro;
    double diferenca;

    for (int i = 0; i < n; i++) {
        x_new[i] = 0.0;
        x_old[i] = 0.0;
    }

    for (iteracao = 0; iteracao < max_iteracoes; iteracao++) {
        // realiza o for em paralelo, schedule(static) pois cada iteracao faz sempre n operacoes (custo igual), soma e privada pois cada thread tem sua propria soma
        // usando o static evitamos o possivel overhead do dynamic
        #pragma omp parallel for schedule(static) private(soma)
        for (int i = 0; i < n; i++) {
            soma = 0.0;
            for (int j = 0; j < n; j++) {
                if (j != i) { 
                    soma += A[i][j] * x_old[j];
                }
            }
            x_new[i] = (b[i] - soma) / A[i][i];
        }

        erro = 0.0;
        // realiza o for em paralelo, schedule(static) pois cada iteracao tem custo igual, reduction(max) combina o maximo de cada thread em erro, diferenca e privada pois cada thread tem sua propria diferenca
        #pragma omp parallel for schedule(static) reduction(max:erro) private(diferenca)
        for (int i = 0; i < n; i++) {
            diferenca = x_new[i] - x_old[i];
            if (diferenca < 0) {
                diferenca = -diferenca;
            }
            if (diferenca > erro) {
                erro = diferenca;
            }
        }

        if (erros != NULL) erros[iteracao] = erro;

        if (erro < epsilon) {
            iteracao++;
            break;
        }

        // realiza o for em paralelo, schedule(static) pois cada iteracao tem custo igual, cada thread atualiza um intervalo de i
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            x_old[i] = x_new[i];
        }
    }

    free(x_old);
    return iteracao;
}
