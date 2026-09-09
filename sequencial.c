#include <stdlib.h>
#include <math.h>
#include "implementacoes.h"

int sequencial(int n, double *a, double *b, double *x_new, double epsilon, int max_iteracoes, double *erros) {
    double (*A)[n] = (double (*)[n]) a;
    double *x_old = malloc(n * sizeof(double));
    int iteracao;
    double soma;
    double erro;
    double diferenca;

    // chute inicial = 0
    for (int i = 0; i < n; i++) {
        x_new[i] = 0.0;
        x_old[i] = 0.0;
    }

    // criterio de parada: numero maximo de iteracoes
    for (iteracao = 0; iteracao < max_iteracoes; iteracao++) {
        // calcula x_new a partir de x_old
        for (int i = 0; i < n; i++) {
            soma = 0.0;
            for (int j = 0; j < n; j++) {
                if (j != i) {
                    soma += A[i][j] * x_old[j];
                }
            }
            x_new[i] = (b[i] - soma) / A[i][i];
        }

        // criterio de parada: quando norma da diferenca entre x_new e x_old < epsilon (convergencia)
        erro = 0.0;
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

        // atualiza x_old com os novos valores
        for (int i = 0; i < n; i++) {
            x_old[i] = x_new[i];
        }
    }

    free(x_old);
    return iteracao;
}
