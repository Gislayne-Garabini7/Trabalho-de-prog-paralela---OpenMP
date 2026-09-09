// compilar: gcc -O2 -fopenmp -o jacobi main.c sequencial.c paralela.c -lm
//
// executar com matriz gerada internamente (n=1000, 4 threads):
//   ./jacobi 1000 4
//
// executar com arquivo de entrada (4 threads):
//   ./jacobi matriz.txt 4
//
// flags opcionais:
//   --epsilon 1e-10     (padrao: 1e-4)
//   --max-iter 100000   (padrao: 10000)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "implementacoes.h"

static void ler_arquivo(const char *arquivo, int *n, double **A, double **b) {
    FILE *arq = fopen(arquivo, "r");
    int tam;

    if (!arq) { perror(arquivo); exit(1); }

    fscanf(arq, "%d", n);
    tam = *n;

    *A = malloc((size_t)tam * tam * sizeof(double));
    *b = malloc(tam * sizeof(double));

    double (*mat)[tam] = (double (*)[tam]) *A;

    for (int i = 0; i < tam; i++) {
        for (int j = 0; j < tam; j++) {
            fscanf(arq, "%lf", &mat[i][j]);
        }
    }

    for (int i = 0; i < tam; i++) {
        fscanf(arq, "%lf", &(*b)[i]);
    }

    fclose(arq);
}

static void gerar_matriz(int n, double **A, double **b) {
    double (*mat)[n];
    double soma_linha;

    srand(10);

    // Usa size_t para evitar overflow na multiplicacao com matrizes gigantes
    *A = malloc((size_t)n * n * sizeof(double));
    *b = malloc(n * sizeof(double));

    mat = (double (*)[n]) *A;

    for (int i = 0; i < n; i++) {
        soma_linha = 0.0;
        for (int j = 0; j < n; j++) {
            if (i != j) {
                mat[i][j] = (rand() % 10) + 1; // [1, 10]
                soma_linha += mat[i][j];
            }
        }
        // margem de 1% acima da soma: quase nao-dominante, forcando muitas iteracoes
        mat[i][i] = soma_linha + (soma_linha * 0.01);
        (*b)[i] = (rand() % 100) + 1; // [1, 100]
    }
}

static void executar(int n, double *A, double *b, int n_threads,double epsilon, int max_iteracoes, int salvar_convergencia) {
    double *x = malloc(n * sizeof(double));
    double *erros = malloc(max_iteracoes * sizeof(double));
    int iter_seq;
    int iter_par;
    double t0;
    double t1;
    double tempo_seq;
    double t2;
    double tempo_par;
    double erro_seq;
    double erro_par;

    t0 = omp_get_wtime();
    iter_seq = sequencial(n, A, b, x, epsilon, max_iteracoes, erros);
    t1 = omp_get_wtime();
    tempo_seq = t1 - t0;
    erro_seq = erros[iter_seq - 1];

    omp_set_num_threads(n_threads);

    t2 = omp_get_wtime();
    iter_par = paralela(n, A, b, x, epsilon, max_iteracoes, erros);
    tempo_par = omp_get_wtime() - t2;
    erro_par = erros[iter_par - 1];

    if (n <= 10) {
        printf("solucao: ");
        for (int i = 0; i < n; i++) {
            printf("x(%d)=%.6f ", i + 1, x[i]);
        }
        printf("\n");
    }

    printf("sequencial   n=%-6d  iter=%-5d  erro=%.2e  tempo=%.6f s  convergiu=%s\n",
           n, iter_seq, erro_seq, tempo_seq, iter_seq < max_iteracoes ? "sim" : "nao");
    printf("paralelo     n=%-6d  iter=%-5d  erro=%.2e  tempo=%.6f s  threads=%d  convergiu=%s\n",
           n, iter_par, erro_par, tempo_par, n_threads, iter_par < max_iteracoes ? "sim" : "nao");
    printf("tolerancia=%.2e  speedup=%.4f  eficiencia=%.4f (%.1f%%)\n",
           epsilon, tempo_seq / tempo_par, tempo_seq / tempo_par / n_threads,
           tempo_seq / tempo_par / n_threads * 100);

    if (salvar_convergencia) {
        FILE *arq = fopen("convergencia.txt", "w");
        fprintf(arq, "iteracao    erro\n");
        for (int i = 0; i < iter_seq; i++) {
            fprintf(arq, "%-10d  %.10e\n", i + 1, erros[i]);
        }
        fclose(arq);
        printf("historico salvo em convergencia.txt\n");
    }

    free(x);
    free(erros);
}

int main(int argc, char *argv[]) {
    int n;
    int n_threads;
    double *A;
    double *b;
    double epsilon = 1e-4;
    int max_iteracoes = 10000;
    int salvar_convergencia = 0;

    if (argc < 2) {
        printf("Para executar utilize o seguinte comando:\n");
        printf("./jacobi <arquivo com matriz ou tamanho da matriz> <n_threads (opcional)> <--epsilon (opcional)> <--max-iter (opcional)> <--convergencia (opcional)>\n");
        return 1;
    }

    // se o argumento for um numero positivo, gera matriz internamente; caso contrario, le do arquivo
    if (atoi(argv[1]) > 0) {
        n = atoi(argv[1]);
        gerar_matriz(n, &A, &b);
    } else {
        ler_arquivo(argv[1], &n, &A, &b);
    }

    // numero de threads: argumento opcional, padrao 2 threads
    if (argc >= 3) {
        n_threads = atoi(argv[2]);
    } else {
        n_threads = 2; // valor padrao
    }

    // percorre os argumentos restantes em busca dos valores opcionais
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--epsilon") == 0 && i + 1 < argc) {
            epsilon = atof(argv[++i]);
        } else if (strcmp(argv[i], "--max-iter") == 0 && i + 1 < argc) {
            max_iteracoes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--convergencia") == 0) {
            salvar_convergencia = 1;
        }
    }

    executar(n, A, b, n_threads, epsilon, max_iteracoes, salvar_convergencia);

    free(A); free(b);
    return 0;
}
