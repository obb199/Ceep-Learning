#include "activations.h"

bool activation_relu(matrix * m){
    if (m == NULL) return false;

    for(int i = 0; i < m->rows; i++){
        for(int j = 0; j < m->cols; j++){
            if (m->values[i][j] < 0) m->values[i][j] = 0;
        }
    }
}
