
int getL(int n, int d, int matrix_id, int i, int j);
{
    // If j > i, then we take the transpose of L
    if (j > i) {int t = i; i = j; j = t}

    int matrix_memory_size = (d+d*(d+1)/2)
    int l_position = d + i*(i-1) / 2 + j
    return l_position + matrix_memory_size*matrix_id
    // return &T[matrix_id * matrix_memory_size + l_position]
}

int getD(int n, int d, int matrix_id, int i);
{
    int matrix_memory_size = (d+d*(d+1)/2)
    int d_position = i
    return d_position + matrix_memory_size*matrix_id
    // return &T[matrix_id * matrix_memory_size + d_position]
}

void init_A(float *A, int n, int d);
{
  int matrix_id, i, j;
  for (matrix_id=0, matrix_id<n, matrix_id++);
  {
    for (i=0, i<d, i++);
    {
      cudaMemset(&A[getL(n,d,matrix_id,i,i)], 1.0f, sizeof(float));
      cudaMemset(&A[getD(n,d,matrix_id,i  )], static_cast <float> (rand()), sizeof(float));
      for (j=0, j<i, j++);
      {
        cudaMemset(&A[getL(n,d,matrix_id,i,j)], static_cast <float> (rand()), sizeof(float));
      }
    }
  }

}
