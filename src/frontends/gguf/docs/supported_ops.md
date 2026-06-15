# GGML Operations Supported by OpenVINO GGUF Frontend

Here is a table of operations supported by the GGUF Frontend.
A "supported operation" is one that the GGUF Frontend can convert to the OpenVINO representation.

## GGML_OP

| Operation Name                  | Supported | Limitation |
|---------------------------------|-----------|------------|
| GGML_OP_ACC                     | NO        |            |
| GGML_OP_ADD                     | YES       |            |
| GGML_OP_ADD1                    | YES       |            |
| GGML_OP_ADD_ID                  | YES       |            |
| GGML_OP_ADD_REL_POS             | NO        |            |
| GGML_OP_ARANGE                  | NO        |            |
| GGML_OP_ARGMAX                  | NO        |            |
| GGML_OP_ARGSORT                 | YES       |            |
| GGML_OP_CLAMP                   | YES       |            |
| GGML_OP_CONCAT                  | YES       |            |
| GGML_OP_CONT                    | YES       |            |
| GGML_OP_CONV_2D                 | NO        |            |
| GGML_OP_CONV_2D_DW              | NO        |            |
| GGML_OP_CONV_3D                 | NO        |            |
| GGML_OP_CONV_TRANSPOSE_1D       | NO        |            |
| GGML_OP_CONV_TRANSPOSE_2D       | NO        |            |
| GGML_OP_COS                     | YES       |            |
| GGML_OP_COUNT_EQUAL             | NO        |            |
| GGML_OP_CPY                     | YES       |            |
| GGML_OP_CROSS_ENTROPY_LOSS      | NO        |            |
| GGML_OP_CROSS_ENTROPY_LOSS_BACK | NO        |            |
| GGML_OP_CUMSUM                  | NO        |            |
| GGML_OP_CUSTOM                  | NO        |            |
| GGML_OP_DIAG                    | NO        |            |
| GGML_OP_DIAG_MASK_INF           | NO        |            |
| GGML_OP_DIAG_MASK_ZERO          | NO        |            |
| GGML_OP_DIV                     | YES       |            |
| GGML_OP_DUP                     | NO        |            |
| GGML_OP_FILL                    | NO        |            |
| GGML_OP_FLASH_ATTN_BACK         | NO        |            |
| GGML_OP_FLASH_ATTN_EXT          | YES       |            |
| GGML_OP_GATED_DELTA_NET         | NO        |            |
| GGML_OP_GATED_LINEAR_ATTN       | NO        |            |
| GGML_OP_GET_REL_POS             | NO        |            |
| GGML_OP_GET_ROWS                | YES       |            |
| GGML_OP_GET_ROWS_BACK           | NO        |            |
| GGML_OP_GROUP_NORM              | NO        |            |
| GGML_OP_IM2COL                  | NO        |            |
| GGML_OP_IM2COL_3D               | NO        |            |
| GGML_OP_IM2COL_BACK             | NO        |            |
| GGML_OP_L2_NORM                 | NO        |            |
| GGML_OP_LEAKY_RELU              | NO        |            |
| GGML_OP_LOG                     | YES       |            |
| GGML_OP_MAP_CUSTOM1             | NO        |            |
| GGML_OP_MAP_CUSTOM2             | NO        |            |
| GGML_OP_MAP_CUSTOM3             | NO        |            |
| GGML_OP_MEAN                    | NO        |            |
| GGML_OP_MUL                     | YES       |            |
| GGML_OP_MUL_MAT                 | YES       |            |
| GGML_OP_MUL_MAT_ID              | YES       |            |
| GGML_OP_NONE                    | YES       |            |
| GGML_OP_NORM                    | YES       |            |
| GGML_OP_OPT_STEP_ADAMW          | NO        |            |
| GGML_OP_OPT_STEP_SGD            | NO        |            |
| GGML_OP_OUT_PROD                | NO        |            |
| GGML_OP_PAD                     | NO        |            |
| GGML_OP_PAD_REFLECT_1D          | NO        |            |
| GGML_OP_PERMUTE                 | YES       |            |
| GGML_OP_POOL_1D                 | NO        |            |
| GGML_OP_POOL_2D                 | NO        |            |
| GGML_OP_POOL_2D_BACK            | NO        |            |
| GGML_OP_REPEAT                  | YES       |            |
| GGML_OP_REPEAT_BACK             | NO        |            |
| GGML_OP_RESHAPE                 | YES       |            |
| GGML_OP_RMS_NORM                | YES       |            |
| GGML_OP_RMS_NORM_BACK           | NO        |            |
| GGML_OP_ROLL                    | NO        |            |
| GGML_OP_ROPE                    | YES       | M-RoPE mode (rotary embedding for vision-language models) is not supported |
| GGML_OP_ROPE_BACK               | NO        |            |
| GGML_OP_RWKV_WKV6               | NO        |            |
| GGML_OP_RWKV_WKV7               | NO        |            |
| GGML_OP_SCALE                   | YES       |            |
| GGML_OP_SET                     | NO        |            |
| GGML_OP_SET_ROWS                | YES       |            |
| GGML_OP_SILU_BACK               | NO        |            |
| GGML_OP_SIN                     | YES       |            |
| GGML_OP_SOFT_MAX                | YES       |            |
| GGML_OP_SOFT_MAX_BACK           | NO        |            |
| GGML_OP_SOLVE_TRI               | NO        |            |
| GGML_OP_SQR                     | YES       |            |
| GGML_OP_SQRT                    | YES       |            |
| GGML_OP_SSM_CONV                | NO        |            |
| GGML_OP_SSM_SCAN                | NO        |            |
| GGML_OP_SUB                     | YES       |            |
| GGML_OP_SUM                     | NO        |            |
| GGML_OP_SUM_ROWS                | YES       |            |
| GGML_OP_TIMESTEP_EMBEDDING      | NO        |            |
| GGML_OP_TOP_K                   | YES       |            |
| GGML_OP_TRANSPOSE               | YES       |            |
| GGML_OP_TRI                     | NO        |            |
| GGML_OP_UPSCALE                 | NO        |            |
| GGML_OP_VIEW                    | YES       |            |
| GGML_OP_WIN_PART                | NO        |            |
| GGML_OP_WIN_UNPART              | NO        |            |

## GGML_UNARY_OP

| Operation Name              | Supported | Limitation |
|-----------------------------|-----------|------------|
| GGML_UNARY_OP_ABS           | NO        |            |
| GGML_UNARY_OP_CEIL          | NO        |            |
| GGML_UNARY_OP_ELU           | YES       |            |
| GGML_UNARY_OP_EXP           | NO        |            |
| GGML_UNARY_OP_EXPM1         | NO        |            |
| GGML_UNARY_OP_FLOOR         | NO        |            |
| GGML_UNARY_OP_GELU          | YES       |            |
| GGML_UNARY_OP_GELU_ERF      | NO        |            |
| GGML_UNARY_OP_GELU_QUICK    | YES       |            |
| GGML_UNARY_OP_HARDSIGMOID   | NO        |            |
| GGML_UNARY_OP_HARDSWISH     | NO        |            |
| GGML_UNARY_OP_NEG           | NO        |            |
| GGML_UNARY_OP_RELU          | YES       |            |
| GGML_UNARY_OP_ROUND         | NO        |            |
| GGML_UNARY_OP_SGN           | NO        |            |
| GGML_UNARY_OP_SIGMOID       | YES       |            |
| GGML_UNARY_OP_SILU          | YES       |            |
| GGML_UNARY_OP_SOFTPLUS      | NO        |            |
| GGML_UNARY_OP_STEP          | NO        |            |
| GGML_UNARY_OP_TANH          | YES       |            |
| GGML_UNARY_OP_TRUNC         | NO        |            |
| GGML_UNARY_OP_XIELU         | NO        |            |

## GGML_GLU_OP

| Operation Name              | Supported | Limitation |
|-----------------------------|-----------|------------|
| GGML_GLU_OP_GEGLU           | YES       |            |
| GGML_GLU_OP_GEGLU_ERF       | NO        |            |
| GGML_GLU_OP_GEGLU_QUICK     | NO        |            |
| GGML_GLU_OP_REGLU           | NO        |            |
| GGML_GLU_OP_SWIGLU          | YES       |            |
| GGML_GLU_OP_SWIGLU_OAI      | YES       |            |
