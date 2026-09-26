# OpenJEV Models Directory (`/models`)

Este directorio almacena los modelos locales y sus respectivos cabezales de decisión (`heads.safetensors`).

## 📁 Estructura Recomendada por Modelo

Cada subdirectorio representa un modelo disponible (por ejemplo `models/qwen2.5-0.5b`):

```text
models/
  ├── qwen2.5-0.5b/
  │     ├── config.json             # Metadatos del backbone (Hugging Face)
  │     ├── tokenizer.json          # Tokenizador BPE
  │     ├── tokenizer_config.json
  │     ├── model.safetensors       # Pesos del backbone (o *.safetensors en 4-bit)
  │     ├── heads.safetensors       # Cabezales canónicos de decisión (Requerido)
  │     └── heads_metadata.json     # Manifiesto de formas y procedencia
  └── README.md
```

## 🎯 Contrato Canónico de Cabezales (`heads.safetensors`)

OpenJEV requiere exactamente los siguientes 5 tensores con aserción estricta de formas:

| Nombre del Tensor | Forma (*Shape*) | Descripción |
|---|---|---|
| `bool_head.linear.weight` | `(1, hidden_dim)` | Proyección lineal para decisiones booleanas |
| `bool_head.linear.bias` | `(1,)` | Sesgo del cabezal booleano |
| `choice_head.w_choice.weight` | `(hidden_dim, hidden_dim)` | Matriz de proyección para opciones múltiples |
| `score_head.w_score.weight` | `(5, hidden_dim)` | Proyección para puntuaciones ordinales (5 niveles) |
| `score_head.w_score.bias` | `(5,)` | Sesgo del cabezal de puntuación |

## 🛠️ Herramienta CLI de Verificación e Inicialización

Para inicializar o verificar cabezales en una carpeta de modelo:

```bash
# Inicializar cabezales canónicos acordes al config.json del modelo
python -m core.model_scanner --model-dir ./models/qwen2.5-0.5b --init-heads

# Verificar la integridad de un heads.safetensors existente
python -m core.model_scanner --model-dir ./models/qwen2.5-0.5b --verify
```
