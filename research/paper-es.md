# Construcción de Modelos de Decisión Calibrados a partir de Modelos de Lenguaje Pequeños Pre-entrenados

### Formulación arquitectónica, patologías empíricas y correcciones metodológicas

**TESINA DE LICENCIATURA**

| | |
|---|---|
| **Universidad** | [completar] |
| **Facultad / Departamento** | [completar] |
| **Carrera** | Licenciatura en [completar] |
| **Autor/a** | [completar] |
| **Director/a** | [completar] |
| **Codirector/a** | [completar, si corresponde] |
| **Lugar y fecha** | [ciudad], octubre de 2026 |
| **Software desarrollado** | OpenJEV |
| **Modelos base utilizados** | Qwen 2.5-0.5B, Gemma 2 2B, OpenELM-450M |

---

## Resumen
Los modelos de lenguaje de gran tamaño (LLM) autorregresivos se utilizan ampliamente como motores de decisión. No obstante, las alucinaciones, latencias, costo computacional, sobreconfianza verbalizada, deriva de calibración y fallos en la entrega del formato de salida esperado representan una limitación importante en flujos operativos automatizados que afecta la confiabilidad tanto de sistemas agénticos como de sistemas tradicionales que consumen el output de los modelos.
Este trabajo explora la posibilidad de transformar un **modelo de lenguaje pequeño (SLM) pre-entrenado**, en un **modelo de decisión** no autorregresivo de salida estructurada y calibrada, capaz de responder de manera rápida y confiable a las tres priomitivas de decisión que son: clasificación (`noul`), enrutamiento (`choice`) y puntuación (`score`) utilizando exclusivamente técnicas de postentrenamiento que no modifiquen el *backbone* del modelo.
Se construyó un conjunto de datos sintético con validacion humana de eventos de decisión (estado/pregunta/respuesta) y se definieron protocolos de postentrenamiento de 30 épocas basados en generación de nuevas cabezas de atención sobre un frozen backbone con diferentes combinaciones de técnicas y se los aplicó a SLM de familias con arquitecturas muy distintas: Qwen 2.5-0.5B, Gemma 2 2B y OpenELM-450m.
El protocolo de entrenamiento con Cross entropy seguido de tratamiento con Focal Loss y Brier Score con reinicio del optimizador entre fases y escalado temperatura final produjo los mejores resultados de la experimentación en los tres modelos. Sobre una partición de prueba ciega (N = 300), alcanzaron una exactitud booleana ≥ 92.00 % (100.00 % en OpenELM), una exactitud de elección ≥ 96.00 % (98.00 % en OpenELM), un 0.00 % de errores continuos severos en la puntuación ordinal, un error de calibración esperado (ECE) entre 1.97 % y 7.34 % y una latencia de decisión inferior a 300 ms en GPU Metal de Apple Silicon.
Las cabezas de atención generadas ocuparon un espacio en disco de 3mb para Qwen2.5, 9mb para Gemma 2 2b y 21.3MB para OpenELM-450m.
Estos resultados respaldan la hipótesis de que es posible construir un modelo de decisión a partir de un SLM pre-entenado y que el metodo popuesto pemite posibilita hacelo con un alta performance en términos de latencia, precisión de decisión y una muy baja huella de hardware.


**Palabras clave:** modelos de decisión; modelos de lenguaje pequeños; calibración de probabilidades; sondeo lineal; clasificación no autorregresiva; flujos agénticos.

## Abstract

Autoregressive Large Language Models (LLMs) are widely used for classification, routing and scoring through token generation, but this paradigm suffers from hallucinations, latency and token costs, loss of output format, and miscalibration. This thesis investigates whether a *decision model* (a non-autoregressive, calibrated, structured-output system operating in a single forward pass) can be built from a pre-trained Small Language Model (SLM) with a frozen backbone. We formalize three decision primitives (Boolean, multi-option choice, and ordinal rubric scoring), implement a training and calibration pipeline, build a synthetic decision-event dataset, and evaluate it on three architecturally divergent model families (Qwen 2.5-0.5B, Gemma 2 2B, OpenELM-450M). On a blind test partition (N = 300), all models reach ≥ 92.00 % Boolean accuracy, ≥ 96.00 % choice accuracy, 0.00 % severe ordinal errors and an ECE between 1.97 % and 7.34 %, with sub-300 ms latency on Apple Silicon. The work also documents the empirical pathologies encountered and their corrections.

**Keywords:** decision models; small language models; probability calibration; linear probing; non-autoregressive classification; agentic workflows.

---

## Índice

1. Introducción
   1.1 Problemática observada · 1.2 Justificación · 1.3 Hipótesis de investigación · 1.4 Objetivo principal · 1.5 Objetivos específicos · 1.6 Tareas · 1.7 Preguntas de investigación · 1.8 Alcance y limitaciones · 1.9 Organización de la tesina
2. Marco teórico
   2.1 Modelos de lenguaje pequeños (SLM) y sus familias · 2.2 LLM autorregresivos y sus limitaciones · 2.3 Modelos de decisión · 2.4 Valor para la Industria 4.0 · 2.5 Proceso actual de creación de modelos de decisión · 2.6 Preguntas y respuestas de un modelo de decisión · 2.7 Técnicas de postentrenamiento · 2.8 Evaluación del rendimiento · 2.9 Diferencias de diseño intrínseco entre modelos
3. Diseño metodológico y arquitectura del sistema
   3.1 Enfoque metodológico · 3.2 Modelos seleccionados · 3.3 Arquitectura de decisión no autorregresiva · 3.4 Estrategia de entrenamiento y calibración · 3.5 Conjunto de datos · 3.6 Sistema de métricas · 3.7 Software y suite de pruebas · 3.8 Entorno experimental
4. Desarrollo iterativo: patologías metodológicas y correcciones
5. Resultados
   5.1 Dinámica de entrenamiento y evaluación ciega en Qwen · 5.2 Estudio de caso adversarial · 5.3 Ablación del ancho de banda σ · 5.4 Ablación de la metodología de entrenamiento (Qwen) · 5.5 Ablación en Gemma · 5.6 Ablación en OpenELM y patología del sumidero de atención · 5.7 Síntesis comparativa entre familias
6. Discusión
7. Conclusiones y trabajo futuro

Referencias

---

## Convenciones y siglas

**Convenciones.** En todo el documento se utiliza el punto como separador decimal, en coincidencia con las salidas del software y con la literatura técnica de referencia. Los nombres de primitivas (`noul`, `choice`, `score`), archivos y funciones se mantienen en su forma original en código.

| Sigla | Significado |
|---|---|
| LLM | *Large Language Model* (modelo de lenguaje de gran tamaño) |
| SLM | *Small Language Model* (modelo de lenguaje pequeño) |
| AR | Autorregresivo |
| BPE | *Byte-Pair Encoding* |
| BOS | *Beginning of Sequence* (token de inicio de secuencia) |
| CE | *Cross-Entropy* (entropía cruzada) |
| ECE | *Expected Calibration Error* (error de calibración esperado) |
| MAE / RMSE | Error absoluto medio / raíz del error cuadrático medio |
| TER | *Tolerance Exceedance Rate* (tasa de excedencia de tolerancia) |
| FFN | *Feed-Forward Network* |
| GQA | *Grouped-Query Attention* |
| RoPE | *Rotary Positional Embedding* |
| OOD | *Out-of-Distribution* (fuera de distribución) |
| MLX | Framework de aprendizaje automático de Apple para Apple Silicon |

---

# Capítulo 1. Introducción

## 1.1 Problemática observada

Los LLM a menudo incurren en "alucinaciones": generan respuestas plausibles pero incorrectas. Además, al ser autorregresivos consumen una gran cantidad de tokens, lo que introduce latencia y costos. Asimismo, en situaciones como el colapso del contexto dejan de respetar los formatos de salida solicitados. En flujos operativos automatizados donde se toman decisiones, esto representa un serio problema, ya que afecta la confiabilidad de los flujos agénticos y la resiliencia de los sistemas tradicionales que reciben la salida del LLM como entrada.

Como respuesta a este problema surgió un nuevo campo de aplicación en la ciencia de datos: los **modelos de decisión**. Estos son capaces de brindar respuestas estructuradas y, al no ser autorregresivos, reducen sustancialmente el costo computacional y la latencia. Sin embargo, al momento de escribir esta tesina, la variedad disponible de este tipo de modelos es limitada, y construir modelos propios desde cero requeriría un gran volumen de datos, un gran costo computacional y un esfuerzo considerable.

Por este motivo, resulta de vital interés crear modelos de decisión eficaces propios a partir de pequeños modelos de lenguaje pre-entrenados de acceso público, de modo que incorporarlos a los flujos agénticos actuales no represente un impacto significativo en la infraestructura de los sistemas existentes.

## 1.2 Justificación

La práctica habitual consiste en pedirle a un modelo generativo que emita un símbolo (por ejemplo, `"Sí"`, `"Facturación"` o `"Puntaje: 3"`) y luego interpretar el texto o las log-probabilidades de los tokens emitidos. Para decisiones que en esencia requieren una única respuesta categórica, ese procedimiento resulta ineficiente, frágil y difícil de calibrar. Disponer de un procedimiento reproducible para convertir SLM públicos en modelos de decisión permitiría reemplazar etapas generativas de los flujos actuales por componentes más rápidos, más baratos, con salida estructurada garantizada y con probabilidades confiables, sin reentrenar ni reemplazar los modelos base.

## 1.3 Hipótesis de investigación

> **"Es posible construir un modelo de decisión a partir de un SLM pre-entrenado."**

En el marco de esta tesina, la hipótesis se contrasta además en su versión de generalidad arquitectónica (denominada *Hipótesis de Traducibilidad Universal a Sistema 1*): que la proyección de representaciones autorregresivas congeladas hacia cabezales de decisión calibrados no es idiosincrática de una única arquitectura, sino una propiedad transferible a familias de modelos de linajes arquitectónicos divergentes.

## 1.4 Objetivo principal

Construir un modelo de decisión a partir de un SLM pre-entrenado.

## 1.5 Objetivos específicos

1. Construir un software que ejecute diferentes técnicas de *fine-tuning* sobre un SLM.
2. Construir una suite de tests que garantice el correcto funcionamiento del software.
3. Construir un conjunto de datos de eventos de decisión (estado/pregunta/respuesta).
4. Seleccionar técnicas de *fine-tuning*.
5. Comparar el impacto de las diferentes técnicas de entrenamiento en el rendimiento de los modelos.
6. Evaluar el rendimiento en modelos procedentes de diferentes familias.

## 1.6 Tareas

1. Identificar las diferentes familias de modelos.
2. Elegir y descargar un SLM de diferentes familias.
3. Generar un conjunto de datos sintético apto para entrenamiento, testeo y validación de un modelo de decisión.
4. Seleccionar un conjunto de técnicas de postentrenamiento.
5. Entrenar los modelos descargados con diferentes conjuntos de técnicas de postentrenamiento.
6. Definir el sistema de métricas de evaluación de rendimiento del modelo.
7. Evaluar el rendimiento de los modelos entrenados.
8. Elaborar una tabla de ablación individual y una comparativa del rendimiento de los modelos postentrenados.

## 1.7 Preguntas de investigación

| N.º | Pregunta | Se responde en |
|---|---|---|
| P1 | ¿Qué es un SLM? | §2.1 |
| P2 | ¿Cuáles son las distintas familias de SLM? | §2.1, §3.2 |
| P3 | ¿Qué es un modelo de decisión y cómo se diferencia de los LLM? | §2.2, §2.3 |
| P4 | ¿Cómo aporta valor un modelo de decisión a la Industria 4.0? | §2.4 |
| P5 | ¿Cómo es el proceso de creación de modelos de decisión en la actualidad? | §2.5, §3.4 |
| P6 | ¿Con qué clase de preguntas y respuestas opera un modelo de decisión? | §2.6, §3.3 |
| P7 | ¿Cuáles son las distintas técnicas de postentrenamiento? | §2.7, §3.4, §5.4 |
| P8 | ¿Cómo se evalúa el rendimiento de un modelo de decisión? | §2.8, §3.6 |
| P9 | ¿Qué desafíos presentan las diferencias de diseño intrínseco de los modelos? | §2.9, Cap. 4, §5.6 |

## 1.8 Alcance y limitaciones

**Alcance.** El estudio abarca (i) tres familias de SLM (Qwen 2.5-0.5B, Gemma 2 2B y OpenELM-450M); (ii) tres primitivas de decisión (`noul`, `choice`, `score`); (iii) ajuste exclusivo de cabezales sobre *backbone* congelado, con distintas combinaciones de funciones de pérdida, currículo de entrenamiento y calibración; (iv) un conjunto de datos sintético bilingüe (inglés y español rioplatense/latinoamericano) del dominio de atención al cliente (facturación, soporte técnico, ventas y cancelaciones); y (v) mediciones de latencia y memoria en GPU Metal de Apple Silicon mediante el framework MLX.

**Limitaciones.** (a) La comparación contra el ajuste fino completo del *backbone* se plantea de manera conceptual (§2.5) y no mediante un experimento propio. (b) Los datos son sintéticos y de un único dominio; no se evaluó sobre tráfico real de producción. (c) Las particiones de prueba contienen 100 instancias por primitiva, por lo que las diferencias de pocos puntos porcentuales deben interpretarse con cautela. (d) Las matrices de ablación se ejecutaron con una única semilla aleatoria (`seed = 42`). (e) Las mediciones de latencia corresponden a un único entorno de hardware.

## 1.9 Organización de la tesina

El Capítulo 2 presenta el marco teórico y responde las preguntas conceptuales de investigación. El Capítulo 3 describe el diseño metodológico: arquitectura de decisión, estrategia de entrenamiento, conjunto de datos, métricas y software. El Capítulo 4 documenta las patologías empíricas halladas durante el desarrollo y sus correcciones. El Capítulo 5 reporta los resultados experimentales, incluidas las tablas de ablación y la comparación entre familias. El Capítulo 6 discute los hallazgos y propone lineamientos prácticos, y el Capítulo 7 presenta las conclusiones y el trabajo futuro.

---

# Capítulo 2. Marco teórico

## 2.1 Modelos de lenguaje pequeños (SLM) y sus familias

**Definición operativa.** En esta tesina se denomina *modelo de lenguaje pequeño (SLM)* a un modelo de lenguaje pre-entrenado, basado en la arquitectura Transformer de tipo *decoder-only*, con un número de parámetros del orden de las décimas a pocas unidades de miles de millones (en este trabajo, entre 0.45 B y 2.0 B), lo bastante compacto como para ejecutarse localmente en hardware de consumo o *edge*, con latencia y memoria acotadas. Se trata de una definición operativa del presente trabajo y no de un umbral normativo.

**Familias.** Se entiende por *familia* un linaje de modelos con decisiones de diseño compartidas (arquitectura, normalización, mecanismo de atención, tokenizador, protocolo de pre-entrenamiento). Además de las tres familias evaluadas en esta tesina, existen otras de tamaño comparable, como Llama (Meta), Phi (Microsoft) o SmolLM (Hugging Face) **[completar con referencias bibliográficas]**. Las familias seleccionadas fueron:

| Familia | Modelo | Parámetros | Capas | Dim. oculta ($d$) | Rasgos distintivos |
|---|---|---|---|---|---|
| Linaje tipo Llama (Alibaba) | Qwen 2.5-0.5B | 0.49 B | 24 | 896 | Transformer *pre-norm* estándar con activaciones SwiGLU, RoPE y atención causal completa uniforme |
| Google | Gemma 2 2B | 2.00 B | 26 | 2304 | Embeddings escalados por $\sqrt{d}$, atención intercalada local (ventana deslizante) y global, *soft-capping* de logits con tangente hiperbólica, doble RMSNorm |
| Apple (no uniforme) | OpenELM-450M | 0.45 B | 20 | 1536 | Asignación de parámetros variable por capa: cabezas de consulta de 12 a 24, cabezas KV de 3 a 6, multiplicadores FFN de 0.5× a 4.0× |

## 2.2 LLM autorregresivos y sus limitaciones

Los LLM generativos se entrenan con un objetivo autorregresivo de predicción del siguiente token:

$$\mathcal{L}_{\text{AR}} = -\sum_t \log P(w_t \mid w_{<t})$$

Para tareas de decisión (clasificación de intención, priorización de tickets, evaluación de sentimiento) se los adapta habitualmente mediante *prompting*: se les solicita emitir un símbolo y luego se analiza el texto generado o las log-probabilidades. Este paradigma introduce limitaciones que se suman a las alucinaciones y a la pérdida de formato mencionadas en §1.1:

1. **Ineficiencia computacional.** La generación consume recursos proporcionales a la longitud de la salida, exige reservar memoria de caché clave-valor (KV) y está limitada por el ancho de banda de memoria secuencial, para juicios que en esencia requieren una sola decisión categórica.
2. **Interferencia de contexto y sensibilidad al orden.** Cuando se formulan varias preguntas sucesivas sobre un mismo texto (el "estado"), el procesamiento autorregresivo obliga a que las preguntas posteriores atiendan a los tokens de las anteriores, lo que genera dilución de contexto, interferencia entre preguntas y permutaciones no deterministas.
3. **Descalibración epistémica.** Las probabilidades de los tokens suelen estar mal calibradas, con sobreconfianza severa ante entradas fuera de distribución y sensibilidad a variaciones superficiales de la redacción del prompt.

## 2.3 Modelos de decisión

Un **modelo de decisión** es un modelo que, dado un *estado* (un texto o *payload* estructurado) y una o más *preguntas*, devuelve respuestas **estructuradas y tipadas** acompañadas de probabilidades calibradas, **en una única pasada hacia adelante y sin generación autorregresiva**. En esta tesina se lo denomina también *modelo de Sistema 1*, en analogía con la distinción de Kahneman (2011) entre un pensamiento rápido, automático e intuitivo (Sistema 1) y uno deliberado y secuencial (Sistema 2), aquí asimilable al razonamiento generativo paso a paso.

| Dimensión | LLM generativo con *prompting* | Modelo de decisión |
|---|---|---|
| Salida | Texto libre que debe interpretarse (*parsing*) | Respuesta estructurada y tipada (`noul`, `choice`, `score`) |
| Pasadas del modelo | Una por token generado | Una única pasada, aun con $K$ preguntas concurrentes |
| Costo y latencia | Proporcionales a la longitud de la salida | Constantes y acotados |
| Respeto del formato | Puede degradarse (p. ej., colapso de contexto) | Garantizado por construcción (cabezales de proyección) |
| Interacción entre preguntas | Las preguntas posteriores atienden a las anteriores | Aislamiento mutuo estricto mediante máscara por bloques |
| Confianza | Probabilidades de tokens, frecuentemente sobreconfiadas | Probabilidades calibradas (pérdida intrínseca + escalado por temperatura) |

## 2.4 Valor para la Industria 4.0

En entornos de Industria 4.0 las decisiones operativas (enrutar una solicitud, clasificar una incidencia, determinar urgencia, puntuar una condición según una rúbrica) se encadenan dentro de flujos automatizados y se consumen por sistemas tradicionales que esperan formatos fijos. Un modelo de decisión aporta valor porque: (i) entrega salidas estructuradas que los sistemas posteriores pueden consumir sin *parsing* frágil; (ii) reduce latencia y costo respecto de la generación de tokens, lo que habilita decisiones de alto rendimiento y baja latencia (en las pruebas de esta tesina, entre 61 ms y 283 ms por decisión concurrente); (iii) entrega probabilidades calibradas que permiten establecer umbrales de confianza y escalar a revisión humana los casos inciertos; (iv) requiere memoria reducida (entre ~380 MB y ~1.47 GB en las pruebas), lo que posibilita su ejecución local o en el borde; y (v) permite reutilizar un único *backbone* congelado con múltiples perfiles de cabezales, sin alterar la infraestructura existente. **[Reforzar con bibliografía sobre Industria 4.0 y sistemas agénticos.]**

## 2.5 Proceso actual de creación de modelos de decisión

Se identifican cuatro vías para obtener un componente de decisión:

1. **Generación con prompting de un LLM**, seguida de *parsing* del texto o de las log-probabilidades (paradigma estándar descrito en §2.2).
2. **Entrenamiento desde cero** de un modelo propio: requiere gran volumen de datos, alto costo computacional y esfuerzo considerable.
3. **Ajuste fino completo (*full fine-tuning*)** del *backbone* preentrenado.
4. **Ajuste exclusivo de cabezales (*head-only tuning*, sondeo lineal o *linear probing*)** sobre un *backbone* congelado, estrategia adoptada en esta tesina.

| Dimensión | Ajuste fino completo | Ajuste exclusivo de cabezales (OpenJEV) |
|---|---|---|
| Huella de parámetros | Actualiza el 100 % de los pesos (0.45 B a 2.0 B) | Actualiza < 0.1 % de los pesos (~3.2 MB en `heads.safetensors`) |
| Integridad de la variedad lingüística | Alto riesgo de **olvido catastrófico** y colapso de representaciones | Preserva la geometría lingüística preentrenada |
| Costo de hardware y cómputo | Clústeres multi-GPU, horas de retropropagación | < 30 segundos en GPU Metal de Apple Silicon (memoria unificada) |
| Agilidad de servicio | Requiere almacenar y cargar *checkpoints* completos separados | Varios perfiles de cabezales sobre un único *backbone* congelado |
| Estabilidad de calibración | Tiende a sobreajustar los logits a la distribución de entrenamiento | Las proyecciones lineales mantienen probabilidades suaves y bien condicionadas |

Esta comparación se sustenta en consideraciones de diseño y no en un experimento de ajuste fino completo realizado en esta tesina (véase §1.8).

## 2.6 Preguntas y respuestas de un modelo de decisión

Un **evento de decisión** es una terna *(estado, pregunta, respuesta)*. El estado $S$ es el texto o *payload* de entrada; cada pregunta $Q_k$ pertenece a una de tres primitivas y determina el tipo de la respuesta:

| Primitiva | Pregunta | Respuesta |
|---|---|---|
| `noul` (booleana) | Evaluación de verdad, cumplimiento o urgencia (p. ej., "¿es urgente?") | Valor $\{0,1\}$ con probabilidad $P(\text{true})$ y confianza |
| `choice` (elección) | Selección entre $N$ opciones, cada una con su descripción (p. ej., `billing`, `technical`, `sales`) | Opción elegida con distribución de probabilidad sobre las $N$ opciones |
| `score` (puntuación ordinal) | Valoración en una rúbrica de $M$ niveles ordenados (p. ej., frustración: 0 = calma, 1 = frustrado pero cordial, 2 = muy enojado) | Distribución $\mathbf{p}$ sobre niveles, puntuación esperada continua $\bar{S}$ y varianza |

Ejemplo de respuesta estructurada para la primitiva `score`:

```json
{
  "frustration": {
    "type": "score",
    "score": 1.96,
    "legend": { "0": "Calm, just stating facts", "1": "Frustrated but civil", "2": "Very angry, strong language" },
    "probabilities": { "0": 0.0117, "1": 0.0188, "2": 0.9695 }
  }
}
```

## 2.7 Técnicas de postentrenamiento

Las técnicas consideradas en esta tesina, todas aplicadas sobre los cabezales de decisión con el *backbone* congelado, son:

- **Entropía cruzada (CE) clásica.** Maximiza la separación lineal de clases y la exactitud *top-1*, pero empuja los logits hacia valores extremos y produce probabilidades sobreconfiadas (Guo et al., 2017).
- **Pérdida Focal** (Lin et al., 2017), que atenúa la contribución de los ejemplos fáciles.
- **Puntaje de Brier** (Brier, 1950), una *regla de puntuación estrictamente propia* (Gneiting & Raftery, 2007): $\frac{1}{C}\sum_c (p_c - y_c)^2$ alcanza su mínimo global si y solo si las probabilidades predichas coinciden con las probabilidades posteriores verdaderas, por lo que penaliza la sobreconfianza durante el entrenamiento. La pérdida combinada es $\mathcal{L} = \mathcal{L}_{\text{Focal}}(\gamma = 1.0) + \lambda \cdot \mathcal{L}_{\text{Brier}}(\lambda = 0.5)$.
- **Pérdida ordinal con suavizado gaussiano de etiquetas.** Reemplaza la etiqueta *one-hot* por una distribución suave centrada en el nivel verdadero $y^*$:
  $$\tilde{y}_i = \frac{\exp\left(-\frac{(i - y^*)^2}{2\sigma^2}\right)}{\sum_{j=0}^{M-1} \exp\left(-\frac{(j - y^*)^2}{2\sigma^2}\right)}$$
  y se optimiza $\mathcal{L}_{\text{score}} = -\sum_i \tilde{y}_i \log p_i$, con ancho de banda $\sigma = 0.35$ (véase §4.5).
- **Currículo de entrenamiento en fases**, con arranque en caliente (*warm-start*) y manejo del estado del optimizador (continuidad o reinicio de los momentos de AdamW).
- **Escalado por temperatura posterior al entrenamiento.** Se optimiza un escalar $T > 0$ sobre la partición de validación, $\hat{p}_i = \text{Softmax}(z_i / T)$ (o $\sigma(z/T)$ en el caso booleano). Al ser una transformación estrictamente monótona, preserva el *argmax* y por lo tanto la exactitud (Guo et al., 2017).

## 2.8 Evaluación del rendimiento de un modelo de decisión

La evaluación combina métricas de acierto, de error ordinal continuo, de calibración y de eficiencia de servicio:

- **Exactitud (*exact match*)** para `noul` y `choice`.
- **Error absoluto medio continuo** $\text{MAE}_{\text{cont}} = \frac{1}{N}\sum_i \lvert \bar{S}_i - y_i^* \rvert$, **RMSE continuo** y **tasa de excedencia de tolerancia** $\text{TER}_{\tau}$ para `score` (§3.6).
- **Error de calibración esperado (ECE)**, que agrupa las predicciones en *bins* de confianza y promedia, ponderando por tamaño, la diferencia absoluta entre la exactitud y la confianza medias de cada *bin*: $\text{ECE} = \sum_b \frac{\lvert B_b \rvert}{N} \lvert \text{acc}(B_b) - \text{conf}(B_b) \rvert$ (Guo et al., 2017). Se reportan variantes de ancho igual y adaptativa por cuantiles.
- **Puntaje de Brier** como medida de dispersión probabilística global.
- **Latencia (p50 y p95)** y **memoria pico** de la decisión concurrente de las tres primitivas.

## 2.9 Diferencias de diseño intrínseco entre modelos

Los SLM difieren en aspectos que afectan directamente a un modelo de decisión construido sobre su *backbone*:

- **Tokenización:** tamaño de vocabulario (32 000, 151 936 y 256 000 tokens en OpenELM, Qwen y Gemma, respectivamente) y políticas sobre tokens especiales (por ejemplo, si el modelo espera un token BOS al inicio).
- **Escala de la representación:** embeddings escalados por $\sqrt{d}$ (Gemma), dimensiones ocultas de 896 a 2304.
- **Normalización:** RMSNorm simple o doble, y *soft-capping* de logits.
- **Atención:** causal uniforme, de consulta agrupada (GQA) o intercalada local/global.
- **Uniformidad por capa:** arquitecturas homogéneas frente a la asignación de cabezas y multiplicadores FFN variable por capa de OpenELM.
- **Convenciones de implementación:** nombres de módulos internos (`.model` frente a `.transformer`) y claves de configuración (`hidden_size` frente a `model_dim`).

Estas diferencias dieron lugar a fallas específicas (sumidero de atención por omisión del token BOS, fricciones de configuración y despacho), documentadas en el Capítulo 4 y en la §5.6.

---

# Capítulo 3. Diseño metodológico y arquitectura del sistema

## 3.1 Enfoque metodológico

La investigación es de tipo experimental y aplicado, con desarrollo iterativo de software. El procedimiento fue: (i) seleccionar un SLM por familia; (ii) diseñar la arquitectura de decisión no autorregresiva; (iii) construir el conjunto de datos sintético y sus particiones; (iv) entrenar cabezales sobre el *backbone* congelado bajo distintas combinaciones de técnicas; (v) diagnosticar y corregir las fallas observadas en cada iteración; y (vi) evaluar sobre una partición de prueba ciega mediante un sistema de métricas definido de antemano, incluyendo matrices de ablación. Un rasgo deliberado del trabajo es que **no se presenta una narrativa de éxito depurada**: se documentan las patologías y los errores metodológicos encontrados (Capítulo 4) porque constituyen un resultado en sí mismo.

## 3.2 Modelos seleccionados

Las tres familias se eligieron por su divergencia arquitectónica, con el fin de contrastar la hipótesis de generalidad:

| Propiedad arquitectónica | Qwen 2.5-0.5B (tipo Llama) | OpenELM-450M (Apple) | Gemma 2 2B (Google) | Impacto metodológico |
|---|---|---|---|---|
| Cantidad de parámetros | 0.49 B | 0.45 B | 2.00 B | Rango de escala 0.45 B → 2.00 B |
| Capas ($L$) | 24 | 20 | 26 | Profundidad comparable |
| Dimensión oculta ($d$) | **896** | **1536** | **2304** | Variación de 2.57× en dimensionalidad latente |
| Uniformidad por capa | Cabezas y FFN uniformes | **No uniforme:** cabezas de consulta 12 → 24, FFN 0.5× → 4.0× | Cabezas y FFN uniformes | Pone a prueba la resiliencia ante asimetría estructural |
| Mecanismo de atención | Autoatención causal uniforme | Atención de consulta agrupada (cabezas KV: 3 → 6) | Ventana deslizante intercalada (4K) y global (8K) | Campos receptivos locales, agrupados y globales |
| Normalización de entrada | Varianza unitaria estándar | RMSNorm previa a la FFN | Embedding escalado por $\sqrt{d} \approx 48.0$ | Escalas iniciales de estado oculto diversas |
| Normalización por capa | RMSNorm única previa a la atención | RMSNorm previa a la atención | RMSNorm doble (previa y posterior a la FFN) | Estructuras de normalización internas variadas |
| Tamaño de vocabulario | 151 936 | 32 000 | 256 000 | Distinta granularidad de tokenización |

Los modelos se descargaron en sus versiones públicas y se utilizaron sin modificar sus pesos; las referencias técnicas son Bai et al. (2024), Gemma Team et al. (2024) y Mehta et al. (2024).

## 3.3 Arquitectura de decisión no autorregresiva

### 3.3.1 Disposición y empaquetado de la secuencia

Sea $S = (s_1, \dots, s_{|S|})$ la secuencia de tokens del estado y sean $Q_1, \dots, Q_K$ las $K$ preguntas concurrentes, con $Q_k = (q_{k,1}, \dots, q_{k,|Q_k|})$. En lugar de ejecutar $K$ pasadas independientes, se empaqueta una única secuencia contigua:

$$X = [S \circ Q_1 \circ Q_2 \circ \dots \circ Q_K]$$

donde $\circ$ denota concatenación y la longitud total es $T = |S| + \sum_{k=1}^K |Q_k|$.

### 3.3.2 Máscara causal por bloques bidimensional e invariancia a permutaciones

Para evaluar las $K$ preguntas de forma concurrente sin contaminación cruzada no puede usarse atención causal triangular inferior completa ni atención bidireccional. Se construye una máscara aditiva $M \in \mathbb{R}^{T \times T}$ con aislamiento estricto por bloques:

$$M_{i, j} = \begin{cases}
0.0 & \text{si } i \in S \text{ y } j \in S \text{ y } j \le i \quad \text{(atención causal del estado)} \\
0.0 & \text{si } i \in Q_k \text{ y } j \in S \quad \text{(la pregunta atiende al estado)} \\
0.0 & \text{si } i \in Q_k \text{ y } j \in Q_k \text{ y } j \le i \quad \text{(atención causal intra-pregunta)} \\
-10^9 & \text{si } i \in S \text{ y } j \in Q_k \quad \text{(el estado nunca atiende a las preguntas)} \\
-10^9 & \text{si } i \in Q_k \text{ y } j \in Q_m \text{ con } k \ne m \quad \text{(atención cruzada entre preguntas nula)}
\end{cases}$$

```
             Topología de la máscara de atención M
              Tokens S     Tokens Q_1    Tokens Q_2
           ┌─────────────┬────────────┬────────────┐
 Tokens S  │ Causal (0)  │   -1e9     │   -1e9     │
           ├─────────────┼────────────┼────────────┤
 Tokens Q_1│ Total (0)   │ Causal (0) │   -1e9     │
           ├─────────────┼────────────┼────────────┤
 Tokens Q_2│ Total (0)   │   -1e9     │ Causal (0) │
           └─────────────┴────────────┴────────────┘
```

**Reinicio de los embeddings posicionales rotatorios (RoPE).** Como RoPE codifica desplazamientos relativos a partir de índices absolutos, ubicar $Q_k$ en los desplazamientos $|S| + \sum_{m<k}|Q_m|$ violaría la invariancia a permutaciones: cambiar el orden de las preguntas en la solicitud alteraría las representaciones. Para garantizar invariancia matemática estricta, los índices posicionales se reinician en cada frontera de segmento:

$$\text{pos}(s_i) = i, \qquad \text{pos}(q_{k,j}) = |S| + j \quad \forall k \in \{1, \dots, K\}$$

Bajo esta formulación, la representación del último token de la pregunta $k$, $h_{Q_k} \in \mathbb{R}^d$, satisface $h_{Q_k}(S, Q_k) \equiv h_{Q_k}(S, Q_1, \dots, Q_K)$ para todo $Q_{m \ne k}$.

### 3.3.3 Cabezales de decisión canónicos

Para cada pregunta $Q_k$, la representación agrupada $h_{Q_k}$ en el último token del segmento se proyecta hacia cabezales lineales especializados.

**Cabezal booleano (`noul`).**
$$P(\text{true}) = \sigma(W_{\text{bool}} h_{Q_k} + b_{\text{bool}}), \quad W_{\text{bool}} \in \mathbb{R}^{1 \times d}, \quad b_{\text{bool}} \in \mathbb{R}$$
$$\text{noul} = \mathbb{I}(P(\text{true}) \ge 0.5) \in \{0, 1\}, \qquad C_{\text{bool}} = \max(P(\text{true}), 1 - P(\text{true})) \in [0.5, 1.0]$$

**Cabezal de elección (`choice`).** Para $N$ opciones candidatas $C_1, \dots, C_N$ con su descripción, el *backbone* congelado codifica cada descripción de forma independiente como un vector $e_{C_i} \in \mathbb{R}^d$. Una interacción bilineal residual puntúa la consulta contra cada candidato:
$$h_{\text{proj}} = h_{Q_k} + W_{\text{choice}} h_{Q_k}, \quad W_{\text{choice}} \in \mathbb{R}^{d \times d}$$
$$z_i = \frac{1}{\tau} \left\langle \frac{h_{\text{proj}}}{\lVert h_{\text{proj}} \rVert_2}, \frac{e_{C_i}}{\lVert e_{C_i} \rVert_2} \right\rangle, \qquad P(C_i) = \frac{\exp(z_i)}{\sum_{j=1}^N \exp(z_j)}, \qquad \text{choice} = \arg\max_i P(C_i)$$
con $\tau > 0$ una temperatura contrastiva (aprendible o calibrada; véase la corrección de la §4.2, que fija $\tau = 0.03$).

**Cabezal de puntuación ordinal (`score`).** Para una rúbrica de $M$ niveles ordenados $\{0, \dots, M-1\}$:
$$z = W_{\text{score}} h_{Q_k} + b_{\text{score}}, \quad W_{\text{score}} \in \mathbb{R}^{M_{\max} \times d}, \quad \mathbf{p} = \text{Softmax}(z_{0:M}) \in \Delta^{M-1}$$
$$\bar{S} = \sum_{i=0}^{M-1} i \cdot p_i \in [0, M-1], \qquad \sigma_S^2 = \sum_{i=0}^{M-1} (i - \bar{S})^2 p_i, \qquad \text{nivel\_discreto} = \text{round}(\bar{S})$$

## 3.4 Estrategia de entrenamiento y calibración

### 3.4.1 Ajuste exclusivo de cabezales con calibración en dos niveles

Se evita deliberadamente el ajuste fino completo del *backbone*. Este actúa como extractor fijo de representaciones contextuales $h_{Q_k} \in \mathbb{R}^d$, y las proyecciones lineales ($W_{\text{bool}}, W_{\text{choice}}, W_{\text{score}}$) solo orientan esas representaciones hacia fronteras de tarea discretas. La calibración se incorpora en dos niveles:

```
        [Estado de contexto + esquema de preguntas]
                          │
                          ▼
              [Backbone Transformer congelado]
              (Qwen 2.5 / Gemma 2 / OpenELM)
                          │
                          ▼
            Vector oculto agrupado h_Q ∈ ℝ^d
                          │
       ┌──────────────────┴──────────────────────┐
       ▼                                         ▼
[Nivel 1: Calibración intrínseca]     [Nivel 2: Calibración extrínseca]
Modelado de la pérdida en            Escalado por temperatura posterior
entrenamiento                         • Platt sobre val.jsonl
• Focal + Brier combinadas            • Ajusta un escalar T en < 50 ms
• Penaliza predicciones erróneas      • Preserva la exactitud (argmax)
  sobreconfiadas                      • Se exporta a calibration.json
```

- **Nivel 1 (intrínseco).** Se rechaza la entropía cruzada estándar, porque alienta a los logits a divergir hacia $\pm\infty$, y se emplea $\mathcal{L} = \mathcal{L}_{\text{Focal}}(\gamma = 1.0) + \lambda\,\mathcal{L}_{\text{Brier}}(\lambda = 0.5)$.
- **Nivel 2 (extrínseco).** Una vez entrenados y congelados los cabezales, se optimiza $T \in \mathbb{R}^+$ minimizando la log-verosimilitud negativa sobre `val.jsonl`. Al ser monótona, no altera el orden de las clases. Ante un dominio nuevo (OOD) no se requiere reentrenar: con 100–200 muestras de calibración del dominio, el ajuste lleva menos de 50 ms y redujo el ECE fuera de distribución de 18.26 % a 3.34 %.

### 3.4.2 Currículo de entrenamiento en tres fases

```
┌──────────────────────────────────────────────────────────────────────────────┐
│            ENTRENAMIENTO DESACOPLADO DE DECISIÓN EN TRES FASES               │
│                                                                              │
│ [Fase 1: Alineación de representaciones]                                     │
│  • Sondeo lineal grueso sobre embeddings del backbone congelado              │
│  • Comparación: entropía cruzada clásica (CE) vs. modelado intrínseco (FB)   │
│  • 60 épocas | η₁ = 1e-3 | AdamW (β₁ = 0.9, β₂ = 0.999)                      │
│                               │                                              │
│                               ▼                                              │
│ [Fase 2: Refinamiento intrínseco de márgenes (arranque en caliente)]         │
│  • Parte de los pesos de la Fase 1; reshapea márgenes con reglas propias     │
│  • Pérdida: Focal + Brier (bool/choice) y ordinal gaussiana (score)          │
│  • 30 épocas | tasa de aprendizaje atenuada η₂ = 2e-4                        │
│  • Estado del optimizador:                                                   │
│      - Modalidad A (reset): reinicia m_t = 0, v_t = 0                        │
│      - Modalidad B (continuous): conserva los momentos acumulados            │
│                               │                                              │
│                               ▼                                              │
│ [Fase 3: Recalibración extrínseca posterior]                                 │
│  • Escalado por temperatura (T > 0) ajustado sobre val.jsonl                 │
│  • Transformación monótona: preserva el argmax y el MAE                      │
│  • Ajusta T_bool por búsqueda en grilla (ECE/NLL); choice y score: T = 1.0   │
└──────────────────────────────────────────────────────────────────────────────┘
```

1. **Fase 1: alineación de representaciones.** Orienta los cabezales inicializados al azar con el subespacio contextual del *backbone*. Se comparan dos modalidades: **`ce`** (entropía cruzada sin regularizar: BCE para `bool`, CE categórica para `choice`, CE *one-hot* para `score`), que maximiza rápido la separación lineal pero produce probabilidades sobreconfiadas ($T = 1.0$); y **`focal_brier`**, que optimiza desde cero la pérdida combinada y, aunque teóricamente bien calibrada, puede asentarse en mínimos locales subóptimos antes de que se establezca una separación semántica gruesa. Tamaños de lote: $B_{\text{bool}} = 32$, $B_{\text{choice}} = 8$, $B_{\text{score}} = 16$.
2. **Fase 2: refinamiento intrínseco de márgenes.** Siempre aplica pérdidas calibradas: `bool`: $\text{BCE}(p, y) + \lambda_{\text{brier}}(p-y)^2$; `choice`: $\text{CE}(p, y) + \lambda_{\text{brier}}\sum_{c=1}^N (p_c - y_c)^2$; `score`: entropía cruzada con objetivos suaves gaussianos ($\sigma = 0.35$); todas con $\lambda_{\text{brier}} = 0.5$. Dado que la función de pérdida cambia abruptamente entre fases, y AdamW mantiene estimaciones exponenciales de los momentos de primer orden ($m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$) y segundo orden ($v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$), se evalúan dos modalidades: **`reset`** (reinicio en frío de $m_t$ y $v_t$) y **`continuous`** (conservación de los momentos acumulados). Esta es la cuestión del *choque de momentum* (*momentum shock*).
3. **Fase 3: recalibración extrínseca.** Un único escalar global $T > 0$ escala los logits ($z/T$) antes de la activación. Se ajusta exclusivamente sobre `val.jsonl` ($N = 300$) mediante búsqueda en grilla y minimización escalar de Brent (ECE y NLL). Para `bool` resulta $T_{\text{bool}} \in (0.5, 0.7)$; para `choice` y `score` se fija $T = 1.0$.

### 3.4.3 Taxonomía de configuraciones experimentales

Las ocho configuraciones desacoplan las combinaciones entre fases para aislar efectos causales:

| ID | Configuración | Fase 1 | Fase 2 | Estado optimizador (Fase 2) | Fase 3 | Hipótesis puesta a prueba |
|---|---|:---:|:---:|:---:|:---:|---|
| **E1** | Fase 1 (CE clásica) | `ce` | No | — | No ($T=1.0$) | **Línea base:** exactitud cruda y sobreconfianza bajo la práctica estándar |
| **E2** | Fase 1 (CE) + Fase 3 | `ce` | No | — | Sí | **Paradigma poshoc clásico:** ¿alcanza el escalado por temperatura solo (Guo et al., 2017)? |
| **E3** | Fase 1 (Focal + Brier) | `focal_brier` | No | — | No | **Sondeo intrínseco directo:** ¿basta optimizar pérdidas calibradas desde cero? |
| **E4a** | Fase 1 (CE) + Fase 2 | `ce` | Sí (30 ép.) | `continuous` | No | **Arranque en caliente con retención de momentum** |
| **E4b** | Fase 1 (CE) + Fase 2 | `ce` | Sí (30 ép.) | `reset` | No | **Arranque en caliente con reinicio del optimizador** |
| **E5a** | Currículo completo (F1 + F2 + F3) | `ce` | Sí (30 ép.) | `continuous` | Sí | **Suite completa con continuidad del optimizador** |
| **E5b** | Currículo completo (F1 + F2 + F3) | `ce` | Sí (30 ép.) | `reset` | Sí | **Suite completa con reinicio:** currículo recomendado |
| **E6** | Pipeline directo (F1[FB] + F3) | `focal_brier` | No | — | Sí | **Línea base de producción directa** (sin preentrenamiento CE) |

## 3.5 Conjunto de datos

### 3.5.1 Generación y dominio

Se generó un corpus sintético multitarea de eventos de decisión (estado/pregunta/respuesta) para el dominio de atención al cliente (facturación, soporte técnico, ventas, cancelaciones), balanceado entre las tres primitivas y bilingüe: inglés y español latinoamericano/rioplatense, incluyendo jerga regional, cancelaciones y disputas. El generador se mantiene en `data/build_training_data.py`. Sus sucesivas ampliaciones (Capítulo 4) llevaron el corpus de 1200 a 1800 ejemplos.

### 3.5.2 Particiones y prevención de fuga de datos

El corpus de 1800 ejemplos se particionó en tres conjuntos disjuntos, almacenados en `train-dataset2/`:

| Partición | Muestras | Proporción | Distribución por primitiva | Rol metodológico |
|---|---|---|---|---|
| `train.jsonl` | 1200 | 66.67 % | 400 Bool, 400 Choice, 400 Score | **Única optimización por gradiente.** Todos los pesos $\theta = \{W_{\text{bool}}, b_{\text{bool}}, W_{\text{choice}}, W_{\text{score}}, b_{\text{score}}\}$ se actualizan con AdamW exclusivamente sobre esta partición |
| `val.jsonl` | 300 | 16.67 % | 100 Bool, 100 Choice, 100 Score | **Monitoreo y calibración posterior.** Nunca expuesta a gradientes. Se usa para seguir la pérdida/exactitud por época y ajustar la temperatura $T_{\text{bool}}$ (escalado de Platt) |
| `test.jsonl` | 300 | 16.67 % | 100 Bool, 100 Choice, 100 Score | **Evaluación ciega.** Sin exposición al entrenamiento, la optimización, el ajuste de hiperparámetros ni la calibración; se evalúa una sola vez tras el entrenamiento |

> **[Nota para revisión]** En la sección 3.6 del documento fuente se indica una partición 80 % / 10 % / 10 % para los 1800 ejemplos, mientras que la tabla de arriba (§4.1 del documento fuente) establece 1200 / 300 / 300 (66.67 % / 16.67 % / 16.67 %). Esta tesina adopta la segunda por ser la que respalda las evaluaciones reportadas; se recomienda unificar la redacción.

**Auditoría de fuga de datos.** En los ciclos tempranos, una auditoría de las particiones sintéticas heredadas reveló una vulnerabilidad crítica: las plantillas ingenuas del generador habían producido un **98 % de solapamiento de cadenas de estado** y una **tasa de 67.7 % de muestras duplicadas literalmente** entre entrenamiento y prueba. Esa fuga inflaba artificialmente las métricas hasta una exactitud aparente del 100.00 %, creando una ilusión de generalización perfecta que enmascaraba el sobreajuste agudo y los choques de momentum. En `train-dataset2/` se estableció aislamiento estricto: (1) **0.00 % de solapamiento de estados** entre particiones, de modo que cada prompt de `test.jsonl` es un texto de estado inédito; (2) cobertura balanceada de primitivas (exactamente 100 instancias de prueba por primitiva); y (3) diversidad de dominio, con representación balanceada de temas del soporte y de expresiones coloquiales.

## 3.6 Sistema de métricas

**Batería de Métricas de Decisión y Calibración.** Para garantizar una evaluación matemáticamente rigurosa y prevenir patologías de colapso de clase o sobreconfianza, OpenJEV formaliza:

1. **Primitiva booleana (`noul` / `bool`):**
   - **Métrica Primaria [P]: $F_1$-Score binario** (junto con macro $F_1$). Como media armónica entre precisión y exhaustividad, el $F_1$-score detecta de inmediato el colapso degenerado a una sola clase (donde un modelo que prediga todo ceros obtendría un 50.00 % engañoso en particiones balanceadas, pero exactamente 0.00 % de $F_1$).
   - **Métrica Secundaria [S]: Exactitud de coincidencia exacta** ($\text{Acc}_{\text{bool}}$).
2. **Primitiva de elección (`choice`):**
   - **Métrica Primaria [P]: Exactitud Top-1 ($\text{Top-1 Acc}$)** sobre el conjunto de opciones candidatas:
     $$\text{Top-1 Acc} = \frac{1}{N}\sum_{i=1}^N \mathbb{I}(\arg\max_k p_{ik} = y_i^*) \times 100\%$$
     Mide el acierto de la opción de mayor probabilidad contra la verdad de terreno en conjuntos arbitrarios de candidatos.
3. **Primitiva ordinal (`score`):**
   - **Puntuación esperada continua** $\bar{S} = \sum k \cdot p_k$, evaluada mediante **MAE continuo** ($\text{MAE}_{\text{cont}}$) y **Fallas Severas** ($\text{TER}_{1.5}$).

**Primitiva ordinal: protocolo de evaluación continua.** En la especificación normativa del sistema (`SPEC.md` §3.2), `score` no emite un entero aislado sino una distribución calibrada $\mathbf{p} = [p_0, \dots, p_{M-1}] \in \Delta^{M-1}$, de la cual se deriva analíticamente la puntuación esperada continua:

$$\bar{S} = \mathbb{E}_{\mathbf{p}}[S] = \sum_{k=0}^{M-1} k \cdot p_k \in [0.0, M-1.0]$$

Una trampa metodológica frecuente es tratar este valor continuo como una clasificación nominal discreta mediante $\hat{y}_{\text{disc}} = \arg\max_k p_k$. Esa discretización introduce distorsiones estadísticas:

1. **Penalización artificial por discontinuidad escalonada.** El $\arg\max$ es una función escalón no diferenciable. Si una consulta ambigua está en el límite semántico entre los niveles 2 y 3, un modelo bien calibrado con $p_2 = 0.49$ y $p_3 = 0.48$ produce $\bar{S} = 2.45$, lo cual refleja fielmente la incertidumbre. Un desplazamiento infinitesimal ($p_3 = 0.486$, $p_2 = 0.484$) hace saltar $\hat{y}_{\text{disc}}$ de 2 a 3 y se contabiliza un error entero completo, aunque la esperanza varió menos de 0.01.
2. **Recompensa de la sobreconfianza patológica.** Los modelos no calibrados entrenados con CE clásica ($E_1$) aprenden distribuciones degeneradas tipo Dirac ($p_{y^*} \to 1.0$). Estas pueden maximizar artificialmente la coincidencia discreta en muestras no ambiguas, pero no brindan estimaciones confiables de incertidumbre y producen errores continuos severos cuando se equivocan.

Por ello se formulan cuatro métricas continuas:

1. **Error absoluto medio continuo:** $\text{MAE}_{\text{cont}} = \frac{1}{N}\sum_{i=1}^N \lvert \bar{S}_i - y_i^* \rvert$.
2. **Raíz del error cuadrático medio continuo:** $\text{RMSE}_{\text{cont}} = \sqrt{\frac{1}{N}\sum_{i=1}^N (\bar{S}_i - y_i^*)^2}$, que penaliza con más fuerza las desviaciones grandes.
3. **Tasa de excedencia de tolerancia:** $\text{TER}_{\tau} = \frac{1}{N}\sum_{i=1}^N \mathbb{I}(\lvert \bar{S}_i - y_i^* \rvert > \tau) \times 100\%$, con tres umbrales: $\text{TER}_{0.5}$ (*desviación perceptible*), $\text{TER}_{1.0}$ (*desviación de un nivel completo*) y $\text{TER}_{1.5}$ (**fallas continuas severas**: desviaciones superiores a 1.5 niveles de rúbrica, que representan un fracaso semántico completo, por ejemplo juzgar una solicitud calmada como furia extrema).
4. **Cumplimiento de cotas ordinales:** porcentaje de casos con $\lvert \bar{S} - y^* \rvert \le 1.0$ (dentro de $\pm 1$ nivel).

**Calibración y servicio.** ECE de ancho igual y adaptativo por cuantiles, puntaje de Brier macro, latencia p50/p95 de la decisión concurrente (`noul` + `choice` + `score` en una sola pasada) y memoria pico de la GPU Metal. Se reportan también, a modo de contraste, las métricas discretas de coincidencia exacta y de fallas $\lvert\Delta_{\text{disc}}\rvert \ge 2$.

## 3.7 Software y suite de pruebas

El software desarrollado, **OpenJEV**, implementa el proceso completo. Sus componentes principales, según se mencionan en el desarrollo, son:

| Componente | Función |
|---|---|
| `core/engine_mlx.py` (`MLXInferenceEngine`) | Motor de inferencia en MLX: carga del *backbone*, empaquetado, máscara por bloques, cabezales y calibración; normalización del residuo de masa de probabilidad |
| `core/batching/packer.py` (`extract_question_text`) | Serialización de preguntas y empaquetado de secuencias (incluye la rúbrica en el texto de las preguntas `score`) |
| `core/losses.py` | Funciones de pérdida (Focal, Brier, ordinal gaussiana con log-softmax estable) |
| `core/model_scanner.py` | Detección de dimensiones y arquitectura del modelo desde `config.json` |
| `data/build_training_data.py` | Generador de datos sintéticos multitarea y bilingües |
| `scripts/train_heads.py` | Entrenamiento por lotes de cabezales con caché de representaciones latentes en memoria, currículo de tres fases y ejecución de las matrices de ablación |
| `heads.safetensors`, `calibration.json` | Artefactos exportados (cabezales ≈ 3.2 MB y temperaturas calibradas) |

La **suite de pruebas** incluye, como mínimo, la verificación de cabezales canónicos (`verify_canonical_heads`) y pruebas de regresión incorporadas al flujo de integración continua; por ejemplo, `test_commercial_pack_contracting_frustration_regression` protege de forma permanente contra la deriva sintáctica y de delimitadores descrita en la §4.9. **[Completar: cantidad de pruebas, cobertura y descripción de la suite, que el documento fuente no detalla.]**

## 3.8 Entorno experimental

Todos los experimentos se ejecutaron en GPU Metal de Apple Silicon (memoria unificada) con el framework MLX. Dado que el *backbone* está congelado, las representaciones latentes de las 1800 muestras se extrajeron una única vez (1.51 s en Qwen) y se almacenaron en memoria, por lo que los bucles internos de entrenamiento operan sobre un caché. Las matrices de ablación usaron la semilla fija `seed = 42`. Los hiperparámetros de las fases se indican en la §3.4.2.

---

# Capítulo 4. Desarrollo iterativo: patologías metodológicas y correcciones

Durante las primeras iteraciones del sistema emergieron múltiples modos de falla en las tres primitivas. Este capítulo cataloga cada uno, expone su causa matemática y empírica, y presenta la corrección verificada. Otras dos patologías, el *choque de momentum* y la omisión del token BOS (sumidero de atención), se tratan en el Capítulo 5, donde se manifestaron (§5.4 y §5.6), y la discusión sobre la métrica discreta se encuentra en la §3.6. La numeración respeta el orden en que se encontraron los problemas.

## 4.1 Patología I: la falacia del cabezal sin entrenar

* **Falla observada.** Con pesos aleatorios $W \sim \mathcal{N}(0, \sigma_0^2)$, el estado urgente *"there is a problem with my billing. you idiots i need this fixed by today"* produjo $\text{is\_urgent} = 0$ (con $P(\text{true}) = 0.189$).
* **Causa raíz.** La representación del último token de un LLM generativo contiene codificaciones semánticas ricas, pero el subespacio lineal que separa "urgente" de "no urgente" no está alineado con una proyección aleatoria arbitraria. Esta produce logits escalares arbitrarios centrados en cero, de modo que la inferencia *zero-shot* es puramente estocástica.
* **Corrección.** Se implementó el ajuste exclusivo de cabezales (sondeo lineal) con el *backbone* estrictamente congelado y, para evitar la sobreconfianza, un objetivo que acopla la entropía cruzada binaria con una penalización de Brier: $\mathcal{L}_{\text{bool}} = \text{BCE}(p, y) + \lambda_{\text{calib}}(p - y)^2$, con $\lambda_{\text{calib}} = 0.5$.

## 4.2 Patología II: anisotropía de representaciones y explosión de logits en cabezales bilineales de elección

* **Falla observada.** En evaluaciones tempranas de enrutamiento entre departamentos (`billing`, `technical`, `sales`), el cabezal asignaba probabilidad abrumadora ($> 93\%$) a una clase incorrecta (por ejemplo, `sales` en lugar de `billing`), pese a la clara superposición semántica con facturación.
* **Traza diagnóstica.** Las normas de los embeddings crudos eran grandes: $\lVert h_Q \rVert_2 \approx 280.4$ y $\lVert e_{C_i} \rVert_2 \approx 265.1$. El producto escalar sin normalizar arrojó $\langle h_Q, e_{C_i} \rangle \approx 74\,800$; al dividir por $\sqrt{d} = \sqrt{896} \approx 29.93$ los logits resultaron $z_i \approx 2500$. En aritmética de punto flotante, aplicar *softmax* a logits separados incluso por márgenes relativos modestos ($z_1 = 2500$, $z_2 = 2470$) produce saturación numérica instantánea ($\mathbf{p} = [1.0, 0.0, 0.0]$ o $[0.0, 0.0, 1.0]$). Es una manifestación de la **anisotropía de representaciones** ("efecto cono", Ethayarajh, 2019), donde los productos escalares sin normalizar miden la magnitud del vector en lugar de la alineación semántica direccional.
* **Corrección.** (1) Proyección residual $h_{\text{proj}} = h_Q + W_{\text{choice}} h_Q$. (2) Normalización $\ell_2$ estricta de la consulta y de los candidatos: $\hat{h} = h_{\text{proj}} / (\lVert h_{\text{proj}} \rVert_2 + \epsilon)$ y $\hat{e}_i = e_{C_i} / (\lVert e_{C_i} \rVert_2 + \epsilon)$. (3) Escalado de la similitud coseno con una temperatura contrastiva $\tau = 0.03$: $z_i = \langle \hat{h}, \hat{e}_i \rangle / \tau$. Esto acota estrictamente los logits a $z_i \in [-33.33, +33.33]$, restablece un flujo de gradiente suave y elimina la saturación del *softmax*.

## 4.3 Patología III: recorte artificial del rango métrico en la puntuación ordinal

* **Falla observada.** Al evaluar la frustración con una rúbrica de 3 niveles (0: calma, 1: frustrado, 2: muy enojado), un estado de furia evidente produjo $\mathbf{p} = [0.02, 0.00, 0.98]$ y, sin embargo, el motor emitió $\text{score} = 0.99$.
* **Causa raíz.** Una implementación heredada en `core/engine_mlx.py` normalizaba la puntuación: $\text{score}_{\text{emitido}} = \sum_i i \cdot p_i / (M-1)$, forzándola al intervalo $[0, 1]$. Para $M = 3$, una puntuación verdadera de 2.0 se comprimía a 1.0, lo que volvía el valor inconsistente con la escala de la rúbrica definida por el usuario.
* **Corrección.** Se eliminó la división: el motor emite la puntuación esperada continua directamente en el intervalo nativo $[0, M-1]$, $\bar{S} = \sum_{i=0}^{M-1} i \cdot p_i$.

## 4.4 Patología IV: inanición de información por desatención a la rúbrica

* **Falla observada.** Ante el prompt rioplatense *"Son unos soretes. Dene de baja de su servicio de mierda ya"*, el modelo asignó solo 0.5 % de probabilidad al nivel 2 ("muy enojado, lenguaje fuerte") y 72.3 % al nivel 1 ("frustrado pero cordial"), con una puntuación de 0.73 y confianza 0.7233.
* **Análisis de causa raíz.** El examen de `core/batching/packer.py:extract_question_text` reveló que, para las preguntas `score`, la secuencia enviada al tokenizador consistía estrictamente en la cadena de instrucción (*"Determine frustration level of the user"*). Los niveles de la rúbrica definidos en la solicitud (*"Level 0: Calm..."*, *"Level 1: Frustrated..."*, *"Level 2: Very angry, strong language"*) **nunca llegaban al tokenizador**. En consecuencia, el *backbone* era completamente ciego a lo que significaban los índices 0, 1 y 2, y el cabezal lineal $W_{\text{score}} \in \mathbb{R}^{M \times d}$ debía aprender las correspondencias semánticas desde cero, sin ninguna atención cruzada entre las palabras del cliente y los descriptores de la rúbrica.
* **Corrección.** Se modificó `extract_question_text` para incorporar explícitamente los criterios de la rúbrica a la secuencia evaluada: $\text{Texto} = \text{instrucciones} + \text{". Rubric: "} + \sum_{i=0}^{M-1}\text{"Level "} + i + \text{": "} + \text{criterio}[i]$. De este modo, las capas de autoatención pueden atender directamente entre los tokens soeces (*"soretes"*, *"mierda"*) y el descriptor *"Level 2: Very angry, strong language"*.

## 4.5 Patología V: ancho de banda gaussiano excesivo que diluye los objetivos extremos

* **Falla observada.** Aun con la rúbrica incorporada, el cabezal seguía colocando una masa de probabilidad sustancial ($> 30\%$) en el nivel 1 cuando la muestra estaba etiquetada con el nivel 2.
* **Causa raíz matemática.** La pérdida ordinal usaba el suavizado gaussiano de etiquetas (§2.7) con ancho de banda por defecto $\sigma = 1.0$. Para $M = 3$ y $y^* = 2$, las distancias cuadráticas son $[4, 1, 0]$ y los pesos sin normalizar son $w_0 = e^{-2} \approx 0.1353$, $w_1 = e^{-0.5} \approx 0.6065$ y $w_2 = 1.0000$ (suma 1.7418). Los objetivos normalizados resultan:

| Nivel | Cálculo | Probabilidad objetivo |
|---|---|---|
| 0 | 0.1353 / 1.7418 | 7.77 % |
| 1 | 0.6065 / 1.7418 | **34.82 %** |
| 2 | 1.0000 / 1.7418 | **57.41 %** |

  Bajo la optimización de $\mathcal{L} = -\sum_i \tilde{y}_i \log p_i$, la predicción óptima es $p_i \equiv \tilde{y}_i$; por lo tanto, la función de pérdida **penalizaba matemáticamente al modelo si no asignaba 34.8 % de probabilidad al nivel 1 en una muestra de nivel 2**. El suavizado era tan amplio que destruía la distinción entre niveles adyacentes.
* **Corrección analítica.** Se recalibró el ancho de banda a **$\sigma = 0.35$** ($2\sigma^2 = 0.245$): $w_0 = e^{-16.33} \approx 8.1 \times 10^{-8}$, $w_1 = e^{-4.082} \approx 0.01688$, $w_2 = 1.0000$ (suma ≈ 1.01688). Los objetivos pasan a ser $\tilde{y}_0 \approx 0.00\%$, $\tilde{y}_1 = 1.66\%$ y $\tilde{y}_2 = 98.34\%$. Con $\sigma = 0.35$, el 98.34 % de la masa del gradiente se concentra en el nivel verdadero, conservando un decaimiento suave por distancia para penalizar errores ordinales (la ablación está en la §5.3).

## 4.6 Patología VI: vulnerabilidad fuera de distribución ante jerga coloquial regional

* **Falla observada.** Ante *"Son unos soretes. Dene de baja de su servicio de mierda ya"*, el cabezal de elección enrutó la solicitud a `sales` (60.69 %) en lugar de `billing` (37.93 %).
* **Causa raíz.** El conjunto de entrenamiento (`data/build_training_data.py`) contenía exclusivamente muestras corporativas estándar de soporte en inglés (*"My credit card was charged twice"*, *"Invoice request"*). En español, *"dene de baja de su servicio"* denota la cancelación de una suscripción (que corresponde a `billing`), pero el embedding preentrenado de Qwen para *"servicio"* no estaba anclado a las bajas de facturación en el español latinoamericano dialectal. Sin muestras de cancelación en español, el vector de proyección de `billing` no se alineaba con los modismos.
* **Corrección.** Se amplió el generador sintético multitarea con datos bilingües balanceados (inglés + español latinoamericano/rioplatense), incorporando jerga regional, cancelaciones y disputas. Positivos para facturación/cancelación: *"Dene de baja de su servicio"*, *"Quiero dar de baja mi suscripción inmediatamente"*, *"Son unos ladrones y unos estafadores"*. Negativos para facturación/urgencia: *"Hola, una consulta tranquila sobre los planes"*, *"Buenas tardes, ¿cómo descargo la factura?"*. El corpus total pasó de 1200 a 1800 ejemplos.

## 4.7 Patología VII: estancamiento de la optimización y procesamiento por lotes vectorizado en GPU

* **Falla observada.** Una implementación inicial en Python iteraba muestra por muestra:

  ```python
  # Implementación secuencial ingenua
  for h_q, e_opts, target in sample_list:
      sel, conf, probs = model(h_q, e_opts)
      loss = loss_fn(probs, target)
      total_loss += loss
  ```

  Con 20 épocas sobre 480 muestras de entrenamiento, AdamW realizaba solo 20 pasos de actualización. La pérdida del cabezal `score` se estancó en $\mathcal{L} = 2.05$ y la exactitud de validación fue pobre (51.67 %). Además, al pasar vectores ocultos sin escalar ($h_Q \in \mathbb{R}^{896}$, norma ≈ 30) directamente a la pérdida por lotes, los gradientes explotaron (la pérdida saltó a 14.62 y se saturó).
* **Corrección.** (1) Agrupar los datos en lotes homogéneos estratificados por cantidad de opciones $N$ (`choice`) y de niveles de rúbrica $M$ (`score`). (2) Vectorizar las pasadas hacia adelante en GPU Metal (`sel, conf, probs = model(h_q_b, e_opts_b)` con $h_{q,b}$ de forma $(B, d)$ y $e_{\text{opts},b}$ de forma $(B, N, d)$). (3) Elevar las épocas a 60 con tasa de aprendizaje $\eta = 10^{-3}$ y decaimiento de pesos de AdamW $\lambda = 10^{-2}$. (4) El entrenamiento vectorizado por lotes se completó en **26.17 segundos**.

## 4.8 Patología VIII: conflación léxica dominio-sentimiento y saturación de gradiente cero en la puntuación ordinal

* **Falla observada.** Ante una consulta comercial cortés sobre precios (*"would it be possible to eventually reduce the cost of my active subscription to your service?"*), el modelo resolvió correctamente `department = sales` (80.68 %) e `is_urgent = 0`, pero emitió una clasificación severamente errónea de `frustration`: **92.27 %** de la masa en el nivel 2 (muy enojado) y una puntuación continua de **1.90 / 2.00**, para una consulta de redacción deferente y no confrontativa.
* **Causa raíz (dos factores).**
  1. **Conflación léxica de dominio en los datos de entrenamiento.** En `SCORE_SAMPLES_FRUSTRATION`, las muestras del nivel 0 (*"Calm, just stating facts"*) eran exclusivamente prompts técnicos de ingeniería (*"API rate limit"*, *"log snippet"*, *"port number"*). En cambio, palabras comerciales como `subscription`, `cost`, `service`, `billing` o `charged` aparecían solo en los niveles 1 y 2 (quejas airadas o reclamos de contracargo). Al estar ausentes los tokens del dominio comercial en el nivel 0, el sondeo lineal aprendió una correlación espuria y no causal que trataba las palabras de cuentas comerciales como indicadores de enojo, en lugar de evaluar el tono cortés.
  2. **Saturación numérica de gradiente cero en `ordinal_score_loss`.** La pérdida original (`core/losses.py`) era $\mathcal{L}_{\text{score}} = -\sum_i \tilde{y}_i \log(\text{clip}(p_i, 10^{-9}, 1.0))$ con $p_i = \text{Softmax}(z)_i$. Como las representaciones sin normalizar tienen normas $\ell_2$ grandes ($\lVert h_Q \rVert_2 \approx 300$), los logits se separan con rapidez ($\Delta z > 21$) y, para las clases no modales, $p_i$ caía por debajo del umbral $10^{-9}$. En diferenciación automática, la derivada de un valor recortado es idénticamente **0.0**: $\partial\,\text{clip}(p_i, 10^{-9}, 1.0)/\partial z_j = 0$ cuando $p_i \le 10^{-9}$. El gradiente de la clase verdadera se desvanecía por completo y producía una meseta de pérdida alta ($\mathcal{L} \approx 10.5$) donde AdamW quedaba bloqueado en saturación y la exactitud de validación colapsaba a 55.56 %.
* **Intervenciones correctivas.**
  1. **Desacople de dominio y muestreo balanceado por nivel.** Se ampliaron las muestras con solicitudes corteses de suscripción, precios, descuentos, facturas y cambios de plan en los niveles 0, 1 y 2, en inglés y español, y se refactorizó `generate_score_dataset` para imponer un muestreo estrictamente igual entre niveles de la rúbrica, eliminando el enredo léxico entre tema y sentimiento.
  2. **Formulación estable de log-softmax.** Se reemplazó $\log(\text{clip}(\text{softmax}(z)))$ por la identidad exacta $\log p_i = z_i - \text{logsumexp}(z)$, implementada con `inputs - mx.logsumexp(inputs, axis=-1, keepdims=True)` y `from_logits=True`. Esto provee gradientes de error $(p_i - \tilde{y}_i)$ suaves y no evanescentes sin importar la separación de logits; la pérdida convergió de 4.25 a 0.1925 en 100 épocas.
* **Resultados empíricos.** En la consulta de referencia, la probabilidad del nivel 0 pasó de **1.79 % a 75.01 %**, la puntuación continua de **1.90 a 0.32** y la probabilidad del nivel 2 de **92.27 % a 7.49 %**. La exactitud de validación del cabezal `score` pasó de 55.56 % a **98.15 %**.

## 4.9 Patología IX: fusión de delimitadores BPE por espacios en blanco finales y sesgo imperativo transaccional en el sondeo ordinal de sentimiento

* **Falla observada.** Ante la transacción comercial directa *"quisiera contratar el pack HBO "*, el modelo emitió una puntuación de frustración elevada de **0.91** (escala $[0, 2]$) con distribución $\{0: 43.84\%,\ 1: 21.15\%,\ 2: 35.01\%\}$, es decir, **56.16 %** de masa en irritación emocional para una intención rutinaria de contratar un paquete de entretenimiento.
* **Causa raíz (dos factores técnicos).**
  1. **Fusión de tokens delimitadores BPE en las uniones del prompt.** La consulta contenía un espacio en blanco al final (`"pack HBO "`). Durante el empaquetado, el estado se concatenaba directamente con el prefijo de pregunta ($X = [S \circ Q_k]$, con $Q_k$ iniciando en `"\n\nQuestion: "`). En vocabularios BPE estándar (como el de 151k tokens de Qwen), los espacios en blanco en los límites de token no se codifican como tokens de espacio aislados: el espacio final se fusiona vorazmente con los bytes de salto de línea siguientes y la secuencia produce el token ID 4710 en lugar del ID 271 (`"\n\n"`). Esa fusión alteró la representación subléxica del límite del prompt, desplazó la codificación posicional y los pesos de autoatención en la unión e inyectó un límite de token fuera de distribución en el vector agrupado $h_Q$.
  2. **Sesgo de selección sintáctico-pragmático.** Tras resolver la patología VIII, las muestras de nivel 0 consistían exclusivamente en consultas deferentes sintácticamente complejas con marcadores de cortesía (*"¿Sería posible consultar por una reducción..."*, *"Hola equipo, muchas gracias por su servicio..."*). Los enunciados transaccionales imperativos directos (*"quisiera contratar el pack HBO"*, *"quiero contratar el servicio"*) y los acrónimos comerciales (*"HBO"*, *"pack fútbol"*) estaban ausentes del nivel 0. Como los enunciados breves y directos, sin saludos corteses, aparecían casi solo en cancelaciones airadas (*"cancelen ya"*, *"son unos ladrones"*), la matriz $W_{\text{score}}$ desarrolló una heurística espuria que asociaba la ausencia de saludos y la presencia de palabras transaccionales con brusquedad y agitación.
* **Intervenciones correctivas.**
  1. **Saneamiento estricto de espacios en blanco en el límite de entrada.** $\tilde{S} = \text{serialize\_state}(S)\text{.strip()}$. Así se garantiza una tokenización subléxica invariante en las uniones, y los delimitadores (`\n\nQuestion:`) se resuelven siempre en los tokens canónicos de salto de línea (ID 271) sin importar el formato de entrada del usuario.
  2. **Aumento balanceado de imperativos transaccionales y acrónimos.** Se ampliaron las muestras con solicitudes comerciales directas, sin prefacio, en los tres niveles: *Nivel 0 (calma):* *"quisiera contratar el pack HBO"*, *"quiero contratar el servicio de internet"*, *"solicito contratar el pack de deportes"*; *Nivel 1 (frustrado):* demoras sin groserías (*"pedí la contratación del pack HBO hace tres días y todavía no tengo los canales"*); *Nivel 2 (enojado):* reclamos indignados con groserías (*"me cobran el pack HBO todos los meses y no anda nunca, son unos estafadores!"*).
  3. **Normalización estricta de masa residual.** Se implementó una compensación posterior al redondeo en `core/engine_mlx.py`, de modo que las distribuciones discretas de todas las primitivas sumen exactamente $1.0000$, eliminando la deriva decimal bajo redondeo a cuatro dígitos.
* **Verificación empírica.** Tras reentrenar con el corpus saneado, la consulta `"quisiera contratar el pack HBO "` quedó resuelta: en **Qwen 2.5-0.5B**, la puntuación de frustración cayó de **0.91 a 0.37**, la probabilidad del nivel 0 subió de **43.84 % a 77.99 %**, la del nivel 2 bajó a 14.57 % y el enrutamiento resolvió con decisión a `sales` (98.66 %); en **Gemma 2 2B** la puntuación fue de **0.08**, con nivel 0 en **92.76 %** y nivel 2 en 1.24 %. Se incorporó a la suite de integración continua la prueba de regresión `test_commercial_pack_contracting_frustration_regression`.

## 4.10 Patología X: asimetría estructural por capa y fricciones de inicialización en *backbones* no uniformes para dispositivos

* **Síntoma.** Al extender el marco de decisión desde arquitecturas uniformes hacia **OpenELM-450M** ($d = 1536$), la carga y el entrenamiento fallaron por tres fricciones arquitectónicas y de bibliotecas:
  1. **Divergencia de claves de configuración.** `config.json` usa `"model_dim": 1536` y no `"hidden_size"` ni `"d_model"`. El escáner de modelos asumía por defecto $d = 896$, lo que causaba excepciones de dimensión (`verify_canonical_heads` lanzaba `ValueError: expected 896, got 1536`).
  2. **Fricción de argumentos de *dataclass* de código remoto.** La ejecución de código remoto de Hugging Face (`configuration_openelm.py`) fallaba en versiones modernas de `transformers` con `TypeError: OpenELMConfig.__post_init__() got an unexpected keyword argument 'use_cache'`.
  3. **Polimorfismo del contenedor del *backbone*.** En MLX, los modelos decodificadores estándar (`Qwen2Model`, `Gemma2Model`) exponen sus capas en `model.model.layers`, mientras que OpenELM las define en `model.transformer.layers`, causando `AttributeError: 'Model' object has no attribute 'model'`.
* **Causa raíz.** OpenELM se aparta de la uniformidad del Transformer estándar mediante una asignación de parámetros variable por capa: las cabezas de consulta escalan de 12 a 24, las KV de 3 a 6 y los multiplicadores FFN de 0.5× a 4.0×. La implementación MLX de Apple encapsula la secuencia de bloques bajo `self.transformer` y no `self.model`; además, el script de configuración remota histórico pasaba argumentos `use_cache` no consumidos que los inicializadores estrictos de *dataclass* de `huggingface_hub` modernos rechazan.
* **Intervención correctiva.** (1) **Generalización del escáner de dimensiones:** `core/model_scanner.py` evalúa en cortocircuito $\text{dim} = \text{data.get("hidden\_size")} \lor \text{data.get("d\_model")} \lor \text{data.get("model\_dim")}$, compatible hacia atrás con Qwen y Gemma. (2) **Despacho polimórfico del *backbone*:** una escalera de atributos resiliente en inferencia (`core/engine_mlx.py`) y entrenamiento (`scripts/train_heads.py`): $\text{backbone} = \text{getattr}(\text{model}, \text{"model"}, \text{getattr}(\text{model}, \text{"transformer"}, \bot))$. (3) **Resolución nativa y saneada del tokenizador:** se evitaron los scripts remotos frágiles con `trust_remote_code=False` y un `config.json` sin `"auto_map"`, de modo que `transformers` resuelve de forma nativa a `LlamaTokenizer` (vocabulario de 32 000) y MLX instancia `mlx_lm.models.openelm` directamente.

## 4.11 Síntesis de las patologías

| N.º | Patología | Causa raíz | Corrección |
|---|---|---|---|
| I | Cabezal sin entrenar | Proyección aleatoria no alineada con el subespacio de la tarea | Sondeo lineal con BCE + Brier |
| II | Anisotropía y explosión de logits en `choice` | Productos escalares sin normalizar ($z \approx 2500$) | Normalización $\ell_2$, residual y $\tau = 0.03$ |
| III | Recorte del rango de `score` | División por $M-1$ | Emitir $\bar{S}$ en $[0, M-1]$ |
| IV | Inanición por rúbrica ausente | La rúbrica no llegaba al tokenizador | Incorporar la rúbrica al texto de la pregunta |
| V | Suavizado gaussiano excesivo | $\sigma = 1.0$ ponía 34.8 % de masa en el nivel vecino | $\sigma = 0.35$ |
| VI | Jerga regional fuera de distribución | Datos solo en inglés corporativo | Datos bilingües balanceados (1200 → 1800) |
| VII | Estancamiento por optimización secuencial | Pocas actualizaciones y gradientes sin escalar | Lotes vectorizados estratificados en GPU |
| VIII | Conflación dominio-sentimiento y gradiente cero | Sesgo léxico y *clipping* en la pérdida | Muestreo balanceado y log-softmax estable |
| IX | Fusión BPE y sesgo imperativo | Espacio final + ausencia de imperativos en nivel 0 | `.strip()` y aumento transaccional |
| X | Asimetría estructural en OpenELM | Claves de configuración y módulos distintos | Escáner generalizado y despacho polimórfico |

---

# Capítulo 5. Resultados

## 5.1 Dinámica de entrenamiento y evaluación ciega en Qwen 2.5-0.5B

### 5.1.1 Convergencia en la partición de validación

Resueltas las patologías del Capítulo 4, los cabezales se entrenaron por lotes sobre las representaciones de `train.jsonl` ($N = 1200$) con el currículo de dos fases y reinicio del optimizador. La progresión y el rendimiento medido sobre `val.jsonl` ($N = 300$) fueron:

| Primitiva | Arquitectura del cabezal | Formulación de la pérdida | Pérdida inicial | Pérdida final (Fase 2) | Rendimiento en validación |
|---|---|---|---|---|---|
| **Noul (booleana)** | Lineal ($896 \to 1$) | Focal-Brier combinada | 0.6931 | **0.0147** | **96.00 %** (exactitud) |
| **Choice** | Bilineal residual ($896 \to 896$) | CE con softmax y temperatura + Brier | 1.0986 | **0.1282** | **96.00 %** (exactitud) |
| **Score** | Lineal ($896 \to 5$) | Pérdida ordinal numéricamente estable ($\sigma = 0.35$) | 4.2516 | **0.4989** | **0.389** (MAE continuo) |

En la partición de validación, el cabezal `score` alcanza una coincidencia exacta de nivel de 74.00 % bajo $\arg\max$ discreto, con predicciones a un nivel de distancia en el 19.00 % de los casos y ninguna falla continua severa ($\lvert \bar{S} - y^* \rvert > 1.5$), lo que confirma transiciones ordinales suaves y estables.

### 5.1.2 Evaluación independiente sobre la partición de prueba ciega

Para verificar la generalización genuina sobre estados inéditos, se evaluaron los 300 ejemplos de `test.jsonl` a través del motor completo de inferencia (`MLXInferenceEngine`) con el currículo calibrado completo ($E_{5b}$):

| Primitiva | Métrica | Instancias | Desempeño (`test.jsonl`, $N = 300$) |
|---|---|---|---|
| **Noul (booleana)** | Coincidencia binaria exacta | 96 / 100 | **96.00 %** |
| **Choice** | Coincidencia exacta de categoría | 96 / 100 | **96.00 %** |
| **Score** | MAE continuo esperado | 100 | **0.389** |
| **Score** | RMSE continuo esperado | 100 | **0.516** |
| **Score** | Dentro de $\pm 1$ nivel ($\lvert \bar{S} - y^* \rvert \le 1.0$) | 93 / 100 | **93.00 %** |
| **Score** | Fallas severas ($> 1.5$ niveles) | 0 / 100 | **0.00 %** |
| **Score** | Coincidencia discreta exacta ($\arg\max_k p_k = y^*$) | 74 / 100 | **74.00 %** |
| **Calibración** | ECE macro (ancho igual) | 200 evaluaciones | **5.27 %** |
| **Calibración** | Puntaje de Brier macro | 200 evaluaciones | **0.0780** |

El 0.00 % de errores continuos severos, junto con un MAE continuo de 0.389 y un ECE de 5.27 %, confirma que el currículo calibrado evita predicciones puntuales tipo Dirac sobreconfiadas y produce distribuciones de probabilidad bien comportadas.

## 5.2 Estudio de caso: el prompt coloquial adversarial

Se evaluó el impacto de las correcciones metodológicas siguiendo la distribución de salida para la consulta adversarial de referencia (*"Son unos soretes. Dene de baja de su servicio de mierda ya"*) a lo largo de las fases de desarrollo. El esquema evaluado fue: (1) `department` (`choice`): `billing` ("Invoices, payments, refunds, subscription cancellations"), `technical` ("Bugs, outages, login issues"), `sales` ("Pricing, new plans, enterprise inquiries"); (2) `frustration` (`score`) con niveles 0 ("Calm, just stating facts"), 1 ("Frustrated but civil") y 2 ("Very angry, strong language"); y (3) `is_urgent` (`noul`).

| Fase de evaluación | Decisión de departamento | Puntuación de frustración ($\bar{S}$) | Probabilidades de frustración | Decisión de urgencia |
|---|---|---|---|---|
| **Línea base (Fase 1: sin entrenar)** | `sales` (93.39 %) ❌ | 0.99 (escala $[0, 1]$) ❌ | $\{0: 2\%, 1: 0\%, 2: 98\%\}$ | `0` (falso) ❌ |
| **Fase 2: entrenado solo en inglés** | `sales` (60.69 %) ❌ | 0.73 (nivel 1) ❌ | $\{0: 27.2\%, 1: 72.3\%, 2: 0.5\%\}$ ❌ | `1` (verdadero) ✅ |
| **Fase 3: rúbrica incorporada al prompt** | `billing` (74.47 %) ✅ | 1.62 (nivel 2) ✅ | $\{0: 13.6\%, 1: 10.5\%, 2: 75.9\%\}$ ✅ | `1` (verdadero) ✅ |
| **Final: totalmente calibrado ($\sigma = 0.35$, 60 ép.)** | **`billing` (90.34 %)** ✅ | **1.96 (nivel 2)** ✅ | **$\{0: 1.2\%, 1: 1.9\%, 2: 96.9\%\}$** ✅ | **`1` (verdadero)** ✅ |

Distribución final calibrada: `billing` 90.34 %, `sales` 8.81 %, `technical` 0.85 %; frustración: nivel 0 = 1.17 %, nivel 1 = 1.88 %, nivel 2 = 96.95 % (puntuación esperada 1.96 / 2.00).

### 5.2.1 Generalización a arquetipos representativos de consulta

Para descartar que el modelo hubiera memorizado la frase puntual, se evaluó el modelo final sobre arquetipos semánticos distintos:

| N.º | Arquetipo | Estado | Departamento | Frustración | Urgente |
|---|---|---|---|---|---|
| 1 | Disputa de facturación con alta ira | *"Son unos ladrones y unos estafadores, me robaron la plata en la cara!"* | `billing` (94.36 %) | 1.98 · {0: 0.18 %, 1: 1.37 %, 2: 98.45 %} | 1 |
| 2 | Disgusto con alta ira | *"Es una porquería de servicio, son unos inútiles de mierda!"* | `billing` (79.29 %) | 1.98 · {0: 0.06 %, 1: 1.59 %, 2: 98.34 %} | 1 |
| 3 | Queja técnica cordial | *"Hace 3 días que estoy intentando configurar la cuenta y sigue dando error."* | `technical` (99.68 %) | 0.93 · {0: 7.73 %, 1: 91.78 %, 2: 0.48 %} | 0 |
| 4 | Consulta comercial calmada | *"Hola, buenos días, una consulta sobre los precios de los planes anuales."* | `sales` (83.30 %) | 0.26 · {0: 74.69 %, 1: 24.99 %, 2: 0.32 %} | 0 |
| 5 | Consulta cortés de suscripción (desacople dominio-sentimiento) | *"would it be possible to eventually reduce the cost of my active subscription to your service?"* | `sales` (94.66 %) | 0.32 · {0: 75.01 %, 1: 17.50 %, 2: 7.49 %} | 0 |
| 6 | Contratación directa de un pack (invariancia sintáctica) | *"quisiera contratar el pack HBO "* | `sales` (98.66 %) | 0.37 · {0: 77.99 %, 1: 7.44 %, 2: 14.57 %} | 0 |

Las salidas confirman una discriminación semántica nítida y confiable, tanto en sentimiento como en enrutamiento de dominio, sin diafonía entre tareas.

## 5.3 Ablación del ancho de banda gaussiano $\sigma$

Para evaluar rigurosamente el impacto del ancho de banda del suavizado de etiquetas en `ordinal_score_loss`, se realizó un estudio de ablación sobre $\sigma \in \{0.20, 0.35, 0.50, 1.00\}$:

| Ancho de banda $\sigma$ | Masa en el objetivo verdadero ($y^* = 2$) | Masa en el vecino ($y = 1$) | Coincidencia de nivel en validación | Estabilidad de convergencia |
|---|---|---|---|---|
| $\sigma = 0.20$ | 99.99 % | 0.01 % | 81.67 % | Propenso a desvanecimiento de gradiente ante errores lejanos |
| **$\sigma = 0.35$** | **98.34 %** | **1.66 %** | **86.67 %** | **Óptimo: objetivos nítidos + preservación ordinal** |
| $\sigma = 0.50$ | 88.08 % | 11.92 % | 78.33 % | Dilución de probabilidad notable en clases extremas |
| $\sigma = 1.00$ | 57.41 % | 34.82 % | 51.67 % | Subpredicción severa de niveles extremos |

La ablación confirma que $\sigma = 0.35$ ofrece el mejor equilibrio entre sensibilidad a la distancia ordinal y nitidez categórica.

## 5.4 Ablación de la metodología de entrenamiento: matriz desacoplada de calibración (Qwen 2.5-0.5B, $d = 896$)

Para establecer la contribución individual de cada componente del ciclo de entrenamiento de los cabezales, se ejecutó una matriz controlada de ablación con las ocho configuraciones de la §3.4.3 sobre `Qwen2.5-0.5B`, con *backbone* congelado, caché de representaciones latentes en memoria y semilla fija (`seed = 42`).

### 5.4.1 Resultados sobre la partición de prueba ciega (`test.jsonl`, $N = 300$, 100 por primitiva)

| ID | Configuración | Optim. F2 | F1 Bool (%) [P] | Exact. Bool (%) [S] | Top-1 Choice (%) [P] | MAE cont. Score | Severas Score ($> 1.5$) | ECE (%) [ancho igual] | ECE (%) [adaptativo] | Brier |
|---|---|---|---|---|---|---|---|---|---|---|
| **E1** | Fase 1 (CE clásica) | — | 93.33 | 93.00 | 79.00 | 0.432 | 2.00 % | 15.25 | 13.98 | 0.2245 |
| **E2** | Fase 1 (CE) + Fase 3 | — | 93.33 | 93.00 | 79.00 | 0.432 | 2.00 % | 15.00 | 13.40 | 0.2206 |
| **E3** | Fase 1 (Focal + Brier) | — | 93.88 | 94.00 | 72.00 | 0.437 | 2.00 % | 13.27 | 13.93 | 0.2417 |
| **E4a** | Fase 1 (CE) + Fase 2 | Continuo | 96.08 | 96.00 | 85.00 | 0.411 | 1.00 % | 11.28 | 10.10 | 0.1635 |
| **E4b** | Fase 1 (CE) + Fase 2 | Reinicio | 96.08 | 96.00 | 96.00 | **0.389** | **0.00 %** | 7.04 | 6.07 | 0.0789 |
| **E5a** | Currículo completo (F1 + F2 + F3) | Continuo | 96.08 | 96.00 | 85.00 | 0.411 | 1.00 % | 10.33 | 9.27 | 0.1614 |
| **E5b** | Currículo completo (F1 + F2 + F3) | Reinicio | **96.08** | **96.00** | **96.00** | **0.389** | **0.00 %** | **5.27** | **5.47** | **0.0780** |
| **E6** | Pipeline directo (F1[FB] + F3) | — | 93.88 | 94.00 | 72.00 | 0.437 | 2.00 % | 12.79 | 13.41 | 0.2399 |

### 5.4.2 Análisis de la matriz desacoplada

1. **Resolución empírica del dilema del choque de momentum (continuo vs. reinicio).** La comparación **E4a** vs. **E4b** y **E5a** vs. **E5b**, sobre un banco de pruebas estrictamente no contaminado, resuelve la cuestión del optimizador en arranque en caliente. Conservar los momentos de AdamW (**`continuous`**) mantiene la inercia direccional del paisaje de CE sin calibrar; al pasar a Focal + Brier, esa inercia resiste la recalibración de fronteras, limitando la exactitud de `choice` a 85.00 % y dejando elevado el error de calibración (ECE = 10.33 % y Brier = 0.1614 en E5a). Reinicializar el estado del optimizador (**`reset`**) con una tasa de aprendizaje atenuada elimina el arrastre direccional: la pérdida calibrada afila los márgenes desde una pizarra limpia, la exactitud de `choice` sube a **96.00 %**, el ECE se reduce a la mitad (**5.27 %**) y el Brier baja a **0.0780** (reducción de error $> 65\%$ respecto de la línea base E1 y del arranque continuo E5a).
2. **Puntuación esperada continua vs. artefacto de discretización escalonada.** Evaluar los cabezales ordinales con $\arg\max$ discreto introduce una penalización escalonada artificial: un modelo calibrado que distribuye masa entre niveles adyacentes ($p_2 = 0.48$, $p_3 = 0.47$, $\bar{S} = 2.42$) es penalizado como error entero ($\lvert 3 - 2 \rvert = 1$) cada vez que el $\arg\max$ cruza la frontera de 0.50. Para demostrarlo, se evaluó el conjunto de prueba ($N = 100$) con ambos paradigmas:

   | Paradigma | Métrica | Línea base (E1: CE) | Arranque continuo (E4a) | Currículo calibrado (E5b: reinicio) | Efecto metodológico observado |
   |---|---|:---:|:---:|:---:|:---:|
   | **Discreto** ($\arg\max$) | Exactitud discreta ($\Delta_{\text{disc}} = 0$) | 78.00 % | 72.00 % | 74.00 % | $-4.00\%$ (artefacto engañoso de discretización) |
   | | Fallas severas discretas ($\lvert \Delta_{\text{disc}} \rvert \ge 2$) | 3.00 % | 8.00 % | 7.00 % | $+4.00\%$ (falsa penalización en colas suaves) |
   | **Continuo** ($\bar{S}$) | **MAE continuo** | 0.432 | 0.411 | **0.389** | **−10.0 %** de error neto |
   | | **RMSE continuo** | 0.566 | 0.540 | **0.516** | **−8.8 %** de varianza |
   | | **$\text{TER}_{0.5}$** ($\lvert \bar{S} - y^* \rvert > 0.5$) | 37.00 % | 34.00 % | **31.00 %** | **−6.00 %** de fallas de frontera |
   | | **$\text{TER}_{1.5}$** (fallas continuas severas) | 2.00 % | 1.00 % | **0.00 %** | **100 %** de eliminación de errores severos |

   Medido de forma continua contra la verdad $y^*$, el currículo calibrado (**E4b/E5b**) supera con claridad a la CE sin calibrar (**E1**): el MAE continuo cae de 0.432 a **0.389** (−10.0 %) y las fallas continuas severas bajan de 2.00 % a **0.00 %** en las 100 muestras de prueba.
3. **El currículo en arranque en caliente supera al entrenamiento directo de una etapa.** Comparar **E5b** con **E6** muestra la importancia del aprendizaje por currículo: preasignar las representaciones semánticas con CE gruesa y luego refinar probabilidades con Focal + Brier logra una exactitud de enrutamiento mucho mayor (96.00 % vs. 72.00 %) y una calibración más ajustada (ECE = 5.27 % vs. 12.79 %) que entrenar desde cero con pérdidas combinadas no convexas.
4. **Ineficacia de la calibración poshoc aislada en LLM compactos (Guo et al., 2017).** La comparación **E1** vs. **E2** revela los límites del escalado por temperatura escalar: cuando las representaciones pequeñas sufren distorsiones de margen no lineales y dependientes de la muestra, escalar todos los logits con un único $T$ casi no modifica el error de calibración (ECE: 15.25 % → 15.00 %; Brier: 0.2245 → 0.2206). El reshapeo intrínseco de márgenes (Fase 2) es matemáticamente imprescindible antes del escalado global.
5. **Asimetría de vulnerabilidad de las primitivas ante la dinámica de optimización.** Las decisiones booleanas sigmoidales (`bool`) poseen un único grado de libertad escalar y son muy estables en todas las configuraciones (93.00 % → 96.00 %). El enrutamiento bilineal multiopción (`choice`) es muy sensible a la inercia del momentum: fluctúa **24 puntos porcentuales** (72.00 % en E3/E6 hasta 96.00 % en E4b/E5b). En topologías softmax competitivas ($\sum p_i = 1$), actualizar un candidato desplaza simultáneamente a todos los competidores, y el arrastre direccional de la Fase 1 perjudica esa realineación de múltiples hiperplanos salvo que se borre el estado del optimizador.
6. **Convergencia del ECE uniforme y adaptativo como prueba de densidad suave.** En los modelos sin calibrar (**E1**) hay una discrepancia entre el ECE de ancho igual y el adaptativo ($\Delta = 1.27\%$: 15.25 % vs. 13.98 %) por agrupamiento patológico de probabilidades en los *bins* de confianza extrema ($p > 0.90$). En el currículo calibrado (**E5b**) esa brecha se reduce a $\Delta = 0.20\%$ (5.27 % vs. 5.47 %), lo que verifica que la distribución de probabilidad aprendida es suave y está bien dispersa en $[0, 1]$.
7. **El caché de características en memoria habilita bucles internos sin sobrecarga.** Como el *backbone* está congelado, las representaciones latentes de las 1800 muestras se extrajeron una vez en 1.51 s. La matriz completa de 8 experimentos (640 épocas en 8 corridas) se ejecutó en **51.76 s en total** (≈ 0.5 s por corrida), lo que demuestra que la síntesis y recalibración de cabezales sobre la marcha es viable dentro de bucles internos de flujos agénticos autónomos.

## 5.5 Matriz de ablación para el linaje de Google (Gemma 2 2B, $d = 2304$)

Para verificar si los hallazgos de $d = 896$ se generalizan a *backbones* de mayor dimensión y arquitectura no estándar (escalado de embeddings por $\sqrt{d}$, RMSNorm doble, *soft-capping* de logits y atención de ventana deslizante), se ejecutó la misma matriz de ocho experimentos sobre **Gemma-2-2B** con representaciones latentes precalculadas.

### 5.5.1 Resultados sobre la partición de prueba ciega (`train-dataset2/test.jsonl`, $N = 300$)

| ID | Configuración | Optim. F2 | F1 Bool (%) [P] | Exact. Bool (%) [S] | Top-1 Choice (%) [P] | MAE cont. Score | Severas Score ($> 1.5$) | ECE (%) [ancho igual] | ECE (%) [adaptativo] | Brier |
|---|---|---|---|---|---|---|---|---|---|---|
| **E1** | Fase 1 (CE clásica) | — | 89.36 | 90.00 | 74.00 | 0.411 | 0.00 % | 12.65 | 12.14 | 0.2692 |
| **E2** | Fase 1 (CE) + Fase 3 | — | 89.36 | 90.00 | 74.00 | 0.411 | 0.00 % | 12.02 | 12.19 | 0.2697 |
| **E3** | Fase 1 (Focal + Brier) | — | 0.00 | 50.00 | 82.00 | 0.409 | 0.00 % | 33.99 | 33.99 | 0.6551 |
| **E4a** | Fase 1 (CE) + Fase 2 | Continuo | 90.53 | 91.00 | 83.00 | 0.402 | 0.00 % | 13.19 | 11.42 | 0.2177 |
| **E4b** | Fase 1 (CE) + Fase 2 | Reinicio | 91.67 | 92.00 | 96.00 | 0.383 | 0.00 % | 6.60 | 5.34 | 0.1087 |
| **E5a** | Currículo completo (F1 + F2 + F3) | Continuo | 90.53 | 91.00 | 83.00 | 0.402 | 0.00 % | 12.95 | 11.29 | 0.2173 |
| **E5b** | Currículo completo (F1 + F2 + F3) | Reinicio | **91.67** | **92.00** | **96.00** | **0.383** | **0.00 %** | **7.34** | **5.02** | **0.1082** |
| **E6** | Pipeline directo (F1[FB] + F3) | — | 0.00 | 50.00 | 82.00 | 0.409 | 0.00 % | 33.99 | 33.99 | 0.6551 |

### 5.5.2 Dinámica arquitectónica y de optimización en Gemma-2-2B

1. **Replicación del choque de momentum en alta dimensión.** Comparar la retención de momentum (**E4a/E5a**) con el reinicio en frío del optimizador (**E4b/E5b**) revela la misma dinámica que en Qwen2.5. En **E5a** (`continuous`), el momentum histórico del paisaje de la Fase 1 limitó la exactitud de `choice` a 83.00 %, con un ECE adaptativo de 11.29 % y un Brier de 0.2173. En **E5b** (`reset`), reiniciar $m_t = 0$ y $v_t = 0$ en la frontera de fase permitió que la pérdida calibrada atenuada remodelara los márgenes: la exactitud de `choice` subió a **96.00 %** (+13.00 puntos), el ECE adaptativo bajó a **5.02 %** (el menor de todo el estudio) y el Brier se redujo a la mitad (**0.1082**). Esto confirma que la patología no es un artefacto de modelos pequeños ($d = 896$), sino una propiedad de las transiciones de objetivo en currículos multietapa de sondeo lineal.
2. **Colapso de la optimización en arranque en frío bajo Focal + Brier en alta dimensión ($E_3$ y $E_6$).** Al entrenar desde cero con Focal + Brier, sin la preasignación por CE de la Fase 1, Gemma-2-2B sufrió una falla catastrófica en el cabezal booleano: cayó a adivinación aleatoria (**50.00 %** de exactitud), con un ECE macro de **33.99 %** y un Brier de **0.6551**. En un espacio de $d = 2304$ y escala inicial $\sqrt{d} \approx 48.0$, el paisaje de calibración no convexo genera regiones planas o de gradiente evanescente; sin una CE gruesa que oriente primero el hiperplano lineal hacia el vector separador principal, el optimizador queda atrapado en estados degenerados que predicen uniformemente máxima entropía. Esto indica que la alineación de representaciones de la Fase 1 es matemáticamente obligatoria en *backbones* de mayor dimensión.
3. **Precisión ordinal superior y cero errores severos.** En las ocho configuraciones, Gemma-2-2B logró **0.00 %** de errores continuos severos de `score`. Con el currículo recomendado (**E5b**), el MAE continuo fue de **0.383** (mejor que el 0.389 de Qwen). La mayor capacidad expresiva del *backbone* de 2 B parámetros brinda una separabilidad semántica más rica a lo largo del continuo ordinal una vez calibrado.

## 5.6 Matriz de ablación para el linaje no uniforme de Apple (OpenELM-450M, $d = 1536$)

Para probar los límites de la proyección de decisión bajo asimetría estructural extrema, se evaluó **OpenELM-450M**, que reemplaza las capas Transformer uniformes por una asignación de parámetros variable por capa.

### 5.6.1 Resultados sobre la partición de prueba ciega (`train-dataset2/test.jsonl`, $N = 300$)

| ID | Configuración | Optim. F2 | F1 Bool (%) [P] | Exact. Bool (%) [S] | Top-1 Choice (%) [P] | MAE cont. Score | Severas Score ($> 1.5$) | ECE (%) [ancho igual] | ECE (%) [adaptativo] | Brier |
|---|---|---|---|---|---|---|---|---|---|---|
| **E1** | Fase 1 (CE clásica) | — | 99.01 | 99.00 | 97.00 | 0.260 | 0.00 % | 8.01 | 8.01 | 0.0617 |
| **E2** | Fase 1 (CE) + Fase 3 | — | 99.01 | 99.00 | 97.00 | 0.260 | 0.00 % | 5.97 | 5.97 | 0.0544 |
| **E3** | Fase 1 (Focal + Brier) | — | 100.00 | 100.00 | 96.00 | 0.265 | 0.00 % | 9.11 | 9.11 | 0.0718 |
| **E4a** | Fase 1 (CE) + Fase 2 | Continuo | 99.01 | 99.00 | 97.00 | 0.254 | 0.00 % | 6.98 | 7.44 | 0.0542 |
| **E4b** | Fase 1 (CE) + Fase 2 | Reinicio | 100.00 | 100.00 | 98.00 | 0.216 | 0.00 % | 4.20 | 3.94 | 0.0306 |
| **E5a** | Currículo completo (F1 + F2 + F3) | Continuo | 99.01 | 99.00 | 97.00 | 0.254 | 0.00 % | 4.72 | 5.18 | 0.0449 |
| **E5b** | Currículo completo (F1 + F2 + F3) | Reinicio | **100.00** | **100.00** | **98.00** | **0.216** | **0.00 %** | **1.97** | **1.70** | **0.0233** |
| **E6** | Pipeline directo (F1[FB] + F3) | — | 100.00 | 100.00 | 96.00 | 0.265 | 0.00 % | 5.35 | 5.35 | 0.0521 |

### 5.6.2 Incidente metodológico: la patología del sumidero de atención y del token BOS

En la ejecución inicial del pipeline sobre OpenELM-450M, con la misma configuración usada para Qwen2.5 y Gemma-2 (`self.tokenizer.encode(..., add_special_tokens=False)`), las métricas resultaron severamente deprimidas en las ocho configuraciones de ablación: la exactitud booleana se estancó en 61.00 %–64.00 % (apenas sobre el azar de 50 %), el enrutamiento `choice` colapsó al azar (24.00 %–28.00 % en una clasificación de 4 vías, donde adivinar equivale a 25.00 %) y el MAE continuo de `score` osciló entre 0.629 y 0.654, con un Brier elevado (0.6126–0.6228). Dado que estos valores eran muy inferiores a los de las arquitecturas uniformes (96.00 % de `choice` en Qwen y Gemma), se detuvo el pipeline para realizar una auditoría diagnóstica exhaustiva. Esta reveló un modo de falla crítico, en la intersección entre tokenización, dinámica de atención y escalado no uniforme de capas: la **patología del sumidero de atención / ancla del token BOS** (Xiao et al., 2023).

**Diagnóstico de causa raíz.**

1. **Asimetría de tokenización entre linajes.** En `Qwen 2.5-0.5B`, el tokenizador se configura con `"add_bos_token": false`, de modo que `add_special_tokens=False` respeta su distribución nativa de preentrenamiento. En `Gemma 2 2B`, Google usa `<bos>` (token ID 2), pero el modelo (2.0 B de parámetros, escalado de embeddings por $\sqrt{d}$ y atención intercalada de ventana deslizante) posee suficiente redundancia representacional para absorber la omisión sin degradación notable. En `OpenELM-450M`, Apple se basó en la especificación del tokenizador LLaMA (`LlamaTokenizer`, vocabulario de 32 000), donde **cada muestra de entrenamiento comenzaba estrictamente con `<s>` (token BOS, ID 1, `"add_bos_token": true`)**.
2. **Fenómeno del sumidero de atención (Xiao et al., 2023).** En los Transformers decodificadores autorregresivos, la normalización *softmax* obliga a las cabezas de atención a depositar una masa inicial inmensa en la posición 0, independientemente de su contenido semántico, que actúa como "sumidero de atención" numérico. Al pasar `add_special_tokens=False` se omitió el `<s>` inicial y el primer token arbitrario de la consulta del cliente (por ejemplo, `'¿'` o `'Hola'`) fue forzado a cumplir ese rol.
3. **Saturación compuesta en arquitecturas de capas variables.** A diferencia de los Transformers uniformes, la capa 0 de OpenELM tiene solo **12 cabezas de consulta, 3 cabezas KV y un multiplicador FFN de 0.5×**, una capacidad extremadamente estrecha. Sin el ancla `<s>` que absorba la masa de atención sin normalizar, este cuello de botella temprano se saturó por completo.
4. **Degeneración de la variedad latente (colapso hiperanisotrópico).** Al medir las similitudes coseno por pares entre consultas independientes en las representaciones congeladas, sin BOS la similitud coseno media entre consultas completamente no relacionadas se disparó a **0.9830** (desvío 0.012, mínimo 0.9047, máximo 0.9993). Todo el espacio de representación colapsó en un cono hiperanisotrópico de ángulo $< 10^\circ$; las incrustaciones de las opciones de `choice` ($e_{C_i}$) tenían similitudes entre sí $> 0.985$, lo que producía gradientes de pérdida evanescentes ($\sim 0.0017$) y atrapaba la proyección bilineal $W_c$ en adivinación aleatoria (24.00 %).

**Intervención correctiva: anclaje canónico de secuencia con BOS.** Se modificaron la capa de empaquetado y tokenización, tanto en entrenamiento (`scripts/train_heads.py`) como en servicio (`core/engine_mlx.py`):

$$\text{tokens} = \begin{cases} [\text{bos\_id}] \oplus \text{tokens}, & \text{si } \text{bos\_id} \ne \bot \land \text{tokens}[0] \ne \text{bos\_id} \\ \text{tokens}, & \text{en otro caso} \end{cases}$$

Además, se saneó `models/openelm-450m/config.json` eliminando las entradas heredadas de `"auto_map"` que disparaban los avisos de inicialización de *dataclass* de Hugging Face.

**Validación empírica controlada (antes y después).** Se evaluaron las mismas particiones de prueba con y sin el ancla `<s>`:

| Primitiva | Métrica | Sin BOS (`add_special_tokens=False`) | Con BOS canónico (`<s>`, ID 1) | Impacto observado |
|---|---|:---:|:---:|---|
| **Booleana (`bool`)** | Exactitud | 75.00 % (61.00 % en la suite completa) | **93.75 %** | **+18.75 %**: alcanza la paridad con modelos uniformes |
| **Elección (`choice`)** | Exactitud de categoría | 29.73 % (24.00 % en la suite completa) | **86.49 %** | **+56.76 %** (casi triplica; supera el piso aleatorio) |
| **Puntuación (`score`)** | MAE continuo esperado | 1.0323 (0.654 en la suite completa) | **0.2758** | **−73.3 %** de error |
| **Puntuación (`score`)** | Fallas severas ($\lvert \Delta \rvert > 1.5$) | 32.26 % (10 / 31) | **3.22 %** (1 / 31) | Reducción de 10× en fallas catastróficas |
| **Geometría latente** | Similitud coseno por pares ($s_1, s_2$) | 0.9863 (cono hiperanisotrópico) | **0.8066** (variedad dispersa) | Desacopla las direcciones semánticas |

El experimento demuestra que OpenELM-450M posee una sólida capacidad de decisión de Sistema 1 (93.75 % en `bool`, 86.49 % en `choice`, MAE de 0.2758 en `score`) y que el colapso inicial de la ablación se debió por completo a la omisión del BOS en el banco de pruebas.

**Reejecución de la matriz de 8 experimentos.** Con el ancla BOS integrada permanentemente en `train_heads.py` y `engine_mlx.py`, se reejecutó la matriz completa (§5.6.1). Bajo el currículo recomendado (**E5b**), OpenELM-450M alcanzó **100.00 %** de exactitud booleana, **98.00 %** en `choice`, un MAE continuo de **0.216**, **0.00 %** de errores severos, un ECE macro de ancho igual de **1.97 %** (adaptativo: **1.70 %**) y un Brier de **0.0233**, superando a los *backbones* uniformes en nitidez de calibración. El reinicio del optimizador entre las Fases 1 y 2 (E4b frente a E4a, E5b frente a E5a) elevó la exactitud de `choice` de 97.00 % a **98.00 %** y redujo el ECE macro de 4.72 % a **1.97 %**, lo que confirma que la mitigación del choque de momentum es aplicable también a arquitecturas no uniformes. Aun en las corridas colapsadas previas a la corrección, la pérdida ordinal con suavizado gaussiano mantuvo un MAE continuo acotado (0.629) con solo 1.00 % de errores severos, y con el anclaje apropiado los errores severos bajaron a **0.00 %** en las ocho configuraciones, lo que indica que el suavizado ordinal continuo constituye un sesgo inductivo estructuralmente resiliente.

## 5.7 Síntesis comparativa entre familias

### 5.7.1 Matriz integral de rendimiento y eficiencia (currículo óptimo E5b)

| Propiedad / Métrica | Qwen 2.5-0.5B (tipo Llama) | OpenELM-450M (Apple asimétrico) | Gemma 2 2B (Google moderno) | Análisis entre linajes |
|---|:---:|:---:|:---:|---|
| **Escala y topología** | | | | |
| Parámetros | 0.49 B | 0.45 B | 2.00 B | Rango de escala de 4.4× |
| Dimensión latente ($d$) | **896** | **1536** | **2304** | Variación de 2.57× en ancho latente |
| Capas ($L$) | 24 | 20 | 26 | Profundidad comparable |
| Arquitectura de capas | Uniforme | No uniforme (FFN 0.5× → 4.0×) | Uniforme con ventana deslizante | Estándar vs. asimétrica vs. compleja |
| **Eficiencia de servicio (GPU Metal)** | | | | |
| Latencia de decisión p50 | **61.43 ms** | 93.83 ms | 282.58 ms | Qwen es 4.6× más rápido que Gemma |
| Latencia de decisión p95 | **62.02 ms** | 97.57 ms | 285.46 ms | Colas muy deterministas |
| Memoria pico en Metal | **~380 MB** | ~872 MB | ~1.47 GB | Apta para despliegue en el borde |
| Tiempo de entrenamiento de la matriz (8 exp.) | **51.76 s** | 86.55 s | 286.58 s | Todas por debajo de 5 minutos gracias al caché latente |
| **Desempeño óptimo (E5b)** | | | | |
| F1 de verificación booleana [P] | **96.08 %** | **100.00 %** | 91.67 % | Alto F1 en todos los linajes (pico en OpenELM) |
| Exactitud booleana [S] | **96.00 %** | **100.00 %** | 92.00 % | Alta exactitud en todos los linajes |
| Exactitud Top-1 de elección [P] | **96.00 %** | **98.00 %** | **96.00 %** | Enrutamiento sobresaliente ($\ge 96\%$) en los tres |
| MAE continuo de `score` | 0.389 | **0.216** | 0.383 | OpenELM logra el mejor MAE |
| Fallas severas de `score` ($> 1.5$) | **0.00 %** | **0.00 %** | **0.00 %** | Garantía universal de 0.00 % |
| ECE macro (ancho igual) | 5.27 % | **1.97 %** | 7.34 % | OpenELM logra el menor ECE |
| ECE macro (cuantiles adaptativos) | 5.47 % | **1.70 %** | 5.02 % | Ajuste probabilístico más ajustado |
| Brier macro | 0.0780 | **0.0233** | 0.1082 | OpenELM entrega la menor dispersión global |

### 5.7.2 Impacto del currículo: línea base (E1) frente al currículo recomendado (E5b)

| Linaje | F1 Bool E1 [P] | F1 Bool E5b [P] | Exact. Bool E1 [S] | Exact. Bool E5b [S] | Top-1 Choice E1 [P] | Top-1 Choice E5b [P] | MAE E1 | MAE E5b | Severas E1 | Severas E5b | ECE (AI) E1 | ECE (AI) E5b | Brier E1 | Brier E5b |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Qwen 2.5-0.5B** ($d = 896$) | 93.33 % | **96.08 %** | 93.00 % | **96.00 %** | 79.00 % | **96.00 %** | 0.432 | **0.389** | 2.00 % | **0.00 %** | 15.25 % | **5.27 %** | 0.2245 | **0.0780** |
| **OpenELM-450M** ($d = 1536$) | 99.01 % | **100.00 %** | 99.00 % | **100.00 %** | 97.00 % | **98.00 %** | 0.260 | **0.216** | 0.00 % | **0.00 %** | 8.01 % | **1.97 %** | 0.0617 | **0.0233** |
| **Gemma 2 2B** ($d = 2304$) | 89.36 % | **91.67 %** | 90.00 % | **92.00 %** | 74.00 % | **96.00 %** | 0.411 | **0.383** | 0.00 % | **0.00 %** | 12.65 % | **7.34 %** | 0.2692 | **0.1082** |

*(AI = ancho igual.)* El currículo completo con reinicio mejora todas las métricas de calibración en los tres linajes.

### 5.7.3 Evaluación comparativa preliminar con submuestras (documento fuente, §4.8)

El documento fuente incluye además una tabla comparativa de evaluación sobre la partición ciega con los siguientes valores:

| Primitiva | Métrica | Qwen 2.5-0.5B | OpenELM-450M | Gemma 2 2B |
|---|---|---|---|---|
| Noul (booleana) | Coincidencia exacta | 96.00 % | 90.00 % | 98.48 % |
| Choice (enrutamiento) | Coincidencia exacta de categoría | 96.00 % | 42.37 % | 84.85 % |
| Score | MAE continuo | 0.389 | 0.482 | 0.412 |
| Score | Dentro de $\pm 1$ nivel | 93.00 % | 98.04 % | 100.00 % |
| Score | Fallas severas ($> 1.5$) | 0.00 % (0 / 100) | 0.00 % (0 / 51) | 0.00 % (0 / 48) |

> **[Nota para revisión]** Los valores de OpenELM y Gemma de esta tabla difieren de los de las matrices E5b de las §5.5 y §5.6 (por ejemplo, `choice` de OpenELM: 42.37 % frente a 98.00 %; `bool` de Gemma: 98.48 % frente a 92.00 %), y las cantidades de muestras de `score` (51 y 48) no coinciden con las 100 por primitiva de la partición de prueba. Es posible que correspondan a una corrida previa a la corrección del BOS y/o a una submuestra, pero el documento fuente no lo aclara; el resumen, las conclusiones y las §5.5–5.7.2 usan los valores E5b. Se recomienda reejecutar esta tabla o eliminarla antes de la versión final. El cumplimiento de cotas ordinales de $\pm 1$ nivel ($\ge 93\%$) citado en el Resumen proviene de esta tabla.

### 5.7.4 Latencia de decisión concurrente en GPU Metal de Apple Silicon

Evaluación concurrente de las tres primitivas (`noul` + `choice` + `score`) en una única pasada hacia adelante:

| *Backbone* | Nivel arquitectónico | Latencia p50 (ms) | Latencia p95 (ms) | Memoria pico Metal | Nivel operativo |
|---|---|---|---|---|---|
| **Qwen 2.5-0.5B** | Rápido | **61.43** | **62.02** | ~380 MB | Enrutamiento de borde de latencia ultrabaja |
| **OpenELM-450M** | Borde eficiente | **93.83** | **97.57** | ~872 MB | Puntuación asimétrica en dispositivo |
| **Gemma 2 2B** | Equilibrado | **282.58** | **285.46** | ~1.47 GB | Decisión semántica de alta capacidad |

### 5.7.5 Síntesis teórica entre arquitecturas

1. **Universalidad del choque de momentum.** En las tres familias, retener los *buffers* del optimizador durante la transición de CE a Focal + Brier degrada la exactitud de enrutamiento y eleva el error de calibración, mientras que reiniciar el estado lleva la exactitud de enrutamiento a $\ge 96.00\%$ (con OpenELM en 98.00 %) y reduce el ECE adaptativo a entre 1.70 % y 5.47 %. El reinicio del optimizador es, por lo tanto, una primitiva obligatoria en la síntesis de sondas lineales calibradas multietapa.
2. **Ley del ancla de arranque en frío de alta dimensión.** Mientras los modelos pequeños ($d = 896$) pueden converger a clasificaciones aceptables con arranque en frío directo bajo Focal + Brier, las variedades latentes de mayor dimensión ($d = 2304$) sufren un colapso severo de la optimización sin preasignación de Fase 1 (50.00 % de exactitud booleana y 33.99 % de ECE en E3 de Gemma). Se propone denominar a este fenómeno la *Ley del Ancla de Arranque en Frío de Alta Dimensión*: a medida que aumenta la dimensionalidad latente $d$, la penalización de calibración no convexa debe ser precedida por un sondeo de entropía cruzada sin regularizar que ancle los hiperplanos separadores antes de refinar los márgenes de probabilidad. Esta formulación es una generalización empírica a partir de dos dimensiones ($d = 896$ y $d = 2304$) y no una ley demostrada.
3. **Monotonicidad universal de las esperanzas ordinales.** En las tres arquitecturas (de 0.45 B a 2.0 B de parámetros, anchos latentes de 896 a 2304, estructuras uniformes y asimétricas), el cabezal ordinal logró **0.00 %** de errores severos con el currículo completo (E5b) y un MAE continuo entre 0.216 y 0.389. El aprendizaje con objetivos suaves gaussianos constituye un sesgo inductivo muy robusto que impone monotonicidad semántica independientemente de la arquitectura.
4. **Compromisos operativos en Apple Silicon.** Qwen 2.5-0.5B ofrece una latencia ultrabaja (61.43 ms) con un presupuesto de memoria mínimo (~380 MB) y alta precisión (96.00 % en Bool y Choice, Brier 0.0780), por lo que es la opción principal para agentes de borde y *middleware* en tiempo real. Gemma 2 2B entrega un error continuo bajo (MAE = 0.383) y una calibración adaptativa ajustada (ECE = 5.02 %), pero requiere 4.6× más latencia (282.58 ms) y 3.8× más memoria (1.47 GB). OpenELM-450M, una vez anclado con el token BOS `<s>`, alcanza la mayor exactitud (100.00 % en Bool, 98.00 % en Choice, MAE de 0.216) y la calibración más nítida (ECE = 1.97 %, Brier = 0.0233) con latencia inferior a 100 ms (93.83 ms) y ~872 MB de memoria, sin ajustar el *backbone*.
5. **Geometría de la variedad latente en `choice`.** La transferencia de los cabezales lineales fue transparente entre *backbones*, pero la exactitud de `choice` refleja la geometría de la variedad latente: en la evaluación preliminar de la §5.7.3 se observaban grandes diferencias entre linajes (Qwen 96.00 %, Gemma 84.85 %, OpenELM 42.37 %) que no se observan en la matriz E5b (96.00 %, 96.00 % y 98.00 %, respectivamente). Si la diferencia se debiera, al menos en parte, a la omisión del token BOS descrita en la §5.6.2, ilustraría que una parte de la variación atribuida al diseño del modelo es en realidad un problema de tokenización; el documento fuente no permite confirmarlo, por lo que se deja como punto a verificar.

---

# Capítulo 6. Discusión

## 6.1 Interpretación de los hallazgos

Los resultados muestran que un *backbone* autorregresivo congelado contiene la información necesaria para sustentar decisiones estructuradas y calibradas, y que el cuello de botella se encuentra en la **forma de proyectar y calibrar** esa información, no en la capacidad del modelo. Tres resultados lo respaldan. Primero, la calibración no se obtiene con un único recurso: el escalado por temperatura aislado (E2) casi no modificó el ECE de la línea base (15.25 % → 15.00 % en Qwen), mientras que el currículo completo con reinicio del optimizador (E5b) lo redujo a 5.27 %. Segundo, la mayor parte de las fallas relevantes no provino de la arquitectura del modelo sino de decisiones de datos, serialización y tokenización (rúbrica ausente del prompt, espacios finales que fusionan tokens BPE, sesgos léxicos del corpus, omisión del BOS). Tercero, las primitivas se comportan de modo distinto frente a la optimización: la booleana es muy estable, mientras que `choice` es muy sensible a la inercia del optimizador (24 puntos porcentuales de variación en Qwen).

## 6.2 Lineamientos prácticos

De la investigación se derivan los siguientes lineamientos para investigadores e ingenieros que adapten *backbones* decodificadores para servicio determinista de decisiones:

1. **No asumir la validez de sondas *zero-shot*.** Los cabezales lineales sobre *backbones* congelados requieren alineación supervisada explícita; ni siquiera los modelos fundacionales bien alineados proyectan por sí solos las fronteras de tarea sobre proyecciones lineales sin calibrar.
2. **Normalizar siempre con $\ell_2$ los embeddings en la elección.** La anisotropía de las representaciones produce explosión de logits y saturación del *softmax* con productos escalares crudos. La puntuación bilineal de `choice` requiere normalizar tanto la proyección de la consulta como los embeddings de las opciones, junto con escalado por temperatura contrastiva ($\tau \approx 0.03$).
3. **Condicionar el Transformer con las definiciones de la rúbrica.** Una capa lineal no puede aprender dinámicamente rúbricas arbitrarias definidas por el usuario; los criterios deben serializarse en el prompt de atención para que las cabezas atiendan entre los tokens del estado y los descriptores de nivel.
4. **Ajustar con cuidado el ancho del suavizado ordinal.** Con $\sigma \ge 1.0$, el suavizado gaussiano diluye matemáticamente los objetivos extremos en rúbricas cortas ($M \le 5$). Se recomienda $\sigma \le 0.40$.
5. **Preservar la variación lingüística y cultural en el ajuste multitarea.** Los modelos multilingües no transfieren de manera fiable la comprensión del lenguaje formal a dialectos regionales coloquiales y emocionalmente cargados (por ejemplo, la jerga del español argentino) sin aumento de datos específico del dominio.
6. **Desacoplar los léxicos de dominio de la polaridad emocional.** Las sondas lineales sobre representaciones congeladas sobreajustan fácilmente a correlaciones léxicas espurias. La cobertura de dominio debe balancearse en todos los niveles discretos, y el entrenamiento ordinal debe usar log-softmax estable ($z - \text{logsumexp}(z)$) en lugar de recorte ingenuo (*clipping*) para evitar la saturación del optimizador.
7. **Calibrar la temperatura según la dimensión del modelo.** Al pasar de *backbones* compactos ($d \le 1024$) a arquitecturas de mayor dimensión ($d \ge 1536$), la temperatura $\tau$ de `choice` debe escalarse inversamente con $\sqrt{d}$ para acomodar los efectos de concentración del coseno en dimensiones latentes altas. *(Este lineamiento se enuncia en el documento fuente como recomendación; los experimentos usaron $\tau = 0.03$ y no incluyen una validación específica del escalado con $\sqrt{d}$.)*
8. **Sanear los delimitadores de unión del prompt y preservar los imperativos directos.** Los tokenizadores de subpalabras fusionan vorazmente los espacios finales con los separadores; los estados deben recortarse (`.strip()`) antes del empaquetado. Las rúbricas deben incluir enunciados transaccionales directos, sin prefacio, para evitar que las sondas confundan la formulación llana con hostilidad.
9. **Implementar despacho polimórfico para arquitecturas no uniformes.** Los *backbones* no uniformes (como OpenELM) usan convenciones distintas de nombres de módulos (`.transformer` frente a `.model`) y de claves de configuración (`model_dim` frente a `hidden_size`). Los entornos de ejecución deben utilizar escaleras de acceso polimórfico a atributos y sanear el código de configuración remota.
10. **Preferir el ajuste exclusivo de cabezales con calibración en dos niveles frente al ajuste fino completo.** El ajuste completo de *backbones* de miles de millones de parámetros introduce olvido catastrófico, distorsiona las geometrías lingüísticas preentrenadas y genera sobrecostos insostenibles de almacenamiento y despliegue. Congelar el *backbone* y realizar sondeo lineal supervisado sobre cabezales livianos (~3.2 MB), con modelado intrínseco de la pérdida (Focal + Brier) y escalado por temperatura extrínseco, dio calibración de ECE entre 1.97 % y 5.27 % en Qwen y OpenELM y Brier entre 0.0233 y 0.0780, entrenamiento en menos de un minuto en GPU Metal y recalibración de dominio fuera de distribución en menos de 50 ms sin mutar pesos del modelo.
11. **Gestionar el estado del optimizador en bucles agénticos de autoentrenamiento.** En flujos autónomos donde los LLM generan tareas de entrenamiento, formulan preguntas y entrenan cabezales de decisión sobre la marcha, las transiciones de arranque en caliente entre exploración sin calibrar (CE) y explotación calibrada (Focal + Brier) deben reinicializar los *buffers* de AdamW en lugar de retener el momentum acumulado. Retenerlo duplica el error de calibración (ECE = 10.33 % frente a 5.27 %; Brier = 0.1614 frente a 0.0780) y limita la exactitud de `choice` a 85.00 % en lugar de 96.00 %. Además, combinar el caché de características del *backbone* congelado con calibración poshoc permite sintetizar cabezales en menos de 0.6 s, de modo que los agentes puedan instanciar y calibrar cabezales de decisión efímeros y especializados en tiempo de ejecución.
12. **Evaluar los cabezales ordinales con valor esperado continuo.** No deben evaluarse con métricas de clasificación discreta (exactitud por $\arg\max$ o distancia entera $\ge 2$), pues introducen una discontinuidad escalonada que penaliza distribuciones de incertidumbre bien calibradas y premia la sobreconfianza patológica tipo Dirac. Deben medirse el MAE continuo ($\text{MAE}_{\text{cont}} = \frac{1}{N}\sum \lvert \bar{S} - y^* \rvert$) y las tasas de excedencia de tolerancia ($\text{TER}_{\tau}$) sobre $\bar{S} = \sum k \cdot p_k$, alineando la evaluación con el contrato operativo en tiempo de ejecución.
13. **Imponer tokens ancla de secuencia específicos del modelo.** No debe asumirse que `add_special_tokens=False` es universalmente seguro. Los modelos preentrenados con tokens obligatorios de inicio de secuencia (como LLaMA y OpenELM con `<s>`, ID 1) usan el token 0 como sumidero de atención (Xiao et al., 2023); omitirlo en *backbones* compactos o de capas variables satura las capas estrechas tempranas (FFN de 0.5× en la capa 0 de OpenELM), colapsa la variedad latente en un cono (similitud coseno $> 0.98$) y reduce la exactitud del enrutamiento lineal al azar (24 %). Los *pipelines* de extracción y servicio deben inspeccionar programáticamente `tokenizer.bos_token_id` y garantizar el anclaje canónico antes del sondeo lineal.

## 6.3 Amenazas a la validez y limitaciones

- **Datos sintéticos y de un solo dominio.** Las particiones provienen de un generador sintético para atención al cliente, por lo que los resultados no se extrapolan automáticamente a tráfico real ni a otros dominios. La recalibración OOD de la §3.4.1 mitiga parcialmente el problema, pero no lo elimina.
- **Tamaño de las particiones de prueba.** Con 100 instancias por primitiva, una diferencia de uno o dos puntos porcentuales (por ejemplo, 96 % frente a 98 %) corresponde a pocas muestras y no debe interpretarse como superioridad sólida sin pruebas de significancia.
- **Semilla única.** Las matrices de ablación se ejecutaron con `seed = 42`; no se informa la variabilidad entre semillas.
- **Comparación con el ajuste fino completo.** El contraste de la §2.5 es de diseño y no experimental; tampoco se compararon otras técnicas de adaptación eficiente del *backbone*.
- **Hardware único.** La latencia y la memoria se midieron exclusivamente en GPU Metal de Apple Silicon.
- **Corrección en el desarrollo.** Varias correcciones (rúbrica, $\sigma$, BOS) se descubrieron inspeccionando fallas puntuales; si bien las pruebas de regresión protegen contra su reaparición, no se descarta la existencia de patologías aún no detectadas.
- **Fuga de datos previa.** La auditoría de fuga de datos (§3.5.2) muestra el riesgo de las plantillas sintéticas; la ausencia de solapamiento de estados en `train-dataset2/` no garantiza ausencia de similitud semántica entre particiones.

---

# Capítulo 7. Conclusiones y trabajo futuro

## 7.1 Contrastación de la hipótesis

La hipótesis de investigación, *"es posible construir un modelo de decisión a partir de un SLM pre-entrenado"*, **se confirma** en las condiciones del estudio. A partir de tres SLM públicos de linajes arquitectónicos divergentes (Qwen 2.5-0.5B, Gemma 2 2B y OpenELM-450M), y sin modificar sus pesos, se construyeron modelos de decisión que resuelven en una única pasada hacia adelante las tres primitivas (booleana, elección y puntuación ordinal) con probabilidades calibradas. Bajo el currículo recomendado (E5b), sobre una partición de prueba ciega de 300 instancias, todos los *backbones* alcanzaron una exactitud booleana de al menos 92.00 % (hasta 100.00 % en OpenELM), una exactitud de elección de al menos 96.00 % (hasta 98.00 % en OpenELM), 0.00 % de errores continuos severos en la puntuación ordinal, un ECE entre 1.97 % y 7.34 % y una latencia de decisión concurrente inferior a 300 ms en GPU Metal de Apple Silicon. Estos resultados respaldan también la versión de generalidad arquitectónica de la hipótesis (Traducibilidad Universal a Sistema 1).

## 7.2 Cumplimiento de los objetivos

| Objetivo específico | Resultado |
|---|---|
| Construir un software que ejecute diferentes técnicas de *fine-tuning* a un SLM | Se desarrolló OpenJEV: motor MLX con empaquetado por bloques, cabezales canónicos, entrenamiento por lotes con caché latente y currículo en fases, y exportación de cabezales y calibración (§3.7) |
| Construir una suite de tests que garantice el correcto funcionamiento | Se incorporaron pruebas de verificación de cabezales y de regresión en integración continua (§3.7); queda por documentar su alcance y cobertura completos |
| Construir un conjunto de datos de eventos de decisión (estado/pregunta/respuesta) | Corpus sintético bilingüe de 1800 ejemplos en particiones disjuntas 1200/300/300, con auditoría de fuga de datos (§3.5) |
| Seleccionar técnicas de *fine-tuning* | Ajuste exclusivo de cabezales con CE, Focal, Brier, pérdida ordinal gaussiana, currículo en fases, reinicio del optimizador y escalado por temperatura (§2.7, §3.4) |
| Comparar el impacto de las diferentes técnicas | Matrices de ablación E1–E6 en las tres familias (§5.4–5.6) y ablación de $\sigma$ (§5.3) |
| Evaluar el rendimiento en modelos de diferentes familias | Evaluación comparativa de eficiencia y desempeño en Qwen, Gemma y OpenELM (§5.7) |

## 7.3 Respuestas sintéticas a las preguntas de investigación

- **P1–P2.** Un SLM es un modelo de lenguaje preentrenado compacto, ejecutable localmente; las familias estudiadas (Qwen, Gemma, OpenELM) difieren en tokenización, normalización, atención y uniformidad por capa (§2.1, §3.2).
- **P3.** Un modelo de decisión produce respuestas estructuradas y calibradas en una sola pasada; a diferencia de un LLM, no genera tokens autorregresivamente, aísla las preguntas entre sí y garantiza el formato por construcción (§2.3).
- **P4.** Aporta salidas consumibles por sistemas tradicionales, baja latencia y bajo consumo de memoria, y confianza utilizable para umbrales y escalamiento humano (§2.4).
- **P5.** Las vías actuales son el *prompting* generativo, el entrenamiento desde cero, el ajuste fino completo y el ajuste exclusivo de cabezales sobre un *backbone* congelado, que fue la elegida (§2.5).
- **P6.** Opera con eventos (estado, pregunta, respuesta) de tres tipos: booleano, elección entre opciones y puntuación ordinal con rúbrica (§2.6).
- **P7.** Sondeo lineal con CE, Focal + Brier y pérdida ordinal gaussiana, currículo en fases con reinicio del optimizador y escalado por temperatura (§2.7, §5.4).
- **P8.** Con exactitud, métricas continuas ($\text{MAE}$, $\text{RMSE}$, $\text{TER}_\tau$), ECE, Brier, latencia y memoria (§2.8, §3.6).
- **P9.** Las diferencias de tokenización (BOS), escala de embeddings, normalización, atención y uniformidad por capa generaron fallas específicas, como el sumidero de atención en OpenELM, que redujeron el rendimiento a niveles cercanos al azar hasta ser corregidas (§4.10, §5.6.2).

## 7.4 Aportes

(i) Un procedimiento reproducible y un software para convertir SLM públicos en modelos de decisión calibrados sin modificar el *backbone*; (ii) un protocolo de evaluación continua para primitivas ordinales; (iii) un catálogo de patologías con diagnóstico matemático y corrección; y (iv) evidencia empírica de que el reinicio del optimizador entre fases es determinante para la calibración en las tres familias.

## 7.5 Trabajo futuro

A partir de las limitaciones de la §6.3 se proponen las siguientes líneas, que constituyen sugerencias de esta tesina y no resultados del estudio: (a) comparar experimentalmente el ajuste exclusivo de cabezales con el ajuste fino completo y con técnicas de adaptación eficiente; (b) validar con datos reales de producción y en otros dominios y lenguas, incluidos otros dialectos regionales; (c) repetir las matrices de ablación con múltiples semillas y pruebas de significancia; (d) incorporar otras familias de SLM y otros tamaños; (e) medir latencia y consumo en otros entornos de hardware; (f) verificar experimentalmente el escalado de $\tau$ con $\sqrt{d}$ y la Ley del Ancla de Arranque en Frío de Alta Dimensión en más dimensiones latentes; y (g) integrar el modelo de decisión en un flujo agéntico real y medir el efecto sobre la confiabilidad y el costo extremo a extremo.

---

# Referencias

Bai, J., et al. (2024). *Qwen2.5: A Comprehensive Foundation Model Series*. Alibaba Cloud Technical Report.

Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review, 78*(1), 1–3.

Ethayarajh, K. (2019). How contextual are contextualized representations? Comparing local and global context in ELMo, BERT, and GPT-2. En *Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP)*.

Gemma Team, et al. (2024). *Gemma 2: Improving open language models at a practical size*. arXiv:2408.00118.

Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association, 102*(477), 359–378.

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. En *Proceedings of the International Conference on Machine Learning (ICML)*.

Kahneman, D. (2011). *Thinking, fast and slow*. Farrar, Straus and Giroux.

Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. En *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*.

Mehta, S., et al. (2024). *OpenELM: An efficient language model family with open training and inference framework*. Apple Inc., arXiv:2404.14619.

Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2024). RoFormer: Enhanced transformer with rotary position embedding. *Neurocomputing, 568*, 127063.

Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2023). *Efficient streaming language models with attention sinks*. arXiv:2309.17453.
