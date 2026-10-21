*Este proyecto ha sido creado como parte del currículo de 42 por hohu.*

# Call Me Maybe

## Descripción

Call Me Maybe es una herramienta local de *function calling* para modelos de
lenguaje. Recibe peticiones en lenguaje natural y una lista de funciones
disponibles, selecciona la función más probable y genera los valores de sus
parámetros.

El objetivo del proyecto es construir una salida estructurada sin depender de
la generación libre de texto. Para ello utiliza el modelo local
`Qwen/Qwen3-0.6B`, proporcionado mediante el paquete `llm_sdk`, y restringe
los tokens que pueden elegirse durante la generación.

La salida actual es un array JSON. Cada elemento incluye el prompt original,
el nombre de la función elegida y los parámetros generados:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": "2",
    "b": "3"
  }
}
```

## Instrucciones

### Requisitos

- Python 3.10 o superior.
- [uv](https://docs.astral.sh/uv/).
- Acceso a Internet la primera vez que se descargan las dependencias y el
  modelo de Hugging Face.

### Instalación

El proyecto usa `uv` y contiene un `Makefile`. El objetivo `install` crea los
directorios de caché para paquetes y modelos, e instala las dependencias de
ejecución y desarrollo:

```bash
make install
```

Por defecto, las cachés se configuran en el `Makefile` mediante estas
variables:

```makefile
HF_HOME = /home/hohu/sgoinfre/call_me_maybe/.llm
UV_CACHE_DIR = /home/hohu/sgoinfre/call_me_maybe/.uv_cache
```

Se pueden sustituir en la línea de comandos:

```bash
make install HF_HOME=/ruta/a/modelos UV_CACHE_DIR=/ruta/a/uv-cache
```

### Ejecución

La forma más directa de ejecutar el programa con los ficheros incluidos en
este repositorio es:

```bash
uv run python -m src \
  --input input/function_calling_tests.json \
  --function_definitions input/functions_definition.json \
  --output output/function_calling_results.json
```

El modelo se descarga o carga durante la creación de `LLMModel`; la primera
ejecución puede tardar más que las siguientes debido a esa descarga y a la
caché local.

Los argumentos disponibles son:

```text
--input <archivo>                 Archivo JSON con los prompts.
--function_definitions <archivo>  Archivo JSON con las funciones disponibles.
--output <archivo>                Ruta del JSON de salida.
```

También hay objetivos del `Makefile`:

```bash
make run
make debug
make clean
make lint
make lint-strict
```

`make debug` inicia el programa con el depurador estándar `pdb`. `make clean`
elimina cachés de Python, mypy y pytest.

> Nota: `make run` usa las rutas predeterminadas configuradas en
> `src/parsing/arguments.py` (`data/input/...`). En esta copia del repositorio
> los ejemplos están en `input/...`; por ello, usa el comando con rutas
> explícitas mostrado arriba, o crea la estructura `data/input/` esperada.

## Formato de entrada

El archivo de prompts debe ser un array JSON de objetos con la clave `prompt`:

```json
[
  {"prompt": "Greet Ada"},
  {"prompt": "What is the sum of 2 and 3?"}
]
```

El archivo de definiciones debe ser un array JSON. Cada función contiene un
nombre, una descripción, sus parámetros y el tipo de retorno:

```json
[
  {
    "name": "fn_greet",
    "description": "Generate a greeting message for a person by name.",
    "parameters": {
      "name": {"type": "string"}
    },
    "returns": {"type": "string"}
  }
]
```

La capa de parsing usa modelos Pydantic para comprobar la estructura. Además,
un `model_validator` verifica que cada parámetro declarado contiene un campo
`type`. Los errores de archivo inexistente, permisos insuficientes, JSON no
válido, listas vacías o estructuras incorrectas se convierten en errores
explicativos.

## Algoritmo de decodificación restringida

La generación se divide en dos fases.

1. Selección de función: el programa tokeniza cada nombre de función válido.
   En cada paso consulta los logits del LLM y solo permite el siguiente token
   que continúe al menos uno de los nombres disponibles. Cuando el prefijo
   generado coincide con un nombre completo y no existe una función más larga
   con el mismo prefijo, la función queda seleccionada.
2. Generación de parámetros: para cada parámetro de la función elegida se crea
   un prompt con los nombres de los parámetros. En cada paso se ordenan los
   logits y se acepta el token de mayor puntuación cuyo texto esté permitido
   por una máscara dependiente del tipo. Las cadenas permiten letras, números,
   espacios, guion bajo y comillas; los números permiten dígitos, signo menos,
   punto y comillas. La generación se detiene al producir una comilla o al
   alcanzar el límite de 64 caracteres.

Esta restricción reduce el espacio de búsqueda del modelo: no puede inventar
un nombre de función fuera de las definiciones y los valores de parámetros se
mantienen dentro de un conjunto de caracteres compatible con su tipo básico.

## Decisiones de diseño

- `LLMProtocol` describe la interfaz mínima que necesita el procesador:
  `encode`, `decode` y `get_logits_from_input_ids`. Así el procesador no queda
  acoplado directamente a un SDK concreto.
- `LLMModel` encapsula `Small_LLM_Model`. Sus campos Pydantic representan la
  configuración del modelo y su atributo privado `_model` guarda la instancia
  real del SDK tras la inicialización.
- `PromptProcessor` crea internamente su LLM y trabaja con objetos
  `FunctionDefinition`, no con diccionarios sin tipo.
- Cada módulo de parsing abre y valida su propio archivo. Esto mantiene juntos
  el formato esperado y sus mensajes de error, sin una capa auxiliar genérica.
- `argparse` se utiliza para exponer las rutas de entrada y salida desde la
  línea de comandos.

## Análisis de rendimiento

El coste principal es una inferencia del LLM por cada token generado. Por ello,
la carga inicial del modelo y la longitud de los prompts influyen directamente
en el tiempo total. La caché de Hugging Face evita volver a descargar el modelo
en ejecuciones posteriores.

La selección de función es más fiable que una generación abierta porque cada
token debe pertenecer a un prefijo de nombre válido. Para parámetros, las
máscaras de caracteres disminuyen resultados con texto libre, aunque no
constituyen por sí solas una validación semántica completa: un valor puede
respetar los caracteres permitidos y aun así no responder correctamente a la
petición. La precisión y el tiempo exactos deben medirse con un conjunto de
pruebas representativo del entorno de ejecución.

## Retos encontrados

- Integrar un LLM local sin filtrar sus detalles por todo el proyecto. Se
  resolvió con el adaptador `LLMModel` y el protocolo `LLMProtocol`.
- Asegurar que el nombre generado pertenece al conjunto de funciones. Se
  resolvió comparando el prefijo tokenizado generado con todos los nombres
  permitidos en cada iteración.
- Gestionar entradas mal formadas. Se añadieron modelos Pydantic y excepciones
  específicas para ficheros ausentes, permisos, JSON inválido y estructuras
  incompatibles.
- Evitar descargar repetidamente el modelo y las dependencias. El `Makefile`
  define cachés separadas para Hugging Face y uv.

## Estrategia de pruebas

La validación del proyecto combina comprobaciones estáticas y pruebas
manuales de integración:

- `make lint` ejecuta `flake8` y `mypy` con reglas de tipos estrictas para
  funciones sin anotaciones, retornos `Any` e ignores sin uso.
- `make lint-strict` ejecuta la variante estricta de mypy.
- `python -m compileall src` comprueba que todos los módulos compilan.
- Se prueban archivos de entrada válidos y casos límite: archivo ausente,
  permisos insuficientes, JSON inválido, lista vacía, prompts vacíos y
  parámetros sin `type`.
- La salida se revisa como JSON y se compara con las definiciones disponibles:
  el nombre debe existir y los parámetros generados deben corresponder a la
  función elegida.

## Ejemplos de uso

Ejecutar con rutas explícitas:

```bash
uv run python -m src \
  --input input/function_calling_tests.json \
  --function_definitions input/functions_definition.json \
  --output output/function_calling_results.json
```

Ejecutar el depurador:

```bash
make debug
```

Comprobar estilo y tipos:

```bash
make lint
```

## Recursos

- [Documentación de Python sobre argparse](https://docs.python.org/3/library/argparse.html).
- [Documentación de Pydantic](https://docs.pydantic.dev/).
- [Documentación de PyTorch](https://pytorch.org/docs/stable/index.html).
- [Documentación de Hugging Face Transformers](https://huggingface.co/docs/transformers/).
- [Documentación del modelo Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B).
- [Function Calling and Other API Updates](https://openai.com/index/function-calling-and-other-api-updates/), introducción al uso de modelos para seleccionar herramientas estructuradas.

### Uso de IA

Se utilizó una herramienta de IA conversacional como apoyo durante el
desarrollo para explicar conceptos de Python, Pydantic, `Protocol`,
`argparse`, validación de datos y organización del código. También se usó para
proponer refactorizaciones, mensajes de error y el primer borrador de esta
documentación. Las decisiones finales de estructura, las rutas, los comandos,
el código presente y las comprobaciones se revisaron frente al contenido real
del repositorio.
