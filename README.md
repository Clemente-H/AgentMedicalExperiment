# Sistema de "Consejo de Modelos" para análisis de imágenes médicas

Este sistema implementa un enfoque de "consejo de modelos" para mejorar la precisión en tareas de reconocimiento visual en imágenes médicas. El sistema utiliza tres modelos "consejeros" (Claude, Grok y DeepSeek R1) y un modelo "decisor" (OpenAI o1) para determinar la respuesta final a preguntas de opción múltiple.

## Características

- **Enfoque de consejo de modelos**: Combina la sabiduría de varios modelos LLM multimodales.
- **Procesamiento paralelo**: Consulta a los modelos consejeros de forma simultánea para reducir el tiempo total.
- **Análisis detallado**: Genera reportes completos sobre el rendimiento de cada modelo y por categorías.
- **Configuración flexible**: Permite ajustar fácilmente los modelos y parámetros a través de un archivo YAML.
- **Sistema robusto**: Manejo de errores y capacidad para reanudar ejecuciones interrumpidas.

## Requisitos

- Python 3.8 o superior
- API keys para: Anthropic (Claude), OpenAI (GPT), X.AI (Grok) y OpenRouter (DeepSeek)

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/medical-image-ensemble.git
cd medical-image-ensemble
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Configurar las API keys en un archivo `.env`:
```
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
XAI_API_KEY=your_xai_key
OPENROUTER_API_KEY=your_openrouter_key
```

## Estructura del proyecto

```
medical-image-ensemble/
├── src/                      # Código fuente
│   ├── __init__.py
│   ├── models.py             # Gestión de modelos LLM
│   ├── prompts.py            # Definiciones de prompts
│   ├── orchestrator.py       # Orquestación del proceso
│   ├── image_utils.py        # Utilidades para imágenes
│   ├── logger.py             # Sistema de registro
│   └── extract_final_answer.py # Extracción de respuestas finales
│
├── configs/
│   └── config.yaml           # Configuración general
│
├── logs/                     # Directorio para resultados
│
├── run.py                    # Script principal
└── README.md                 # Documentación
```

## Uso

### Ejecución básica

```bash
python run.py
```

### Opciones

```bash
# Ejecutar en modo prueba con una muestra pequeña
python run.py --test --sample 10

# Especificar modelos diferentes
python run.py --advisors claude-3-opus grok-2-vision deepseek-r1-vision --decision-model gpt-4o

# Continuar desde una ejecución interrumpida
python run.py --resume 42  # Continuar desde pregunta ID 42
python run.py --resume logs/20240422_153045/  # Continuar desde un directorio

# Usar un archivo de configuración personalizado
python run.py --config mi_configuracion.yaml
```

## Configuración

El sistema se configura a través del archivo `configs/config.yaml`. Los principales parámetros son:

```yaml
models:
  advisors:
    claude:
      provider: claude
      model: claude-3-5-sonnet-20241022
    # Otros modelos...
  
  decision:
    provider: openai
    model: gpt-4o

prompts:
  # Plantillas de prompts...

dataset:
  path: data/todas_las_preguntas.xlsx
  
logging:
  # Opciones de registro...
```

## Análisis de resultados

Los resultados se guardan en el directorio `logs/[timestamp]/` e incluyen:

- `results.json`: Resultados completos de todas las preguntas.
- `summary.md`: Resumen de métricas en formato Markdown.
- `stats.json`: Estadísticas detalladas en formato JSON.
- `raw/`: Directorio con las respuestas sin procesar de cada modelo.
- Gráficos de rendimiento por categoría.

## Contribuir

Las contribuciones son bienvenidas. Por favor, abre un issue para discutir cambios importantes antes de enviar un pull request.

## Licencia

[MIT](LICENSE)