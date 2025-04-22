"""
Módulo para extraer las respuestas finales de los resultados.
"""
import os
import json
import re
import pandas as pd
from pathlib import Path


def extract_answer_from_text(text):
    """
    Extrae la alternativa de respuesta de un texto.
    
    Args:
        text (str): Texto que contiene la respuesta.
        
    Returns:
        str: Alternativa de respuesta (a, b, c, d) o cadena vacía si no se encuentra.
    """
    # Patrones comunes para encontrar respuestas
    patterns = [
        r'(?:respuesta|answer):\s*["\']?([a-d])["\']?',  # "Respuesta: a" o 'Answer: "b"'
        r'(?:alternativa|option):\s*["\']?([a-d])["\']?',  # "Alternativa: c" o 'Option: "d"'
        r'["\']respuesta["\']:\s*["\']([a-d])["\']',  # "respuesta": "a"
        r'["\']answer["\']:\s*["\']([a-d])["\']',  # "answer": "b"
        r'(?:la respuesta correcta es|the correct answer is)\s*["\']?([a-d])["\']?',  # "La respuesta correcta es: c"
        r'(?:elijo|opto por|i choose)\s*["\']?([a-d])["\']?',  # "Elijo a" o "I choose b"
        r'^["\']?([a-d])["\']?  # Solo la letra
    ]
    
    # Buscar con expresiones regulares
    text_lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    # Si no encontramos nada con los patrones anteriores
    # Buscar la primera ocurrencia de a, b, c o d
    for char in text_lower:
        if char in 'abcd':
            return char
    
    return ""


def parse_json_response(response):
    """
    Intenta extraer la respuesta de un texto en formato JSON.
    
    Args:
        response (str): Texto en formato JSON.
        
    Returns:
        str: Alternativa de respuesta o cadena vacía si no se puede extraer.
    """
    try:
        # Intentar parsear directamente
        data = json.loads(response)
        
        # Buscar claves comunes que pueden contener la respuesta
        possible_keys = ['respuesta', 'answer', 'opcion', 'option', 'alternativa']
        for key in possible_keys:
            for data_key in data.keys():
                if key.lower() in data_key.lower():
                    answer = data[data_key]
                    if isinstance(answer, str) and len(answer) > 0:
                        if answer[0].lower() in 'abcd':
                            return answer[0].lower()
                        else:
                            # Buscar a, b, c, d en el valor
                            return extract_answer_from_text(answer)
        
        # Si no encontramos nada en claves específicas, buscar en todo el objeto
        response_text = json.dumps(data)
        return extract_answer_from_text(response_text)
        
    except json.JSONDecodeError:
        # Si no es JSON válido, intentar extraer de texto plano
        return extract_answer_from_text(response)


def extract_final_answers(input_file, output_file):
    """
    Extrae las respuestas finales de los resultados y guarda un nuevo archivo.
    
    Args:
        input_file (str): Ruta al archivo de resultados.
        output_file (str): Ruta donde guardar el archivo procesado.
    """
    results = []
    
    # Leer archivo línea por línea
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # Cargar JSON de la línea
                result = json.loads(line.strip())
                
                # Extraer la respuesta del modelo
                model_response = result.get('respuesta_modelo', '')
                
                # Intentar extraer la alternativa
                final_answer = parse_json_response(model_response)
                
                # Agregar al resultado
                result['respuesta_extraida'] = final_answer
                
                # Comparar con la respuesta correcta
                correcta = result.get('respuesta_correcta', '').lower()
                if correcta and final_answer:
                    result['es_correcta'] = final_answer.lower() == correcta.lower()
                else:
                    result['es_correcta'] = False
                
                results.append(result)
                
            except json.JSONDecodeError:
                print(f"Error al procesar línea en {input_file}")
                continue
    
    # Guardar resultados procesados
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')
    
    # Generar estadísticas
    if results:
        total = len(results)
        correctas = sum(1 for r in results if r.get('es_correcta', False))
        precision = (correctas / total) * 100 if total > 0 else 0
        
        print(f"Archivo procesado: {output_file}")
        print(f"Total de preguntas: {total}")
        print(f"Respuestas correctas: {correctas}")
        print(f"Precisión: {precision:.2f}%")
        
        # Guardar resumen en CSV
        output_base = os.path.splitext(output_file)[0]
        stats_file = f"{output_base}_stats.csv"
        
        # Agrupar por categorías
        categoria_stats = {}
        for r in results:
            cat1 = r.get('categoria_1', 'Unknown')
            cat2 = r.get('categoria_2', 'Unknown')
            key = f"{cat1}_{cat2}"
            
            if key not in categoria_stats:
                categoria_stats[key] = {'total': 0, 'correctas': 0}
                
            categoria_stats[key]['total'] += 1
            if r.get('es_correcta', False):
                categoria_stats[key]['correctas'] += 1
        
        # Convertir a DataFrame
        stats_data = []
        for key, stats in categoria_stats.items():
            cat_parts = key.split('_')
            cat1 = cat_parts[0] if len(cat_parts) > 0 else 'Unknown'
            cat2 = cat_parts[1] if len(cat_parts) > 1 else 'Unknown'
            
            precision_cat = (stats['correctas'] / stats['total']) * 100 if stats['total'] > 0 else 0
            
            stats_data.append({
                'categoria_1': cat1,
                'categoria_2': cat2,
                'total': stats['total'],
                'correctas': stats['correctas'],
                'precision': precision_cat
            })
        
        if stats_data:
            df = pd.DataFrame(stats_data)
            df.to_csv(stats_file, index=False)
            print(f"Estadísticas por categoría guardadas en: {stats_file}")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python extract_final_answer.py <archivo_entrada> [archivo_salida]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        # Generar nombre de salida automáticamente
        input_path = Path(input_file)
        output_file = str(input_path.with_name(f"{input_path.stem}_processed{input_path.suffix}"))
    
    extract_final_answers(input_file, output_file)