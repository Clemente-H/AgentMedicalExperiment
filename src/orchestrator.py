"""
Orquestación del proceso completo de análisis de imágenes.
"""
import os
import json
import time
import yaml
import pandas as pd
import concurrent.futures
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Tuple

from src.models import ModelManager
from src.prompts import PromptManager
from src.image_utils import is_valid_image
from src.logger import Logger

class ModelEnsemble:
    """
    Clase para orquestar el proceso completo de análisis de imágenes.
    """
    def __init__(self, config_path='configs/config.yaml'):
        """
        Inicializa el orquestador.
        
        Args:
            config_path (str): Ruta al archivo de configuración.
        """
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        # Inicializar componentes
        self.model_manager = ModelManager(config_path)
        self.prompt_manager = PromptManager(config_path)
        self.logger = Logger(config_path)
        
        # Información del dataset
        self.dataset_path = self.config['dataset']['path']
        self.image_base_path = self.config['dataset']['image_base_path']
    
    def load_dataset(self):
        """
        Carga el dataset de preguntas.
        
        Returns:
            DataFrame: DataFrame con las preguntas.
        """
        try:
            return pd.read_excel(self.dataset_path)
        except Exception as e:
            raise ValueError(f"Error al cargar el dataset: {str(e)}")
    
    def _parse_response(self, response: str) -> Tuple[str, str]:
        """
        Parsea la respuesta de un modelo para extraer la alternativa seleccionada.
        
        Args:
            response (str): Respuesta completa del modelo.
            
        Returns:
            tuple: (alternativa, justificación)
        """
        # Si la respuesta está vacía o es un error
        if not response or response.startswith("Error:"):
            return "", ""
        
        # Limpiar la respuesta - eliminar comillas triples y json code blocks
        clean_response = response
        if "```json" in clean_response:
            clean_response = clean_response.replace("```json", "").replace("```", "").strip()
        elif "```" in clean_response:
            clean_response = clean_response.replace("```", "").strip()
        
        try:
            # Intentar parsear como JSON
            import json
            data = json.loads(clean_response)
            
            # Buscar la respuesta en diferentes formatos de claves
            answer = ""
            justification = ""
            
            # Claves comunes para respuesta
            answer_keys = ["respuesta", "Respuesta", "response", "Response", "answer", "Answer"]
            for key in answer_keys:
                if key in data:
                    answer = data[key]
                    break
            
            # Claves comunes para justificación
            justification_keys = ["justificacion", "Justificacion", "justification", "Justification", "reasoning", "Reasoning"]
            for key in justification_keys:
                if key in data:
                    justification = data[key]
                    break
            
            # Verificar formato de la respuesta
            if answer and isinstance(answer, str):
                answer = answer.strip().lower()
                # Si la respuesta es más larga, extraer solo la primera letra si es a,b,c,d
                if len(answer) > 1 and answer[0] in 'abcd':
                    answer = answer[0]
                return answer, justification
                
        except json.JSONDecodeError:
            # Si no podemos parsear como JSON, buscar patrones en el texto
            pass
        
        # Buscar patrones en texto plano si fallamos con JSON
        import re
        response_lower = response.lower()
        
        # Buscar patrones comunes para la respuesta
        patterns = [
            r"(?:respuesta|answer):\s*[\"']?([a-d])[\"']?",
            r"(?:opción|option):\s*[\"']?([a-d])[\"']?",
            r"(?:alternativa|alternative):\s*[\"']?([a-d])[\"']?",
            r"[\"'](?:respuesta|answer)[\"']:\s*[\"']([a-d])[\"']",
            r"[\"'](?:opción|option)[\"']:\s*[\"']([a-d])[\"']",
            r"(?:la respuesta correcta es|the correct answer is)\s+[\"']?([a-d])[\"']?",
            r"(?:elijo|i choose)\s+[\"']?([a-d])[\"']?"
        ]
        
        # Probar cada patrón
        for pattern in patterns:
            match = re.search(pattern, response_lower)
            if match:
                # Buscar una justificación después de la respuesta
                justification_match = re.search(r"(?:justificacion|justification|reasoning):\s*[\"']?(.*?)[\"']?(?:\}|\n|$)", response_lower)
                justification = justification_match.group(1).strip() if justification_match else ""
                return match.group(1), justification
        
        # Si llegamos aquí, buscar la primera letra a,b,c,d en el texto
        for char in response_lower:
            if char in 'abcd':
                return char, ""
        
        # Si todo falla, devolver valores vacíos
        return "", ""
    
    def _query_advisor(self, advisor_name, advisor, image_path, question):
        """
        Consulta a un modelo consejero.
        
        Args:
            advisor_name (str): Nombre del consejero.
            advisor: Instancia del modelo consejero.
            image_path (str): Ruta a la imagen.
            question (str): Pregunta a responder.
            
        Returns:
            dict: Información de la respuesta del consejero.
        """
        # Obtener el prompt específico
        prompt = self.prompt_manager.get_advisor_prompt(question)
        
        # Consultar al modelo
        response_data = advisor.query_model(image_path, prompt)
        raw_response = response_data["raw_response"]
        processing_time = response_data["processing_time"]
        
        # Parsear la respuesta
        parsed_answer, reasoning = self._parse_response(raw_response)
        
        return {
            "raw_response": raw_response,
            "parsed_answer": parsed_answer,
            "reasoning": reasoning,
            "processing_time": processing_time,
            "error": response_data.get("error", False)
        }
    
    def _query_advisors_parallel(self, image_path, question):
        """
        Consulta a todos los modelos consejeros en paralelo.
        
        Args:
            image_path (str): Ruta a la imagen.
            question (str): Pregunta a responder.
            
        Returns:
            dict: Respuestas de todos los consejeros.
        """
        advisor_responses = {}
        advisors = self.model_manager.get_all_advisors()
        
        # Usar threads para consultas paralelas
        with ThreadPoolExecutor(max_workers=len(advisors)) as executor:
            # Preparar las tareas
            future_to_advisor = {
                executor.submit(self._query_advisor, advisor_name, advisor, image_path, question): advisor_name
                for advisor_name, advisor in advisors.items()
            }
            
            # Recopilar resultados
            for future in concurrent.futures.as_completed(future_to_advisor):
                advisor_name = future_to_advisor[future]
                try:
                    advisor_responses[advisor_name] = future.result()
                except Exception as e:
                    print(f"Error al consultar al consejero {advisor_name}: {str(e)}")
                    advisor_responses[advisor_name] = {
                        "raw_response": f"Error: {str(e)}",
                        "parsed_answer": "",
                        "reasoning": "",
                        "processing_time": 0,
                        "error": True
                    }
        
        return advisor_responses
    
    def _query_decision_model(self, image_path, question, advisor_responses):
        """
        Consulta al modelo decisor.
        
        Args:
            image_path (str): Ruta a la imagen.
            question (str): Pregunta a responder.
            advisor_responses (dict): Respuestas de los consejeros.
            
        Returns:
            dict: Decisión final.
        """
        # Obtener respuestas de consejeros
        claude_response = advisor_responses.get("claude", {}).get("raw_response", "Error al obtener respuesta")
        grok_response = advisor_responses.get("grok", {}).get("raw_response", "Error al obtener respuesta")
        deepseek_response = advisor_responses.get("deepseek", {}).get("raw_response", "Error al obtener respuesta")
        
        # Obtener el prompt para el decisor
        prompt = self.prompt_manager.get_decision_prompt(
            question,
            claude_response,
            grok_response,
            deepseek_response
        )
        
        # Consultar al modelo decisor
        decision_model = self.model_manager.get_decision_model()
        response_data = decision_model.query_model(image_path, prompt)
        raw_response = response_data["raw_response"]
        processing_time = response_data["processing_time"]
        
        # Parsear la respuesta
        final_answer, reasoning = self._parse_response(raw_response)
        
        return {
            "raw_response": raw_response,
            "final_answer": final_answer,
            "reasoning": reasoning,
            "processing_time": processing_time,
            "error": response_data.get("error", False)
        }
    
    def process_question(self, row):
        """
        Procesa una pregunta completa.
        
        Args:
            row (Series): Fila del DataFrame con la información de la pregunta.
            
        Returns:
            dict: Resultado completo del procesamiento.
        """
        # Extraer información de la pregunta
        question_id = row.name
        question_text = row['pregunta']
        correct_answer = row['respuesta_correcta'].lower()
        image_path = os.path.join(self.image_base_path, row['ruta'])
        category_1 = row['categoria_1']
        category_2 = row['categoria_2']
        
        # Verificar que la imagen sea válida
        print(f"DEBUG - Verificando imagen: {image_path} (Existe: {os.path.exists(image_path)})")
        valid, reason = is_valid_image(image_path)
        if not valid:
            print(f"Imagen no válida para pregunta {question_id}: {reason}")
            return None
        
        # 1. Consultar a los modelos consejeros
        print(f"Procesando pregunta {question_id}...")
        start_time = time.time()
        advisor_responses = self._query_advisors_parallel(image_path, question_text)
        
        # 2. Consultar al modelo decisor
        decision = self._query_decision_model(image_path, question_text, advisor_responses)
        
        # 3. Determinar si la respuesta es correcta
        decision["is_correct"] = decision["final_answer"].lower() == correct_answer.lower()
        
        # 4. Construir el resultado completo
        result = {
            "question_id": question_id,
            "question_text": question_text,
            "image_path": image_path,
            "correct_answer": correct_answer,
            "category_1": category_1,
            "category_2": category_2,
            "advisors": advisor_responses,
            "decision": decision,
            "processing_time": {
                "total": time.time() - start_time
            }
        }
        
        return result
    
    def run(self, test_mode=False, sample_size=None, resume_from=None):
        """
        Ejecuta el experimento completo.
        
        Args:
            test_mode (bool): Si es True, muestra información adicional.
            sample_size (int): Número de muestras a procesar (para testing).
            resume_from (int): ID de pregunta desde donde continuar.
            
        Returns:
            str: Ruta al directorio de resultados.
        """
        # Cargar dataset
        df = self.load_dataset()
        
        # Limitar muestra si es necesario
        if sample_size and sample_size < len(df):
            df = df.iloc[:sample_size]
        
        # Filtrar si se está reanudando
        if resume_from is not None:
            df = df[df.index >= resume_from]
        
        # Procesar cada pregunta
        for idx, row in df.iterrows():
            # Procesar la pregunta
            result = self.process_question(row)
            
            # Registrar resultado
            if result:
                self.logger.log_result(result)
            
            # En modo test, mostrar más información
            if test_mode and result:
                print(f"\nPregunta: {result['question_text']}")
                print(f"Respuesta correcta: {result['correct_answer']}")
                print("Respuestas de consejeros:")
                for advisor_name, data in result['advisors'].items():
                    print(f"  - {advisor_name}: {data['parsed_answer']} (Tiempo: {data['processing_time']:.2f}s)")
                print(f"Decisión final: {result['decision']['final_answer']} (Correcta: {result['decision']['is_correct']})")
                print(f"Tiempo total: {result['processing_time']['total']:.2f}s")
                print("-" * 80)
        
        # Guardar resultados
        self.logger.save_results()
        
        return self.logger.results_dir

# Si el archivo se ejecuta directamente, realizar una prueba con una muestra pequeña
if __name__ == "__main__":
    # Ejemplo de uso
    ensemble = ModelEnsemble()
    
    # Ejecutar con una muestra pequeña en modo test
    results_dir = ensemble.run(test_mode=True, sample_size=2)
    
    print(f"\nExperimento completado. Resultados guardados en: {results_dir}")