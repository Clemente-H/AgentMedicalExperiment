"""
Módulo para gestionar las conexiones con los diferentes modelos LLM.
"""
import os
import base64
import yaml
import time
import requests
from pathlib import Path
from typing import Dict, Optional, Union
from anthropic import Anthropic
from openai import OpenAI
from dotenv import load_dotenv

class ModelClient:
    """
    Clase base para los clientes de modelos LLM.
    """
    def __init__(self, provider=None, model=None, temperature=0.1):
        """
        Inicializa el cliente base para modelos.
        
        Args:
            provider (str): El proveedor del modelo (claude, openai, openrouter, grok).
            model (str): El nombre específico del modelo.
            temperature (float): Temperatura para la generación (0.0 - 1.0).
        """
        # Cargar variables de entorno
        load_dotenv()
        
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.client = None
        self.setup_client()
    
    def setup_client(self):
        """
        Configura el cliente específico según el proveedor.
        """
        if self.provider == "claude":
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY no encontrada en .env")
            self.client = Anthropic(api_key=self.api_key)
            
        elif self.provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY no encontrada en .env")
            self.client = OpenAI(api_key=self.api_key)
        
        elif self.provider == "openrouter":
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            if not self.api_key:
                raise ValueError("OPENROUTER_API_KEY no encontrada en .env")
        
        elif self.provider == "grok":
            self.api_key = os.getenv("XAI_API_KEY")
            if not self.api_key:
                raise ValueError("XAI_API_KEY no encontrada en .env")
            self.client = OpenAI(api_key=self.api_key, base_url="https://api.x.ai/v1")
        else:
            raise ValueError("El proveedor debe ser 'claude', 'openai', 'openrouter', o 'grok'")

    def _encode_image(self, image_path: str) -> str:
        """
        Convierte una imagen a base64.
        
        Args:
            image_path (str): Ruta a la imagen.
            
        Returns:
            str: Cadena base64 de la imagen.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def query_model(self, image_path, prompt):
        """
        Método base para consultar el modelo. Será sobrescrito por subclases.
        """
        raise NotImplementedError("Esta función debe ser implementada por subclases")


class AdvisorModel(ModelClient):
    """
    Clase para modelos que actúan como consejeros.
    """
    def query_model(self, image_path, prompt):
        """
        Consulta al modelo consejero con una imagen y un prompt.
        
        Args:
            image_path (str): Ruta a la imagen.
            prompt (str): Prompt para el modelo.
            
        Returns:
            dict: Contiene la respuesta completa y el tiempo de procesamiento.
        """
        print(f"\nDEBUG - [{self.provider}] Prompt enviado:\n{prompt[:500]}...\n")  # Primeros 500 caracteres
        image_b64 = self._encode_image(image_path)
        start_time = time.time()
        
        try:
            if self.provider == "claude":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    temperature=self.temperature,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_b64
                                }
                            }
                        ]
                    }]
                )
                raw_response = response.content[0].text
                
            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_b64}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=1000
                )
                raw_response = response.choices[0].message.content
            
            elif self.provider == "openrouter":
                # Ajustar el formato para OpenRouter
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://medical-image-ensemble.com",
                    },
                    json={
                        "model": self.model,
                        "temperature": self.temperature,
                        "max_tokens": 1000,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": prompt
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{image_b64}"
                                        }
                                    }
                                ]
                            }
                        ]
                    }
                )
                raw_response = response.json()['choices'][0]['message']['content']
            
            elif self.provider == "grok":
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_b64}",
                                        "detail": "high"
                                    }
                                }
                            ]
                        }
                    ]
                )
                raw_response = response.choices[0].message.content

            processing_time = time.time() - start_time
            
            return {
                "raw_response": raw_response,
                "processing_time": processing_time
            }
                
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                "raw_response": f"Error: {str(e)}",
                "processing_time": processing_time,
                "error": True
            }


class DecisionModel(ModelClient):
    """
    Clase para el modelo que actúa como decisor final.
    """
    def query_model(self, image_path, prompt):
        """
        Consulta al modelo decisor con una imagen y un prompt.
        
        Args:
            image_path (str): Ruta a la imagen.
            prompt (str): Prompt para el modelo.
            
        Returns:
            dict: Contiene la respuesta completa y el tiempo de procesamiento.
        """
        image_b64 = self._encode_image(image_path)
        start_time = time.time()
        
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_b64}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=1000
                )
                raw_response = response.choices[0].message.content
            
            else:
                # En caso de que se use otro proveedor como decisor
                # Usamos la misma implementación que para los consejeros
                advisor = AdvisorModel(provider=self.provider, model=self.model, temperature=self.temperature)
                result = advisor.query_model(image_path, prompt)
                raw_response = result["raw_response"]
                processing_time = result["processing_time"]
                return {
                    "raw_response": raw_response,
                    "processing_time": processing_time,
                    "error": "error" in result
                }

            processing_time = time.time() - start_time
            
            return {
                "raw_response": raw_response,
                "processing_time": processing_time
            }
                
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                "raw_response": f"Error: {str(e)}",
                "processing_time": processing_time,
                "error": True
            }


class ModelManager:
    """
    Clase para gestionar los diferentes modelos del sistema.
    """
    def __init__(self, config_path='configs/config.yaml'):
        """
        Inicializa el gestor de modelos.
        
        Args:
            config_path (str): Ruta al archivo de configuración.
        """
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        self.advisor_models = {}
        self.decision_model = None
        
        self._initialize_models()
    
    def _initialize_models(self):
        """
        Inicializa todos los modelos consejeros y el modelo decisor.
        """
        # Inicializar modelos consejeros
        for name, model_config in self.config['models']['advisors'].items():
            self.advisor_models[name] = AdvisorModel(
                provider=model_config['provider'],
                model=model_config['model'],
                temperature=model_config.get('temperature', 0.1)
            )
        
        # Inicializar modelo decisor
        decision_config = self.config['models']['decision']
        self.decision_model = DecisionModel(
            provider=decision_config['provider'],
            model=decision_config['model'],
            temperature=decision_config.get('temperature', 0.1)
        )
    
    def get_advisor(self, name):
        """
        Obtiene un modelo consejero por nombre.
        
        Args:
            name (str): Nombre del consejero (claude, grok, deepseek).
            
        Returns:
            AdvisorModel: El modelo consejero.
        """
        return self.advisor_models.get(name)
    
    def get_decision_model(self):
        """
        Obtiene el modelo decisor.
        
        Returns:
            DecisionModel: El modelo decisor.
        """
        return self.decision_model
    
    def get_all_advisors(self):
        """
        Obtiene todos los modelos consejeros.
        
        Returns:
            dict: Diccionario con todos los modelos consejeros.
        """
        return self.advisor_models

# Si el archivo se ejecuta directamente, mostrar información de los modelos
if __name__ == "__main__":
    manager = ModelManager()
    
    print("Modelos consejeros:")
    for name, model in manager.get_all_advisors().items():
        print(f"- {name}: {model.model} (Proveedor: {model.provider})")
    
    decision_model = manager.get_decision_model()
    print(f"\nModelo decisor: {decision_model.model} (Proveedor: {decision_model.provider})")