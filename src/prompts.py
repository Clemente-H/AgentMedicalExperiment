"""
Módulo para gestionar los prompts utilizados por los diferentes modelos.
"""
import yaml

class PromptManager:
    """
    Clase para gestionar los prompts utilizados en el sistema.
    """
    def __init__(self, config_path='configs/config.yaml'):
        """
        Inicializa el gestor de prompts.
        
        Args:
            config_path (str): Ruta al archivo de configuración.
        """
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        # Cargar templates de prompts como strings normales
        self.advisor_template = self.config['prompts']['advisor_template']
        self.decision_template = self.config['prompts']['decision_template']
    
    def get_advisor_prompt(self, question):
        """
        Genera el prompt para un modelo consejero.
        
        Args:
            question (str): La pregunta a responder.
            
        Returns:
            str: El prompt formateado.
        """
        # Reemplazar directamente la variable
        return self.advisor_template.replace('{question}', question)
    
    def get_decision_prompt(self, question, claude_response, grok_response, deepseek_response):
        """
        Genera el prompt para el modelo decisor.
        
        Args:
            question (str): La pregunta a responder.
            claude_response (str): Respuesta del modelo Claude.
            grok_response (str): Respuesta del modelo Grok.
            deepseek_response (str): Respuesta del modelo DeepSeek.
            
        Returns:
            str: El prompt formateado.
        """
        # Reemplazar directamente todas las variables
        prompt = self.decision_template
        prompt = prompt.replace('{question}', question)
        prompt = prompt.replace('{claude_response}', claude_response)
        prompt = prompt.replace('{grok_response}', grok_response)
        prompt = prompt.replace('{deepseek_response}', deepseek_response)
        return prompt

# Si el archivo se ejecuta directamente, mostrar ejemplos de prompts
if __name__ == "__main__":
    # Ejemplo de uso
    prompt_manager = PromptManager()
    
    # Ejemplo de prompt para consejero
    sample_question = "Indique la estructura embrionaria contenida en el elemento 7: (a) Vena umbilical obliterada (b) Arteria umbilical obliterada (c) Uraco (d) Conducto venoso"
    advisor_prompt = prompt_manager.get_advisor_prompt(sample_question)
    print("PROMPT PARA CONSEJERO:")
    print(advisor_prompt)
    print("\n" + "-"*80 + "\n")
    
    # Ejemplo de prompt para decisor
    sample_claude = '{"Respuesta": "a", "Justificacion": "El elemento 7 muestra la vena umbilical obliterada (ligamento redondo)."}'
    sample_grok = '{"Respuesta": "b", "Justificacion": "El elemento 7 corresponde a la arteria umbilical obliterada."}'
    sample_deepseek = '{"Respuesta": "a", "Justificacion": "El elemento 7 muestra claramente la vena umbilical obliterada."}'
    
    decision_prompt = prompt_manager.get_decision_prompt(
        sample_question,
        sample_claude,
        sample_grok,
        sample_deepseek
    )
    
    print("PROMPT PARA DECISOR:")
    print(decision_prompt)