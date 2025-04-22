"""
Sistema de registro y análisis de resultados.
"""
import os
import json
import time
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

class Logger:
    """
    Clase para el registro y análisis de resultados.
    """
    def __init__(self, config_path='configs/config.yaml'):
        """
        Inicializa el sistema de registro.
        
        Args:
            config_path (str): Ruta al archivo de configuración.
        """
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"logs/{self.timestamp}"
        self.raw_dir = f"{self.results_dir}/raw"
        
        # Crear directorios
        os.makedirs(self.results_dir, exist_ok=True)
        if self.config['logging']['save_raw_responses']:
            os.makedirs(self.raw_dir, exist_ok=True)
        
        # Inicializar lista de resultados
        self.results = []
        
        # Inicializar estadísticas
        self.stats = {
            'start_time': time.time(),
            'total_questions': 0,
            'correct_answers': 0,
            'model_stats': defaultdict(lambda: {'correct': 0, 'total': 0, 'time': 0.0})
        }
    
    def log_result(self, result):
        """
        Registra un resultado.
        
        Args:
            result (dict): Diccionario con los resultados.
        """
        self.results.append(result)
        
        # Actualizar estadísticas
        self.stats['total_questions'] += 1
        
        if result['decision']['is_correct']:
            self.stats['correct_answers'] += 1
        
        # Actualizar estadísticas por modelo
        for advisor_name, advisor_data in result['advisors'].items():
            parsed_answer = advisor_data.get('parsed_answer', '').lower()
            correct_answer = result['correct_answer'].lower()
            
            self.stats['model_stats'][advisor_name]['total'] += 1
            if parsed_answer == correct_answer:
                self.stats['model_stats'][advisor_name]['correct'] += 1
            self.stats['model_stats'][advisor_name]['time'] += advisor_data.get('processing_time', 0)
        
        # Actualizar estadísticas del decisor
        self.stats['model_stats']['decision']['total'] += 1
        if result['decision']['is_correct']:
            self.stats['model_stats']['decision']['correct'] += 1
        self.stats['model_stats']['decision']['time'] += result['decision'].get('processing_time', 0)
        
        # Guardar respuestas sin procesar si está configurado
        if self.config['logging']['save_raw_responses']:
            with open(f"{self.raw_dir}/{result['question_id']}.json", 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        
        # Si es verboso, mostrar progreso
        if self.config['logging']['verbose']:
            print(f"Pregunta {result['question_id']} procesada: {result['decision']['final_answer']} (Correcta: {result['correct_answer']})")
    
    def save_results(self):
        """
        Guarda los resultados acumulados.
        """
        # Guardar resultados completos
        with open(f"{self.results_dir}/results.json", 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        # Generar informes si está configurado
        if self.config['logging']['summary_report']:
            self._generate_summary_report()
        
        if self.config['logging']['category_analysis']:
            self._generate_category_analysis()
        
        # Guardar resumen de estadísticas
        with open(f"{self.results_dir}/stats.json", 'w', encoding='utf-8') as f:
            # Convertir defaultdict a dict para JSON
            stats_dict = {k: dict(v) if isinstance(v, defaultdict) else v for k, v in self.stats.items()}
            json.dump(stats_dict, f, ensure_ascii=False, indent=2)
    
    def _generate_summary_report(self):
        """
        Genera un informe resumen en Markdown.
        """
        # Calcular estadísticas adicionales
        total_time = time.time() - self.stats['start_time']
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        
        # Calcular precisión global
        global_accuracy = 0
        if self.stats['total_questions'] > 0:
            global_accuracy = (self.stats['correct_answers'] / self.stats['total_questions']) * 100
        
        # Calcular precisión por modelo
        model_accuracy = {}
        for model_name, stats in self.stats['model_stats'].items():
            if stats['total'] > 0:
                accuracy = (stats['correct'] / stats['total']) * 100
                avg_time = stats['time'] / stats['total']
                model_accuracy[model_name] = {
                    'correct': stats['correct'],
                    'total': stats['total'],
                    'accuracy': accuracy,
                    'avg_time': avg_time
                }
        
        # Análisis de consenso
        consensus_analysis = self._analyze_consensus()
        
        # Top 10 preguntas más difíciles
        difficult_questions = self._get_difficult_questions(10)
        
        # Generar informe Markdown
        with open(f"{self.results_dir}/summary.md", 'w', encoding='utf-8') as f:
            f.write(f"# Análisis de resultados - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Métricas generales\n")
            f.write(f"- Total de preguntas: {self.stats['total_questions']}\n")
            f.write(f"- Respuestas correctas: {self.stats['correct_answers']}\n")
            f.write(f"- Precisión global: {global_accuracy:.1f}%\n")
            f.write(f"- Tiempo total de procesamiento: {minutes}m {seconds}s\n\n")
            
            f.write("## Rendimiento por modelo\n")
            f.write("| Modelo | Correctas | Precisión | Tiempo promedio |\n")
            f.write("|--------|-----------|-----------|----------------|\n")
            for model_name, stats in model_accuracy.items():
                f.write(f"| {model_name} | {stats['correct']} | {stats['accuracy']:.1f}% | {stats['avg_time']:.2f}s |\n")
            f.write("\n")
            
            if consensus_analysis:
                f.write("## Análisis de consenso\n")
                for consensus_level, stats in consensus_analysis.items():
                    f.write(f"- {consensus_level}: {stats['count']} preguntas ({stats['accuracy']:.1f}% de precisión)\n")
                f.write("\n")
            
            if difficult_questions:
                f.write("## Top 10 preguntas más difíciles\n")
                for i, q in enumerate(difficult_questions, 1):
                    f.write(f"{i}. ID {q['question_id']}: {q['correct_models']}/{len(model_accuracy)} modelos acertaron\n")
                f.write("\n")
    
    def _analyze_consensus(self):
        """
        Analiza el nivel de consenso entre los modelos consejeros.
        
        Returns:
            dict: Estadísticas por nivel de consenso.
        """
        consensus_stats = {
            'Acuerdo total (3/3)': {'count': 0, 'correct': 0},
            'Acuerdo parcial (2/3)': {'count': 0, 'correct': 0},
            'Sin acuerdo (0/3)': {'count': 0, 'correct': 0}
        }
        
        for result in self.results:
            # Contar respuestas de consejeros
            advisor_answers = [
                advisor_data.get('parsed_answer', '').lower()
                for advisor_name, advisor_data in result['advisors'].items()
            ]
            
            # Determinar el nivel de consenso
            answer_counts = Counter(advisor_answers)
            most_common_answer, most_common_count = answer_counts.most_common(1)[0]
            
            if most_common_count == 3:  # Todos de acuerdo
                consensus_level = 'Acuerdo total (3/3)'
            elif most_common_count == 2:  # Dos de acuerdo
                consensus_level = 'Acuerdo parcial (2/3)'
            else:  # Sin acuerdo
                consensus_level = 'Sin acuerdo (0/3)'
            
            consensus_stats[consensus_level]['count'] += 1
            if result['decision']['is_correct']:
                consensus_stats[consensus_level]['correct'] += 1
        
        # Calcular precisión
        for level, stats in consensus_stats.items():
            if stats['count'] > 0:
                stats['accuracy'] = (stats['correct'] / stats['count']) * 100
            else:
                stats['accuracy'] = 0.0
        
        return consensus_stats
    
    def _get_difficult_questions(self, limit=10):
        """
        Obtiene las preguntas más difíciles (donde menos modelos acertaron).
        
        Args:
            limit (int): Número máximo de preguntas a retornar.
            
        Returns:
            list: Lista de preguntas difíciles.
        """
        questions = []
        
        for result in self.results:
            # Contar cuántos modelos acertaron
            correct_models = 0
            
            # Contar aciertos en consejeros
            for advisor_name, advisor_data in result['advisors'].items():
                parsed_answer = advisor_data.get('parsed_answer', '').lower()
                if parsed_answer == result['correct_answer'].lower():
                    correct_models += 1
            
            # Contar acierto del decisor
            if result['decision']['is_correct']:
                correct_models += 1
            
            questions.append({
                'question_id': result['question_id'],
                'correct_models': correct_models,
                'total_models': len(result['advisors']) + 1  # Consejeros + decisor
            })
        
        # Ordenar por número de aciertos (ascendente)
        questions.sort(key=lambda x: x['correct_models'])
        
        return questions[:limit]
    
    def _generate_category_analysis(self):
        """
        Genera un análisis por categorías.
        """
        if not self.results:
            return
        
        # Agrupar resultados por categoría
        category_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        
        for result in self.results:
            category_key = f"{result['category_1']}/{result['category_2']}"
            category_stats[category_key]['total'] += 1
            if result['decision']['is_correct']:
                category_stats[category_key]['correct'] += 1
        
        # Calcular precisión por categoría
        for category, stats in category_stats.items():
            if stats['total'] > 0:
                stats['accuracy'] = (stats['correct'] / stats['total']) * 100
            else:
                stats['accuracy'] = 0.0
        
        # Guardar como CSV
        categories = []
        for category_key, stats in category_stats.items():
            cat1, cat2 = category_key.split('/')
            categories.append({
                'categoria_1': cat1,
                'categoria_2': cat2,
                'total': stats['total'],
                'correctas': stats['correct'],
                'precision': stats['accuracy']
            })
        
        df = pd.DataFrame(categories)
        df.to_csv(f"{self.results_dir}/category_analysis.csv", index=False)
        
        # Generar visualización
        self._generate_category_plots(df)
    
    def _generate_category_plots(self, df):
        """
        Genera gráficos para el análisis por categorías.
        
        Args:
            df (DataFrame): DataFrame con estadísticas por categoría.
        """
        # Configurar estilo
        sns.set(style="whitegrid")
        
        # Crear figura para precisión por categoría principal
        plt.figure(figsize=(12, 6))
        cat1_df = df.groupby('categoria_1').agg({
            'total': 'sum',
            'correctas': 'sum'
        }).reset_index()
        cat1_df['precision'] = (cat1_df['correctas'] / cat1_df['total']) * 100
        
        sns.barplot(x='categoria_1', y='precision', data=cat1_df)
        plt.title('Precisión por Categoría Principal')
        plt.xlabel('Categoría')
        plt.ylabel('Precisión (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/precision_categoria_1.png")
        
        # Crear figura para las categorías secundarias más frecuentes
        top_cat2 = df.groupby('categoria_2')['total'].sum().nlargest(10).index
        cat2_df = df[df['categoria_2'].isin(top_cat2)]
        cat2_df = cat2_df.groupby('categoria_2').agg({
            'total': 'sum',
            'correctas': 'sum'
        }).reset_index()
        cat2_df['precision'] = (cat2_df['correctas'] / cat2_df['total']) * 100
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='categoria_2', y='precision', data=cat2_df)
        plt.title('Precisión por Categoría Secundaria (Top 10)')
        plt.xlabel('Categoría')
        plt.ylabel('Precisión (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/precision_categoria_2.png")
        
        plt.close('all')

# Si el archivo se ejecuta directamente, realizar algunas pruebas
if __name__ == "__main__":
    # Ejemplo de uso
    logger = Logger()
    
    # Ejemplo de resultado
    sample_result = {
        'question_id': 1,
        'question_text': 'Indique la estructura embrionaria contenida en el elemento 7...',
        'image_path': 'imagenes/AnatomiaTopografica/Abdomen/Fig5-4-abd.jpg',
        'correct_answer': 'a',
        'category_1': 'AnatomiaTopografica',
        'category_2': 'Abdomen',
        'advisors': {
            'claude': {
                'raw_response': '{"Respuesta": "a", "Justificacion": "..."}',
                'parsed_answer': 'a',
                'confidence': 0.85,
                'reasoning': '...',
                'processing_time': 2.1
            },
            'grok': {
                'raw_response': '{"Respuesta": "b", "Justificacion": "..."}',
                'parsed_answer': 'b',
                'confidence': 0.6,
                'reasoning': '...',
                'processing_time': 1.8
            },
            'deepseek': {
                'raw_response': '{"Respuesta": "a", "Justificacion": "..."}',
                'parsed_answer': 'a',
                'confidence': 0.75,
                'reasoning': '...',
                'processing_time': 2.3
            }
        },
        'decision': {
            'raw_response': '{"Respuesta": "a", "Justificacion": "..."}',
            'final_answer': 'a',
            'reasoning': '...',
            'is_correct': True,
            'processing_time': 2.0
        }
    }
    
    # Registrar resultado de ejemplo
    logger.log_result(sample_result)
    
    # Guardar resultados
    logger.save_results()
    
    print(f"Resultados guardados en {logger.results_dir}")