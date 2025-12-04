import torch
import numpy as np
import pickle
from scipy.special import expit
from transformers import AutoModelForCausalLM, AutoTokenizer

from vectorized_db import Embedder

def get_category(text):
    if any(word in text for word in ['суп', 'борщ', 'щи', 'солянка']):
        return 'суп'
    elif any(word in text for word in ['салат', 'закуска']):
        return 'салат'
    elif any(word in text for word in ['торт', 'десерт', 'печенье', 'пирог']):
        return 'десерт'
    elif any(word in text for word in ['второе', 'гарнир', 'мясо', 'рыба', 'курица']):
        return 'основное блюдо'
    elif any(word in text for word in ['блины', 'оладьи', 'выпечка', 'булка']):
        return 'выпечка'
    return ''

class RecipeRAGSystem:
    def __init__(self, dataset, db_file, emb_model_name, gen_model_name):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dataset = dataset
        with open(db_file, 'rb') as f:
            self.database = pickle.load(f)

        self.embedder = Embedder(emb_model_name, device=self.device)

        self.gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
        self.gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name).to(self.device)
        self.gen_model.eval()

    def retrieve_relevant_recipes(self, query, top_k):
        query_embedding = self.embedder.encode([query])[0]

        results, indices = self.database.search(
            query=query_embedding,
            num_results=top_k,
            use_lsh=True,
            return_indices=True
        )

        distances = np.linalg.norm(results - query_embedding, axis=1)
        retrieved_recipes = dict()

        for res_idx, distance in zip(indices, distances):
            chunk_info = self.parse_recipe(self.database.metadata[res_idx]['id'])
            chunk_info.update({
                # 'chunk_id': res_idx,
                # 'text': self.database.metadata[res_idx]['text'],  # chunk
                'relevance_score': 10 * expit(1 / distance),
            })

            recipe_id = chunk_info['recipe_id']
            if recipe_id in retrieved_recipes:
                retrieved_recipes[recipe_id]['relevance_score'] = max(retrieved_recipes[recipe_id]['relevance_score'],
                                                                      chunk_info['relevance_score'])
            else:
                retrieved_recipes[recipe_id] = chunk_info

        return list(retrieved_recipes.values())

    def parse_recipe(self, recipe_id):
        name = components['name'][recipe_id]
        instructions = self.dataset['text'][recipe_id]
        all_text = name + '\n' + instructions
        
        components = {
            'recipe_id': recipe_id,
            'name': name,
            'ingredients': self.dataset['ingredients'][recipe_id][1:-1].replace("'", ""),
            'instructions': instructions,
            'category': get_category(all_text.lower())
        }

        return components

    def build_recipes_prompt(self, retrieved_recipes):
        prompt = """
        Доступные рецепты:\n\n
        """

        for i, recipe in enumerate(retrieved_recipes):
            prompt += f"Пример {i + 1}:\n"
            if recipe['relevance_score'] is not None:
                prompt += f"Релевантность: {recipe['relevance_score']}"

            prompt += f"Название блюда: {recipe['name']}\n"

            if len(recipe['category']):
                prompt += f"Категория: {recipe['category']}\n"

            prompt += f"Ингредиенты: {recipe['ingredients']}\n"
            prompt += f"Рецепт: {recipe['instructions']}\n\n"

        return prompt

    def generate_answer(self, recipes_prompt, query):
        messages = [
            {"role": "system",
             "content": """Ты - кулинарный помощник, который пишет рецепты блюд.
             Тебе предоставлено несколько примеров с рецептами похожих блюд и их релевантностью.
             Используй ТОЛЬКО информацию из предоставленных примеров рецептов. 
             Релевантность рецептов в примерах оценена в 10-балльной шкале (чем больше, тем лучше).
             Для блюда, которое указал пользователь, пиши список ингредиентов и рецепт с пошаговой инструкцией по приготовлению блюда.
             Список ингредиентов должен быть небольшим, а рецепт - точным и информативным. 
             Не пиши лишнего, твой ответ должен содержать только список ингредиентов и инструкцию по приготовлению указанного блюда. 
             
             """
             },
            {"role": "system", "content": recipes_prompt},
            {"role": "user", "content": query}
        ]
        text = self.gen_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.gen_tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.gen_model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.gen_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response

    def answer_query(self, query, top_k_retrieve=5, rel_threshold=0, print_metainfo=False, use_web=False):
        if print_metainfo:
            print("\n1. Поиск релевантных фрагментов...")
        retrieved_recipes = self.retrieve_relevant_recipes(query, top_k=top_k_retrieve)
        retrieved_recipes = list(filter(lambda item: item['relevance_score'] >= rel_threshold))

        if print_metainfo:
            print(f"   Найдено {len(retrieved_recipes)} релевантных фрагментов:")
            for i, chunk in enumerate(retrieved_recipes):
                print(
                    f"   {i + 1}. (№{chunk['recipe_id']}) Оценка: {chunk['relevance_score']:.2f} {chunk.get('name', 'Без названия')}")

        if use_web:
            # TODO
            web_recipes = []  # TODO
            web_recipes = web_recipes[:top_k_retrieve - len(retrieved_recipes)]
            
            pass

        if print_metainfo:
            print("\n2. Создание промпта с фрагментами рецептов...")
        recipes_prompt = self.build_recipes_prompt(retrieved_recipes)

        if print_metainfo:
            print("\n3. Генерация ответа...")
        answer = self.generate_answer(recipes_prompt, query)

        return answer
