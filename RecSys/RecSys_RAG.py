from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from getpass import getpass
from time import time
import pandas as pd
import numpy as np
import random
import torch
import os
import re

class RecSys:
    def __init__(self, model:str, vector_db:str, access_token:str, product_description:str, k:int = 15, use_4bit:bool = False):
        self.model = model
        self.vector_db = vector_db
        self.product_description = product_description
        self.k = k
        self.use_4bit = use_4bit
        self.device = ''
        self.access_token = access_token
        self.prompt_in_chat_format = [
            {
                "role": "system",
                "content": """Using the information contained in context, give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Information of recommended products must be correct and matched in context, do not falsify information.
If the answer cannot be deduced from the context, do not give an answer.

Response must include product id, title, and reason for recommendation.
Response must strictly follow the template below:
i. **Product ID: <Product ID>** - <Title>
Reason: <Reason>

Answer examples:
1. **Product ID: B0C3WNM5X7** - Simple Joys by Carter's Toddler Boys' Hooded Sweater Jacket with Sherpa Lining
Reason: This product is highly rated with an average rating of 4.8, offering excellent value for its price.

2. **Product ID: B0C1X12894** - Oversized Wearable Blanket Hoodie for Women Men Comfy Sweatshirt
Reason: This product is highly rated with an average rating of 4.8, making it a great option for those looking for a cozy and warm garment.

3. **Product ID: B0C68CBFKS** - Columbia Women's West Bend Hoodie
Reason: This product is highly rated with an average rating of 4.4, offering a reliable and affordable option for casual wear.""",
            },
            {
                "role": "user",
                "content": """Context:
{context}

---

Now here is the question you need to answer.

Question: {question}""",
            },
        ]

    def initialize(self):
        # Set hardware
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available else "cpu")
        print(f"[*] Using device: {self.device}")

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        match self.model:
            case "Llama3.2":
                model_id = "meta-llama/Llama-3.2-1B-Instruct"
            case "Qwen2.5":
                model_id = "Qwen/Qwen2.5-1.5B-Instruct"

        '''Tokenizer and Model'''
        # Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=self.access_token)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        print("[*] Tokenizer loaded.")

        # Load Model
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        if self.use_4bit:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                token=self.access_token,
                quantization_config=bnb_config,
            ).to(self.device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                token=self.access_token,
            ).to(self.device)

        print("[*] Model loaded.")

        return tokenizer, model

    def load_vector_db(self):
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            multi_process=True,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
        )

        loaded_vector_database = FAISS.load_local(
            self.vector_db,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True,
        )

        print('[*] Vector database loaded.')

        return loaded_vector_database

    def recommend(self):
        print("[*] Initrializing...")
        tokenizer, model = self.initialize()
        KNOWLEDGE_VECTOR_DATABASE = self.load_vector_db()
        print('[*] Done.')

        '''Prompt Template'''
        RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
            self.prompt_in_chat_format, tokenize=False, add_generation_prompt=True
        )

        '''Retrieve Similar Products from Vector Database'''
        print(f'[*] Retrieving {self.k} similar products...')
        retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=self.product_description, k=self.k+1)[1:]  # The first one will always be the qurey one, so skip it.
        retrieved_docs_text = [doc.metadata['text'] for doc in retrieved_docs]
        print('[*] Done.')

        '''Format Prompt'''
        print('[*] Formatting prompt...')
        context = "\nExtracted products:"
        context += "".join(
            [f"\n\nProduct {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)]
        )

        final_prompt = RAG_PROMPT_TEMPLATE.format(
            question="Base on this product, recommend 5 best products from Context.", context=context
        )

        # print(f'\n[!] Prompt: \n{final_prompt}\n')

        print('[*] Done.')

        '''Generate Recommendations'''
        Rec_LLM = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            do_sample=True,
            temperature=0.2,
            repetition_penalty=1.1,
            return_full_text=False,
            max_new_tokens=1000,
            device=self.device,
        )

        print('[*] Generating Recommendations...')
        recommedations = Rec_LLM(final_prompt)[0]["generated_text"]
        print('[*] Done.')

        return recommedations, retrieved_docs_text, KNOWLEDGE_VECTOR_DATABASE


class RecInterpreter:
    def __init__(self, KNOWLEDGE_VECTOR_DATABASE, recommendations:str, target_id:str):
        self.faiss_idx = KNOWLEDGE_VECTOR_DATABASE.index
        self.metadata = KNOWLEDGE_VECTOR_DATABASE.docstore._dict
        self.vectors = self.faiss_idx.reconstruct_n(0, self.faiss_idx.ntotal)
        self.recommendations = recommendations
        self.target_id = target_id

    def preprocess(self):
        '''vector database'''
        pca = PCA(n_components=3)
        reduced_vectors = pca.fit_transform(self.vectors)

        vector_categories = {}
        for i, v in enumerate(self.metadata.values()):
            curr_category = v.metadata['category']
            vector_categories.setdefault(curr_category, [])
            vector_categories[curr_category].append(reduced_vectors[i])

        # sort in descending by category length
        sorted_vector_categories = dict(sorted(vector_categories.items(), key=lambda item: len(item[1]), reverse=True))

        '''recommendations'''
        pattern = re.compile(r"\*\*Product ID: (.*)\*\*")
        ids = [x.split()[-1] for x in pattern.findall(self.recommendations)]

        # Check IDs
        # print(f'[!] recommended IDs: {ids}')

        recommedation_vectors = {}
        target_retrieved, recommedations_retrieved = 0, 0
        for i, v in enumerate(self.metadata.values()):
            curr_id = v.metadata['id']
            if curr_id == self.target_id and target_retrieved == 0:
                recommedation_vectors.setdefault('Target Product', [])
                recommedation_vectors['Target Product'].append(reduced_vectors[i])
                target_retrieved += 1
                
            elif curr_id in ids and recommedations_retrieved < 5:
                recommedation_vectors.setdefault('Recommended Product', [])
                recommedation_vectors['Recommended Product'].append(reduced_vectors[i])
                recommedations_retrieved += 1

        # Check
        # for k, v in recommedation_vectors.items():
        #     print(f'[!] {k}: {len(v)}')

        return sorted_vector_categories, recommedation_vectors

    def vis_3d(self):
        sorted_vector_categories, recommedation_vectors = self.preprocess()

        # Plot in 3D
        fig = plt.figure(figsize=(40, 20))

        '''Vector Database'''
        ax1 = fig.add_subplot(121, projection='3d')

        for idx, (subset_name, subset_vectors) in enumerate(sorted_vector_categories.items()):
            subset_vectors = np.array(subset_vectors)

            # Reduce background size
            # subset_vectors_len = len(subset_vectors)
            # if subset_vectors_len >= 1000:
            #     subset_vectors = subset_vectors[:1000]
            # else:
            #     subset_vectors = subset_vectors[:int(round(subset_vectors_len/2,0))]

            ax1.scatter(
                subset_vectors[:, 0],
                subset_vectors[:, 1],
                subset_vectors[:, 2],
                label=subset_name.lower().title(),  # Add label for the legend
                alpha=0.3,
            )

        ax1.set_title("3D Visualization of Products Vector Database (Whole)", fontsize=40)
        ax1.tick_params(axis='x', labelsize=15)
        ax1.tick_params(axis='y', labelsize=15)
        ax1.tick_params(axis='z', labelsize=15)
        ax1.legend(fontsize=15)

        '''Target & Recommendation'''
        ax2 = fig.add_subplot(122, projection='3d')

        # Vector Database
        for idx, (subset_name, subset_vectors) in enumerate(sorted_vector_categories.items()):
            subset_vectors = np.array(subset_vectors)

            # Reduce background size
            subset_vectors_len = len(subset_vectors)
            if subset_vectors_len >= 1000:
                subset_vectors = subset_vectors[:500]
            else:
                subset_vectors = subset_vectors[:int(round(subset_vectors_len / 2, 0))]

            ax2.scatter(
                subset_vectors[:, 0],
                subset_vectors[:, 1],
                subset_vectors[:, 2],
                label=subset_name.lower().title(),  # Add label for the legend
                alpha=0.3,
            )

        # Target Product & Recommeded Products
        for idx, (subset_name, subset_vectors) in enumerate(recommedation_vectors.items()):
            subset_vectors = np.array(subset_vectors)

            marker = 'x'
            if subset_name == 'Target Product':
                marker = 'o'

            ax2.scatter(
                subset_vectors[:, 0],
                subset_vectors[:, 1],
                subset_vectors[:, 2],
                label=subset_name,  # Add label for the legend
                alpha=1,
                marker=marker,
                color='black',
                linewidths=3,
                s=300,
            )

        ax2.set_title("3D Visualization of Target Product & Recommended Products", fontsize=40)
        ax2.tick_params(axis='x', labelsize=15)
        ax2.tick_params(axis='y', labelsize=15)
        ax2.tick_params(axis='z', labelsize=15)
        ax2.legend(fontsize=15)

        plt.tight_layout()
        plt.show()

    def recommendations_info(self):
        pattern = re.compile(r"\*\*(.*)\*\*")
        ids = [x.split()[-1] for x in pattern.findall(self.recommendations)]

        recs_info = {}
        for v in self.metadata.values():
            curr_id = v.metadata['id']
            if curr_id in ids:
                recs_info.setdefault(curr_id, '')
                recs_info[curr_id] = v.metadata['text']

        print(f'[*] Recommended Products Information:')
        for v in recs_info.values():
            print(f'{v}\n')
            

if __name__ == '__main__':
    def retrieve_product_information(df, query_value):
        product_index = df.index[df['PRODUCT_ID'] == query_value].tolist()[0]
        full_text = df.loc[product_index, 'TEXT']
        product_id = df.loc[product_index, 'PRODUCT_ID']
        print(f'[*] Retrieved product full content:\n{full_text}')
    
        return df.loc[product_index, 'DESCRIPTION'], full_text, product_id
    
    random.seed(time())
    formatted_df = pd.read_csv('trainData/amazon_products.train.formatted.csv')
    random_product_id = random.choice(formatted_df['PRODUCT_ID'])
    
    test_description, full_text, target_id = retrieve_product_information(formatted_df, random_product_id)
    
    '''RecSys'''
    recSys = RecSys(
        model='Llama3.2',
        # model='Qwen2.5',
        vector_db='./Vector_DB',
        access_token='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',
        product_description=test_description,
    )
    
    recommedations, retrieved_docs_text, KNOWLEDGE_VECTOR_DATABASE = recSys.recommend()
    
    print(f'Recommendations:\n{recommedations}\n\n')
    
    '''Interpreter'''
    recInterpreter = RecInterpreter(
        KNOWLEDGE_VECTOR_DATABASE=KNOWLEDGE_VECTOR_DATABASE,
        recommendations=recommedations,
        target_id=target_id,
    )
    
    recInterpreter.vis_3d()
    recInterpreter.recommendations_info()
