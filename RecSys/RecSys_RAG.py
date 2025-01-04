from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from getpass import getpass
import torch
import os

class RecSys:
    def __init__(self, model:str, vector_db:str, product_description:str, k:int = 15, use_4bit:bool = False):
        self.model = model
        self.vector_db = vector_db
        self.product_description = product_description
        self.k = k
        self.use_4bit = use_4bit
        self.device = ''
        self.access_token = ''
        self.prompt_in_chat_format = [
            {
                "role": "system",
                "content": """Using the information contained in context, give a comprehensive answer to the question.
                Respond only to the question asked, response should be concise and relevant to the question.
                Information of recommended products must be correct and matched in context, do not falsify information.
                If the answer cannot be deduced from the context, do not give an answer.
                
                Response must include product id, title, and reason for recommendation.
                Response must strictly follow the template below:
                i. - Product ID: <product id>: 
                   - Title: <title>
                   - Reason: <reason for recommendation>""",
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

        # self.access_token = getpass("Enter your access token for HuggingFace: ")
        self.access_token = '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'

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

        return recommedations, retrieved_docs_text
