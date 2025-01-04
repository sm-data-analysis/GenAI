from langchain_openai import OpenAI, ChatOpenAI
from dataclasses import dataclass, field
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from typing import List, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
import re
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from operator import itemgetter

@dataclass
class MCQGenerator:
    api_key: str
    model: str = 'gpt-4o-mini'
    temperature: float = 0.3
    max_retries: int = 10
    llm: OpenAI = field(init=False)
    prompt: PromptTemplate = field(init=False)
    output_parser: StructuredOutputParser = field(init=False)
    

    def __post_init__(self):
        self.llm = ChatOpenAI(api_key=self.api_key, model=self.model, temperature=self.temperature)
        self.setup_parser()

    def setup_parser(self):
        response_schemas = [
            ResponseSchema(name="question", description="The multiple choice question"),
            ResponseSchema(name="options", description="A python list of 4 options for the question. Separated by comma(,)"),
            ResponseSchema(name="answer", description="The correct option (A, B, C, or D)"),
            ResponseSchema(name="explanation", description="Explanation of why the answer is correct")
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        
        self.prompt = PromptTemplate(
            template = (
                "Generate a unique multiple-choice question about {topic} with an initial complexity level of {complexity}.\n"
                "Question requirements:\n"
                "1. Adjust the difficulty based on the complexity level (1-5, where 1 is very simple and 5 is very complex).\n"
                "2. For lower complexity (1-2), focus on basic facts, definitions, or simple concepts.\n"
                "3. For higher complexity (4-5), include more advanced concepts, require analysis, or combine multiple ideas.\n"
                "4. Review this history of previous questions:\n{history}\n"
                "5. If your generated question is similar to any in the history, INCREASE THE COMPLEXITY LEVEL BY 1 (up to a maximum of 5) and generate a new, more advanced question.\n"
                "6. Repeat step 5 until you generate a question that is significantly different from all questions in the history.\n"
                "7. Ensure the final question explores aspects or applications of the topic not covered in previous questions.\n"
                "8. Make all options plausible, but only one should be correct.\n"
                "{format_instructions}\n"
                "Formatting rules:\n"
                "1. Present the options as a Python list: ['A. option', 'B. option', 'C. option', 'D. option']\n"
                "2. The answer must be a single letter: 'A', 'B', 'C', or 'D'.\n"
                "3. Provide a clear explanation for the correct answer, matching the final complexity level.\n"
                "4. Include the final complexity level (1-5) used to generate the question in your response.\n"
                "5. Strictly adhere to the specified output format.\n"
                "6. Ensure the JSON is valid and can be parsed by a standard JSON parser.\n"
                "\nBefore submitting, double-check that your question is not repeating or closely resembling any in the history."
            ),
            input_variables=['topic', 'complexity', 'history'],
            partial_variables={'format_instructions': self.output_parser.get_format_instructions()}
        ) 

    def generate_mcq(self, topic: str, complexity: int, history: List[str]) -> dict:
        retries = 0
        while retries < self.max_retries:
            try:
                formatted_history = "\n".join([f"- {q}" for q in history])
                _input = self.prompt.format(topic=topic, complexity=complexity, history=formatted_history)
                output = self.llm.invoke(_input)
                
                if isinstance(output, AIMessage):
                    content = output.content
                    # print(content)
                elif isinstance(output, list) and len(output) > 0 and isinstance(output[0], AIMessage):
                    content = output[0].content
                else:
                    raise ValueError("Unexpected output format from ChatOpenAI")
                # print(content)
                try:
                    result = self.output_parser.parse(content)
                except:
                        
                    def preprocess_json(json_string):
                        # Replace single quotes with double quotes, but only for strings
                        return re.sub(r"'([^']*)'", r'"\1"', json_string)

                    preprocessed_content = preprocess_json(content)
                    result = self.output_parser.parse(preprocessed_content)
                # print(content)
                if isinstance(result.get('options'), list) and len(result['options']) == 4 and result['question'].lower() not in history:
                    return result

                print("Error: Options are not in the correct format. Retrying...")
                retries += 1

            except Exception as e:
                print(f"Parsing failed with error: {e}. Retrying... ({retries + 1}/{self.max_retries})")
                retries += 1

        print("Failed to generate a valid MCQ after maximum retries. Click next and try again.")
        return {
            'question': 'Could not generate a question. Press Next to try Again.',
            'options': ['Option A', 'Option B', 'Option C', 'Option D'],
            'answer': 'A',
            'explanation': 'No explanation available.'
        }

class ChatSessionHandler:
    def __init__(self):
        self.store = {}
        self.session_id = None
        self.mcq_details = None
        self.chain = None
        
    def create_model(self, api_key, temp):
        self.llm = ChatOpenAI(api_key=api_key, model='gpt-4o-mini', temperature=temp)

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Retrieves the chat history for a given session. If the session does not exist, it creates a new one.
        """
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def create_prompt(self, mcq_details: str) -> ChatPromptTemplate:
        """
        Creates a chat prompt template with provided MCQ details.
        """
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""
                    You are an assistant discussing the following multiple-choice question with the user.
                    {mcq_details}

                    Your task:
                    1. Respond to user questions about this MCQ, its topic, its options, or closely related information.
                    2. Provide concise but informative answers, including additional relevant facts when appropriate.
                    3. If a question is somewhat related to the MCQ topic, even if not directly about the question itself, provide a brief answer and then guide the conversation back to the MCQ.
                    4. Only if a question is completely unrelated to the MCQ topic, respond with: "That's not directly related to the question about [brief topic]. Would you like to know more about [specific aspect of the MCQ]?"
                    5. You can provide general information about locations, dates, or contexts related to the MCQ topic if asked.
                    6. If you don't have specific information about a related aspect, it's okay to say so and offer what you do know about the topic.

                    Remember, your primary focus is on the MCQ and related information, but be flexible in addressing closely associated topics to enhance understanding.
                    """
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

    def initialize_session(self, session_id: str, mcq_details: str):
        """
        Initializes the session with the provided session_id and mcq_details.
        Should be called only once at the start.
        """
        self.session_id = session_id
        self.mcq_details = mcq_details
        prompt = self.create_prompt(mcq_details)
        trimmer = trim_messages(
            max_tokens=300,
            strategy="last",
            token_counter=self.llm,
            include_system=True,
            allow_partial=False,
            start_on="human",
        )

        self.chain = (
            RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
            | prompt
            | self.llm
        )
        

    def invoke_response(self, messages):
        """
        Invokes the response using the stored session history, prompt chain, and given messages.
        """
        if not self.chain or not self.session_id:
            raise ValueError("Session not initialized. Please call 'initialize_session' first.")

        with_message_history = RunnableWithMessageHistory(self.chain, self.get_session_history, input_messages_key="messages")
        config = {"configurable": {"session_id": self.session_id}}

        response = with_message_history.invoke(
            {
                "messages": messages,
                'mcq_details': self.mcq_details  # Use stored mcq_details
            }, config=config
        )
        return response.content

   