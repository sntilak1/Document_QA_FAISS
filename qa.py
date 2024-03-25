from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent

from langchain.memory import ChatMessageHistory
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

loader = TextLoader("sotu.txt",encoding="utf8")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)
retriever = db.as_retriever()

tool = create_retriever_tool(
    retriever,
    "search_state_of_union",
    "Searches and returns excerpts from the 2022 State of the Union.",
)
tools = [tool]


system_instructions = "This tool searches and returns excerpts from the 2022 State of the Union. You can ask me to search for specific topics or keywords, and I will return relevant excerpts. What would you like to know about the State of the Union?"

retriever = db.as_retriever()
prompt = ChatPromptTemplate.from_messages([
    ("system", system_instructions),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])



llm = ChatOpenAI(temperature=0)

memory = ConversationBufferMemory(return_messages=True,memory_key="chat_history")

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, memory=memory,tools=tools)

result = agent_executor.invoke({"input": "hi, im bob"})

result['output']