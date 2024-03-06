from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from app import GOOGLE_API_KEY
from app import assistant_chain
from app import system_message


def eval_expected_words(
		sys_message,
		question,
		expected_words,
		human_template="{question}",
		llm=ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7, top_p=0.85,
		                           convert_system_message_to_human=True, google_api_key=GOOGLE_API_KEY),
		output_parser=StrOutputParser()):
	assistant = assistant_chain(system_message)
	answer = assistant.invoke({"question": question})
	print(answer)

	assert any(word in answer.lower() \
	           for word in expected_words), \
		f"Expected the assistant questions to include \
    '{expected_words}', but it did not"


def evaluate_refusal(
		system_message,
		question,
		decline_response,
		human_template="{question}",
		llm=ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7, top_p=0.85,
		                           convert_system_message_to_human=True, google_api_key=GOOGLE_API_KEY),
		output_parser=StrOutputParser()):
	assistant = assistant_chain(human_template,
	                            system_message,
	                            llm,
	                            output_parser)

	answer = assistant.invoke({"question": question})
	print(answer)

	assert decline_response.lower() in answer.lower(), \
		f"Expected the bot to decline with \
    '{decline_response}' got {answer}"
