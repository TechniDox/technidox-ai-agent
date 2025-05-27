import os
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature=0.3, openai_api_key=os.getenv("OPENAI_API_KEY"))

def load_prompt(file_path):
    with open(file_path, "r") as file:
        return file.read()

def inclusive_review(doc_text):
    prompt_template = PromptTemplate.from_template(
        load_prompt("prompts/inclusive_review.txt")
    )
    return llm.invoke(prompt_template.format(doc_text=doc_text))

def onboarding_guide(role, experience_level, stack):
    prompt_template = PromptTemplate.from_template(
        load_prompt("prompts/onboarding_helper.txt")
    )
    return llm.invoke(
        prompt_template.format(
            role=role,
            experience_level=experience_level,
            stack=stack
        )
    )

if __name__ == "__main__":
    # Inclusive review test
    sample_doc = "Whitelist these IPs and make sure no junior breaks the config."
    print("📝 Inclusive Review Output:\n")
    print(inclusive_review(sample_doc).content)

    print("\n🚀 Onboarding Guide Output:\n")
    print(onboarding_guide("Frontend Developer", "Junior", "React, Next.js, GitHub").content)

