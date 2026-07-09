# ./app/prompt/template.py
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

summary_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """You are a helpful programmer that write the java project based on the following requirements. 
The first thing you need to do is to summarize the requirements below and extract the parts about the project background and skeleton.
""",
        ),
        HumanMessagePromptTemplate.from_template("{requirements}"),
    ]
)

complete_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """You are a helpful java programmer that write the project based on the following context. 
Java is a high-level, class-based, object-oriented programming language that is designed to have as few implementation dependencies as possible.
{code_context}
"""
        ),
        HumanMessagePromptTemplate.from_template(
            template="""Complete the code and give the complete class.
{code}"""
        ),
    ]
)

complete_template_tests = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """Given the following coding task specifications, provide a java test suite with up to 10 test methods with the goal of fully testing the solutions for the coding task: <METHOD_SIGNATURE>.
Java is a high-level, class-based, object-oriented programming language that is designed to have as few implementation dependencies as possible.
{code_context}
"""
        ),
        HumanMessagePromptTemplate.from_template(
            template="""## Coding Task:
{code}
## The test file should start with the following header:
<TEST_HEADER>
"""
        ),
    ]
)

fix_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            template="""You are a helpful java programmer that give me correct code to replace the error code based on the following context(Don't modified the context code):
Java is a high-level, class-based, object-oriented programming language that is designed to have as few implementation dependencies as possible.
{code_context}
"""
        ),
        HumanMessagePromptTemplate.from_template(
            """Now give me correct code to replace the error code.
error: {error_message}
```java
{error_code}
```
```java
{error_content}
```
"""
        ),
    ]
)
