# This file is to smoke test to check whether the OpenAI API is working correctly.
# It is not part of the main application logic.

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4", temperature=0)
resp = llm.invoke(["What is machine learning in 50 words!"])
print(resp.content )

# Use below code to run this file step by step:
# 1. create virtual environment - 
#  python -m venv .\venv
# 2. activate virtual environment
# .\venv\Scripts\Activate.ps1
# 3 If any error in activating virtual environment, run this command in PowerShell: 
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# 4. install dependencies one by one
# pip install langchain
# pip install langchain-openai
# pip install langchain-ollama
# pip install python-dotenv
# 4. set OPENAI_API_KEY in .env variable
# $env:OPENAI_API_KEY="your_api_key_here" 
# 5. Check the Open AI API is set correctly
# echo $OPENAI_API_KEY  > would show the saved API key (Not showing all the times because of some reason)
# 6. Now run the file
# python .\src\agents\z_test_openai.py