# This file is to smoke test file to check whether the Ollama is working correctly.
# It is not part of the main application logic.

from langchain_ollama import ChatOllama
llm = ChatOllama(model="mistral:7b")
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
# 4. set OPENAI_API_KEY in .env variable ( This may not be required for Ollama, but just in case you want to test OpenAI also)
# $env:OPENAI_API_KEY="your_api_key_here" 
# 5. Check the Open AI API is set correctly
# echo $OPENAI_API_KEY  > would show the saved API key (Not showing all the times because of some reason)
# 6 Now run the ollama.
# open powershell seperately from run window > – Windows +R > powershell
# 6.a Type ollama > run > shows the commands
# 6.b Type ollama run mistral:7b  (this will start the ollama server)
# 6.c Type a sample prompt like "What is machine learning?" to check whether it is working fine.
# 7. Now run the file (make sure ollama is running before the runnning this command)
# python .\src\agents\z_test_ollama.py