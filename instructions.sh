# install dependencies first
pip3 install -r requirements.txt

# install ollama from the official website - https://ollama.com/
# then pull one of the supported models
ollama pull phi2

# run the light version of the app - it answers the question of who I am in a funny way
python3 ragme_light.py

# run the full version of the app - it answers the question of where I have worked in a normal way.
python3 ragme_advanced.py