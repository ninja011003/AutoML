# About AutoML
Automated Machine Learning, also known as, AutoML is an application which in simple words, automate the ML training process. 
All the user required is the data, regardless of whether it is preprocessed or not, 
and the target variable they want. The required output for this is given to the user, without any significant effort on their part.
Using streamlit, an interface is provided for the users to interact with our service

# It's Features
Fully automated preprocessing, feature selection and model selection
Application of Genetic Algorithm to select best features
Using Mult-Threading Techniques to select best models
Simple User Interface

# How to run?
The service is already hosted in this link - [Automated ML](https://alphabyte-automl.streamlit.app/)
But if the need arises to run it offline, then here is the procedure

> [!NOTE]
> VS Code is used in this case

### To run offline
1) Python Extension in VS Code is required, which can be installed within VS Code, by presing Ctrl+Shift+X an extension sidebar appears, where you can search for python and install it
2) After opening the project folder, in the menu bar, press Ctrl+Shift+P, this opens Command Palette, search for Create Environment and select it.
3) Select venv and then select the python interpreter in your system, then make sure the requirements.txt is selected to install required packages.
4) The venv will be activated automatically once everything is done, in VS Terminal located at the bottom, run the command
   ```
   streamlit run streamlit_app.py
   ```
5) A new browser window will open to run the project in local.
