import os
import joblib
import pandas as pd
import gradio as gr


# =====================================
# Load Trained Model
# =====================================

MODEL_PATH = "Salary_Prediction_Model.pkl"

try:
    model = joblib.load(MODEL_PATH)

    try:
        MODEL_FEATURES = list(model.feature_names_in_)
    except:
        MODEL_FEATURES = []

except Exception as e:
    raise Exception(
        f"❌ Model loading failed.\n"
        f"Make sure {MODEL_PATH} is in the same folder.\n\n{e}"
    )


print("Model Features:")
print(MODEL_FEATURES)



# =====================================
# Prediction Function
# =====================================

def predict_salary(age, experience, education):

    try:

        # Create input dictionary
        input_data = {}


        # Add numerical values
        input_data["Age"] = float(age)
        input_data["Years of Experience"] = float(experience)



        # Default all model columns to zero
        for feature in MODEL_FEATURES:

            if feature not in input_data:
                input_data[feature] = 0



        # Education encoding
        education_columns = {
            "Bachelor's": "Education Level_Bachelor's",
            "Bachelor's Degree": "Education Level_Bachelor's Degree",
            "High School": "Education Level_High School",
            "Master's": "Education Level_Master's",
            "Master's Degree": "Education Level_Master's Degree",
            "PhD": "Education Level_PhD",
            "Other": "Education Level_Other"
        }


        selected_column = education_columns.get(
            education,
            "Education Level_Other"
        )


        if selected_column in input_data:
            input_data[selected_column] = 1



        # Create dataframe in exact model order
        df = pd.DataFrame(
            [input_data]
        )


        df = df[MODEL_FEATURES]



        prediction = model.predict(df)[0]


        return f"💰 Predicted Salary: ₹ {prediction:,.2f}"



    except Exception as e:

        return f"❌ Error: {str(e)}"




# =====================================
# CSS
# =====================================

css = """

.gradio-container{

    background:#eef3f8;

}


.main-box{

    background:white;

    padding:30px;

    border-radius:20px;

    box-shadow:0px 8px 25px rgba(0,0,0,0.15);

}


/* Make all text black */

.main-box *{

    color:black !important;

}


button{

    font-weight:bold !important;

}


footer{

    display:none;

}

"""



# =====================================
# Gradio UI
# =====================================

with gr.Blocks(
    css=css,
    title="Salary Prediction"
) as demo:


    with gr.Column(
        elem_classes="main-box"
    ):


        gr.Markdown(
            """
            # 💼 Salary Prediction using Machine Learning

            Predict salary based on age, experience and education level.
            """
        )


        gr.Markdown("---")



        with gr.Row():


            # Input Section

            with gr.Column(scale=2):


                gr.Markdown(
                    "## 📥 Enter Details"
                )


                age = gr.Number(
                    label="Age",
                    value=25
                )


                experience = gr.Number(
                    label="Years of Experience",
                    value=2
                )


                education = gr.Dropdown(

                    choices=[
                        "Bachelor's",
                        "Bachelor's Degree",
                        "High School",
                        "Master's",
                        "Master's Degree",
                        "PhD",
                        "Other"
                    ],

                    value="Bachelor's",

                    label="Education Level"
                )


                predict_btn = gr.Button(
                    "🚀 Predict Salary"
                )


                output = gr.Textbox(
                    label="Prediction Result"
                )



            # Developer Section

            with gr.Column(scale=1):


                gr.Markdown(
                    """
                    ## 👨‍💻 Developer Details


                    **Name:**  
                    Manik Jindal


                    **College:**  
                    Panipat Institute of Engineering and Technology


                    **Project:**  
                    Salary Prediction using Machine Learning


                    ## 🛠 Technology Used


                    - Python
                    - Pandas
                    - Scikit-Learn
                    - Regression Model
                    - Joblib
                    - Gradio


                    ## 📌 About Project


                    This application predicts employee salary
                    using:


                    👤 Age  
                    💼 Years of Experience  
                    🎓 Education Level  


                    The prediction is generated using a trained
                    Machine Learning model.
                    """
                )



        predict_btn.click(

            fn=predict_salary,

            inputs=[
                age,
                experience,
                education
            ],

            outputs=output

        )



# =====================================
# Launch
# =====================================

if __name__ == "__main__":

    demo.launch(

        server_name="0.0.0.0",

        server_port=int(
            os.environ.get(
                "PORT",
                7860
            )
        )

    )
