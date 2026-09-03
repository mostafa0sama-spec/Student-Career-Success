import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title='Student Career Success Predictor', page_icon='🎓', layout='wide')
st.title('🎓 Student Career Success Predictor')
st.caption('Two-stage academic ML/DL demo: placement prediction + starting-salary estimation.')
st.warning('Educational demonstration only. Do not use this as an automated hiring/admissions or compensation decision system.')

placement_config_path = Path('deployment_choice.json')
if not placement_config_path.exists():
    st.error('Run the notebook through the deployment-export sections first.')
    st.stop()

deployment = json.load(open(placement_config_path))

if deployment['type'] == 'sklearn':
    placement_model = joblib.load(deployment['model_file'])
else:
    import tensorflow as tf
    placement_model = tf.keras.models.load_model(deployment['model_file'])
    placement_preprocessor = joblib.load(deployment['preprocessor_file'])

salary_model = None
salary_info = None
salary_config_path = Path('salary_model_info.json')

if salary_config_path.exists():
    salary_info = json.load(open(salary_config_path))
    salary_model_path = Path(salary_info['model_file'])
    if salary_model_path.exists():
        salary_model = joblib.load(salary_model_path)

with st.form('student'):
    a, b, c = st.columns(3)

    with a:
        Age = st.slider('Age', 18, 30, 21)
        Gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
        University_Year = st.selectbox('University Year', ['Freshman', 'Sophomore', 'Junior', 'Senior'])
        Major = st.selectbox('Major', ['Computer Science', 'Software Engineering', 'Artificial Intelligence', 'Data Science', 'Cybersecurity', 'Information Technology', 'Business Analytics', 'Electrical Engineering'])
        Attendance_Percentage = st.slider('Attendance %', 50, 100, 82)
        Study_Hours_Per_Week = st.slider('Study Hours / Week', 5, 45, 21)
        CGPA = st.slider('CGPA', 2.0, 4.0, 3.10, 0.01)
        Academic_Performance = st.selectbox('Academic Performance', ['Poor', 'Average', 'Good', 'Excellent'])

    with b:
        Programming_Skill = st.slider('Programming Skill', 1, 10, 7)
        Projects_Completed = st.slider('Projects Completed', 0, 15, 8)
        Certifications = st.slider('Certifications', 0, 8, 2)
        Hackathons = st.slider('Hackathons', 0, 10, 3)
        GitHub_Profile = st.selectbox('GitHub Profile', ['Yes', 'No'])
        Internships = st.slider('Internships', 0, 5, 4)
        Leadership_Experience = st.selectbox('Leadership Experience', ['Yes', 'No'])
        LinkedIn_Profile = st.selectbox('LinkedIn Profile', ['Yes', 'No'])

    with c:
        Resume_Score = st.slider('Resume Score', 44, 100, 95)
        Communication_Skills = st.slider('Communication Skills', 3, 10, 8)
        Teamwork = st.slider('Teamwork', 2, 10, 7)
        Problem_Solving = st.slider('Problem Solving', 1, 10, 7)
        English_Proficiency = st.selectbox('English Proficiency', ['Basic', 'Intermediate', 'Advanced'])
        Interview_Score = st.slider('Interview Score', 17, 100, 78)
        Employability_Score = st.slider('Employability Score', 82.3, 279.05, 214.0, 0.05)

    submit = st.form_submit_button('Predict Career Outcome')

if submit:
    row = pd.DataFrame([{
        'Age': Age,
        'Gender': Gender,
        'University_Year': University_Year,
        'Major': Major,
        'Attendance_Percentage': Attendance_Percentage,
        'Study_Hours_Per_Week': Study_Hours_Per_Week,
        'CGPA': CGPA,
        'Academic_Performance': Academic_Performance,
        'Programming_Skill': Programming_Skill,
        'Projects_Completed': Projects_Completed,
        'Certifications': Certifications,
        'Hackathons': Hackathons,
        'GitHub_Profile': GitHub_Profile,
        'Internships': Internships,
        'Leadership_Experience': Leadership_Experience,
        'LinkedIn_Profile': LinkedIn_Profile,
        'Resume_Score': Resume_Score,
        'Communication_Skills': Communication_Skills,
        'Teamwork': Teamwork,
        'Problem_Solving': Problem_Solving,
        'English_Proficiency': English_Proficiency,
        'Interview_Score': Interview_Score,
        'Employability_Score': Employability_Score
    }])

    if deployment['type'] == 'sklearn':
        pred = int(placement_model.predict(row)[0])
        if hasattr(placement_model, 'predict_proba'):
            risk = float(placement_model.predict_proba(row)[0, 1])
        elif hasattr(placement_model, 'decision_function'):
            score = float(placement_model.decision_function(row)[0])
            risk = float(1 / (1 + np.exp(-score)))
        else:
            risk = float(pred)
    else:
        x = placement_preprocessor.transform(row)
        risk = float(placement_model.predict(x, verbose=0).ravel()[0])
        pred = int(risk >= 0.5)

    st.subheader('Placement Prediction')

    if pred == 1:
        st.error('Predicted class: Not Placed / At Risk')
        st.metric('Estimated Not-Placed Risk', f'{risk * 100:.1f}%')
        st.info('Starting salary is not estimated because the placement classifier predicted Not Placed.')
    else:
        st.success('Predicted class: Placed')
        st.metric('Estimated Not-Placed Risk', f'{risk * 100:.1f}%')

        st.subheader('Estimated Starting Salary')
        if salary_model is not None:
            estimated_salary = float(salary_model.predict(row)[0])
            estimated_salary = max(0.0, estimated_salary)
            st.metric('Predicted Starting Salary', f'${estimated_salary:,.0f}')

            if salary_info is not None:
                mae = salary_info.get('MAE_USD')
                model_name = salary_info.get('model_name', 'salary regression model')
                st.caption(f'Salary model: {model_name}')
                if mae is not None:
                    st.caption(f'Test MAE ≈ ${mae:,.0f}. Interpret this as a rough estimate, not an exact offer.')
        else:
            st.warning('Salary model artifacts were not found. Run the salary-regression section of the notebook first.')

    st.caption('These outputs are for an academic project and are not definitive judgments about a real student or salary offer.')
