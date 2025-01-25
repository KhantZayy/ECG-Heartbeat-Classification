import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image

def load_model():
    return tf.keras.models.load_model('ECG_CNN_images4_256.h5')

def predict(input_image):
    model = load_model()
    input_image = Image.open(input_image).convert('RGB')
    image_resize = input_image.resize((256, 256))
    image_rescale = np.array(image_resize) / 255.0
    image_final = np.expand_dims(image_rescale, axis = 0)

    prediction = model.predict(image_final)
    predict_index = [np.argmax(i) for i in prediction]
    st.write(f'Confidence: {np.max(prediction) * 100:.2f} %')
    return predict_index[0]

st.sidebar.title(':red[Dashboard]')
app_mode = st.sidebar.selectbox('Choose an option', ['Home', 'About', 'Know Your Heart Rhythm'], placeholder= 'Select One')

if app_mode == 'Home':
   st.title('ECG Heartbeat Classification Homepage')
   st.image('heartbeat-icon-ecg-pathology-trace-vector-27113726.jpg', )
   image_input = st.file_uploader('Upload Image')
   heartbeat_types = [':blue[Fusion of ventricular and normal beat]', ':red[Myocardial Infarction]', ':green[Normal beat]', ':blue[Unclassifiable beat]', ':green[Supraventricular premature beat]', ':blue[Premature ventricular contraction]']
   if image_input:
       image = Image.open(image_input)
       st.image(image, width= 256, caption='Uploaded Image')
   if st.button('Classify'):
        if image_input:
            predict_index = predict(image_input)
            st.write(heartbeat_types[predict_index])
        else:
            st.write('Upload an image first.')

   st.divider()
   st.subheader(":green[Feedback]")

   st.write("Please provide your feedback:")

   col1, col2 = st.columns(2)

   with col1:
        if st.button("👍 Thumbs Up"):
            st.success("Thank you for your positive feedback! 😊")

   with col2:
        if st.button("👎 Thumbs Down"):
            st.warning("Thank you for your feedback! We'll strive to improve. 🙁")

if app_mode == 'About':
    st.subheader('ECG Heartbeat Categorization 📊')
    st.write("This notebook involves the making of machine learning models to classify the given data of heartbeat ECG into different classes. We'll undergo machine learning processes to classify them. As given in the dataset, we are provided with 6 different classes of heartbeat as ")
    st.code("1. Normal beat\n2. Supraventricular premature beat\n3. Premature ventricular contraction\n4. Fusion of ventricular & normal beat\n5. Unclassifiable beat\n6. Myocardial infarction")
    
if app_mode == 'Know Your Heart Rhythm':
    st.title('Heartbeat Types Overview 📊')
    st.subheader('1. :blue[Normal beat.]')
    st.image('normal-heartbeat.jpg', caption='Normal Heartbeat Image')
    st.markdown(':red[What is a Normal Heartbeat?]')
    st.write('''A normal heartbeat is the regular rhythm at which your heart pumps blood to supply oxygen and nutrients to your body. In healthy adults, the normal heart rate typically ranges from 60 to 100 beats per minute (BPM) when at rest. This rate can vary based on factors like age, fitness level, and emotional state.''')

    st.write('Characteristics of a Normal Heartbeat:')

    st.code('''
	1. Rhythmic: The heart beats in a steady, regular pattern without skipping or adding extra beats.
	2. Efficient: Each beat ensures blood is efficiently pumped to the body and lungs.
	3. Adaptable: The heart rate can increase during exercise or stress and slow down during rest or relaxation, maintaining balance.''')

    st.markdown(':red[Why is it Important?]')

    st.write('''A normal heartbeat is essential for maintaining good health and ensuring all body tissues receive enough oxygenated blood. Any irregularities could indicate underlying heart conditions that may require attention.

If you notice symptoms like palpitations, dizziness, or fatigue despite being classified as having a normal heartbeat, it’s always a good idea to consult a healthcare professional.
''')
    st.divider()
    st.subheader('2. :blue[Supraventricular premature beat]')
    st.image('big_6573c63b1001a2.21469319.jpg', caption= 'Supraventricular premature heartbeat Image')
    st.markdown(':red[What is a Supraventricular Premature Beat (SPB)?]')
    st.write('''A Supraventricular Premature Beat (SPB), also known as a premature atrial contraction (PAC), is an early heartbeat that originates in the upper chambers of the heart (the atria). It’s a common type of irregular heartbeat that is usually harmless.''')
    
    st.write('Characteristics of SPB:')
    
    st.code('''
    1. Extra Beats: SPBs occur when the atria send an extra signal, causing the heart to beat earlier than expected.
2. Reset Rhythm: After the early beat, the heart typically pauses briefly to reset, which might feel like a “skipped beat.”
3. Mild Symptoms: SPBs can sometimes cause a fluttering sensation, mild palpitations, or no symptoms at all.
''')
    st.markdown(':red[Why Does It Happen?]')
    st.code('''	• Stress or Fatigue
	• Caffeine, Alcohol, or Smoking
	• Electrolyte Imbalances
	• Underlying Heart Conditions
''')
    st.markdown(':red[Is It Dangerous?]')
    st.write('In most cases, SPBs are harmless and do not require treatment. However, frequent SPBs may warrant further evaluation, especially if accompanied by dizziness, chest pain, or shortness of breath.')
    st.divider()

    st.subheader('3. :blue[Premature ventricular contraction]')
    st.image('big_6573c7c0177fa2.46479097.jpg', caption= 'Premature ventricular contraction Image')
    st.markdown(':red[What is a Premature Ventricular Contraction (PVC)?]')
    st.write('''A Premature Ventricular Contraction (PVC) is an extra heartbeat that begins in the heart’s lower chambers (the ventricles). It occurs earlier than the next expected regular heartbeat and is one of the most common types of irregular heart rhythms.
''')
    st.write('Characteristics of PVC:')
    st.code('''1. Extra or Skipped Beats: PVCs create an irregular rhythm by causing the heart to beat too early. This may feel like a “skipped” or “extra” beat.
2. Brief Pause: After the early beat, the heart typically pauses momentarily to return to its normal rhythm.
3. Mild to No Symptoms: Some people may feel fluttering or a strong “thump” in the chest, while others might not notice it at all.
''')
    st.markdown(':red[Why Does It Happen?]')
    st.code('''	• Stress or Anxiety
	• Caffeine, Alcohol, or Smoking
	• Exercise or Fatigue
	• Electrolyte Imbalances
	• Heart Disease or High Blood Pressure
''')
    st.markdown(':red[Is It Dangerous?]')
    st.write('For most people, occasional PVCs are harmless and do not require treatment. However, frequent PVCs or PVCs associated with symptoms like chest pain, dizziness, or fainting may indicate an underlying heart condition and should be evaluated by a healthcare professional.')
    st.divider()

    st.subheader('4. :blue[Fusion of ventricular & normal beat]')
    st.image('Capture-and-fusion-beats.png', caption= 'Fusion Beat Image')
    st.markdown(':red[What is a Fusion Beat?]')
    st.write('A Fusion Beat occurs when a normal heartbeat (originating from the atria) and a premature ventricular contraction (PVC) happen at nearly the same time. This causes the two signals to “merge,” producing a hybrid or combined heartbeat.')
    st.write('Characteristics of Fusion Beats:')
    st.code('''1. Merged Signals: Fusion beats are a combination of a normal sinus beat and a ventricular beat, resulting in a unique appearance on an ECG (electrocardiogram).
2. Intermediate Shape: The beat’s morphology (shape) is a mix of both normal and ventricular beats, neither fully normal nor entirely abnormal.
3. Rare Sensation: Most people do not feel fusion beats, but they may be detected during cardiac monitoring.
''')
    st.markdown(':red[Why Do Fusion Beats Happen?]')
    st.write('Fusion beats occur when:')
    st.code('''	• A normal electrical signal from the upper chambers (atria) coincides with an abnormal signal from the lower chambers (ventricles).
• This can happen in people with underlying conditions like premature ventricular contractions (PVCs), especially when the timing overlaps.
''')
    st.markdown(':red[Is It Dangerous?]')
    st.write('In most cases, fusion beats are not dangerous and are simply a sign of occasional electrical “miscommunication” in the heart. However, if fusion beats are frequent or occur alongside symptoms like dizziness, chest pain, or fainting, they may indicate an underlying heart condition that should be evaluated by a healthcare provider.')
    st.divider()

    st.subheader('5. :blue[Unclassifiable beat]')
    st.markdown(':red[What is an Unclassifiable Beat?]')
    st.write('''An Unclassifiable Beat refers to a heartbeat that does not fit into any standard category of heart rhythms, such as normal beats, premature beats, or specific arrhythmias. It may occur due to irregular or unusual electrical activity in the heart that makes it challenging to interpret.
''')
    st.write('Characteristics of Unclassifiable Beats:')
    st.code('''1. Irregular Pattern: The beat does not match typical patterns seen in normal or abnormal heart rhythms.
2. Transient: These beats may occur sporadically and are not always persistent.
3. ECG Challenges: On an electrocardiogram (ECG), the beat’s shape and timing might not conform to standard classifications.
''')
    st.markdown(':red[Why Does It Happen?]')
    st.code('''	• Noise or artifact during ECG recording (e.g., body movement or interference).
• Rare or complex arrhythmias that do not follow typical patterns.
• Errors in detection by monitoring equipment.
''')
    st.markdown(':red[Is It Dangerous?]')
    st.write('In many cases, an unclassifiable beat may simply be due to technical issues or an isolated occurrence without clinical significance. However, frequent unclassifiable beats or their presence alongside symptoms such as dizziness, fainting, or chest pain may require further evaluation to rule out underlying conditions.')
    st.divider()

    st.subheader('6. :blue[Myocardial infarction]')
    st.image('360_F_618693958_Nc6XTz1qNGHqDLjlmJZJjEDwSY1GN8RR.jpg', caption= 'Myocardial infarction Image')
    st.markdown(':red[What is Myocardial Infarction?]')
    st.write('Myocardial Infarction (MI), commonly known as a heart attack, occurs when blood flow to a part of the heart is blocked, leading to damage or death of heart muscle tissue. This blockage is typically caused by a buildup of fatty deposits (plaque) in the coronary arteries, which supply blood to the heart.')
    st.write('Characteristics of Myocardial Infarction:')
    st.code('''1. Symptoms: Common symptoms include chest pain or discomfort (often described as pressure or squeezing), shortness of breath, sweating, nausea, lightheadedness, or pain that radiates to the arms, neck, jaw, or back.
2. Emergency Condition: MI is a medical emergency that requires immediate attention. Time is critical, as prolonged lack of blood flow can lead to severe heart damage or death.
3. Diagnosis: An MI is diagnosed using various tests, including an electrocardiogram (ECG) and blood tests to check for markers of heart damage.
''')
    st.write('Risk Factors:')
    st.code('''	• High Blood Pressure
	• High Cholesterol Levels
	• Smoking
	• Diabetes
	• Obesity
	• Sedentary Lifestyle
	• Family History of Heart Disease''')
    st.subheader(':green[Treatment:]')
    st.write('Immediate treatment for myocardial infarction may include medications (like aspirin or thrombolytics), procedures to open blocked arteries (such as angioplasty and stenting), or surgery (such as coronary artery bypass grafting). Long-term management often involves lifestyle changes and medications to reduce the risk of future heart attacks.')
    st.divider()

    st.text('Thank you for taking the time to learn about heart health.\n\tUnderstanding your heart’s rhythms and conditions \n\t\tis essential for maintaining overall wellness.💕💐')
    if st.button('Wishing you all the best!🩶'):
        st.balloons()
    