
import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn import *

pickle_in = open("banglore_home_prices_model.pickle","rb")
model = pickle.load(pickle_in)
loc=['1st Block Jayanagar', '1st Phase JP Nagar',
        '2nd Phase Judicial Layout', '2nd Stage Nagarbhavi',
        '5th Block Hbr Layout', '5th Phase JP Nagar', '6th Phase JP Nagar',
        '7th Phase JP Nagar', '8th Phase JP Nagar', '9th Phase JP Nagar',
        'AECS Layout', 'Abbigere', 'Akshaya Nagar', 'Ambalipura',
        'Ambedkar Nagar', 'Amruthahalli', 'Anandapura', 'Ananth Nagar',
        'Anekal', 'Anjanapura', 'Ardendale', 'Arekere', 'Attibele',
        'BEML Layout', 'BTM 2nd Stage', 'BTM Layout', 'Babusapalaya',
        'Badavala Nagar', 'Balagere', 'Banashankari',
        'Banashankari Stage II', 'Banashankari Stage III',
        'Banashankari Stage V', 'Banashankari Stage VI', 'Banaswadi',
        'Banjara Layout', 'Bannerghatta', 'Bannerghatta Road',
        'Basavangudi', 'Basaveshwara Nagar', 'Battarahalli', 'Begur',
        'Begur Road', 'Bellandur', 'Benson Town', 'Bharathi Nagar',
        'Bhoganhalli', 'Billekahalli', 'Binny Pete', 'Bisuvanahalli',
        'Bommanahalli', 'Bommasandra', 'Bommasandra Industrial Area',
        'Bommenahalli', 'Brookefield', 'Budigere', 'CV Raman Nagar',
        'Chamrajpet', 'Chandapura', 'Channasandra', 'Chikka Tirupathi',
        'Chikkabanavar', 'Chikkalasandra', 'Choodasandra', 'Cooke Town',
        'Cox Town', 'Cunningham Road', 'Dasanapura', 'Dasarahalli',
        'Devanahalli', 'Devarachikkanahalli', 'Dodda Nekkundi',
        'Doddaballapur', 'Doddakallasandra', 'Doddathoguru', 'Domlur',
        'Dommasandra', 'EPIP Zone', 'Electronic City',
        'Electronic City Phase II', 'Electronics City Phase 1',
        'Frazer Town', 'GM Palaya', 'Garudachar Palya', 'Giri Nagar',
        'Gollarapalya Hosahalli', 'Gottigere', 'Green Glen Layout',
        'Gubbalala', 'Gunjur', 'HAL 2nd Stage', 'HBR Layout',
        'HRBR Layout', 'HSR Layout', 'Haralur Road', 'Harlur', 'Hebbal',
        'Hebbal Kempapura', 'Hegde Nagar', 'Hennur', 'Hennur Road',
        'Hoodi', 'Horamavu Agara', 'Horamavu Banaswadi', 'Hormavu',
        'Hosa Road', 'Hosakerehalli', 'Hoskote', 'Hosur Road', 'Hulimavu',
        'ISRO Layout', 'ITPL', 'Iblur Village', 'Indira Nagar', 'JP Nagar',
        'Jakkur', 'Jalahalli', 'Jalahalli East', 'Jigani',
        'Judicial Layout', 'KR Puram', 'Kadubeesanahalli', 'Kadugodi',
        'Kaggadasapura', 'Kaggalipura', 'Kaikondrahalli',
        'Kalena Agrahara', 'Kalyan nagar', 'Kambipura', 'Kammanahalli',
        'Kammasandra', 'Kanakapura', 'Kanakpura Road', 'Kannamangala',
        'Karuna Nagar', 'Kasavanhalli', 'Kasturi Nagar', 'Kathriguppe',
        'Kaval Byrasandra', 'Kenchenahalli', 'Kengeri',
        'Kengeri Satellite Town', 'Kereguddadahalli', 'Kodichikkanahalli',
        'Kodigehaali', 'Kodigehalli', 'Kodihalli', 'Kogilu', 'Konanakunte',
        'Koramangala', 'Kothannur', 'Kothanur', 'Kudlu', 'Kudlu Gate',
        'Kumaraswami Layout', 'Kundalahalli', 'LB Shastri Nagar',
        'Laggere', 'Lakshminarayana Pura', 'Lingadheeranahalli',
        'Magadi Road', 'Mahadevpura', 'Mahalakshmi Layout', 'Mallasandra',
        'Malleshpalya', 'Malleshwaram', 'Marathahalli', 'Margondanahalli',
        'Marsur', 'Mico Layout', 'Munnekollal', 'Murugeshpalya',
        'Mysore Road', 'NGR Layout', 'NRI Layout', 'Nagarbhavi',
        'Nagasandra', 'Nagavara', 'Nagavarapalya', 'Narayanapura',
        'Neeladri Nagar', 'Nehru Nagar', 'OMBR Layout', 'Old Airport Road',
        'Old Madras Road', 'Padmanabhanagar', 'Pai Layout', 'Panathur',
        'Parappana Agrahara', 'Pattandur Agrahara', 'Poorna Pragna Layout',
        'Prithvi Layout', 'R.T. Nagar', 'Rachenahalli',
        'Raja Rajeshwari Nagar', 'Rajaji Nagar', 'Rajiv Nagar',
        'Ramagondanahalli', 'Ramamurthy Nagar', 'Rayasandra',
        'Sahakara Nagar', 'Sanjay nagar', 'Sarakki Nagar', 'Sarjapur',
        'Sarjapur  Road', 'Sarjapura - Attibele Road',
        'Sector 2 HSR Layout', 'Sector 7 HSR Layout', 'Seegehalli',
        'Shampura', 'Shivaji Nagar', 'Singasandra', 'Somasundara Palya',
        'Sompura', 'Sonnenahalli', 'Subramanyapura', 'Sultan Palaya',
        'TC Palaya', 'Talaghattapura', 'Thanisandra', 'Thigalarapalya',
        'Thubarahalli', 'Thyagaraja Nagar', 'Tindlu', 'Tumkur Road',
        'Ulsoor', 'Uttarahalli', 'Varthur Road', 'Vasanthapura',
        'Vidyaranyapura', 'Vijayanagar', 'Vishveshwarya Layout',
        'Vishwapriya Layout', 'Vittasandra', 'Whitefield',
        'Yelachenahalli', 'Yelahanka', 'Yelahanka New Town', 'Yelenahalli',
        'Yeshwanthpur', 'other']

def main():

    st.title("Bengaluru House Price Prediction")
    st.subheader("Can be used to predict prices of available homes in Bengaluru")
    st.write("Fill the information below to predict")

    location = st.selectbox("Select Location", loc)

    bhk = st.slider("Select number of BHK", 0, 20)

    bathroom = st.slider("Select number of bathrooms", 0, 20)

    total_sqft = st.number_input("Select Total Sqft", 500, 5000)

    balcony = st.checkbox("Balconey")

    if st.button("Predict Price"):
        predict_price(location,total_sqft,bathroom,bhk)
    
def predict_price(location,total_sqft,bathroom,bhk):    
    X = np.zeros(3+len(loc))
    X[0]=total_sqft
    X[1]=bathroom
    X[2]=bhk
    X[3+loc.index(location)]=1
    st.success("The House Price is Rs.{} lakhs".format(round(model.predict([X])[0]),3))

main()